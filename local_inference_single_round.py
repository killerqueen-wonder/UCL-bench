import os
import json
import torch
import logging
import argparse
import requests
import re
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
import transformers
from transformers import set_seed

os.umask(0)
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

def list_to_dict(data_list):
    classified_dict = {}
    for item in data_list:
        task_name = item["task_name"]
        if task_name not in classified_dict:
            classified_dict[task_name] = [item]
        else:
            classified_dict[task_name].append(item)
    return classified_dict

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        with open(data_path) as f:
            raw_data = json.load(f)
        self.dataset = [item for sublist in raw_data.values() for item in sublist]  

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
    
    def collate_fn(self, batch):
        return batch

# 核心系统指令更新
NEW_SYSTEM_PROMPT = """你是一个严谨且专业的法律AI助手。你的任务是通过逐步思考用户请求并回答法律问题。回答必须基于事实，严禁编造法律条文或案例。

### 核心指令：
0. **第一步思考**: 识别用户请求的核心法律实体，关键事实，并且提出可能涉及的法律条文。
1. **判断是否需要检索**：你可以使用检索工具。如果问题基础且你非常有把握，也可以不使用工具直接作答。
2. **支持的检索工具**（两种）：
   - **法律检索**：需要确认某项罪名的具体刑期、适用条件、或者某一司法解释的原文时使用。
   - 不要试图一次性把所有关键词都搜完。每次最多只查找两项最相关的法律。
   - 先搜索最核心的概念。不要用缩写词，尽量用完整的，最有特点的，区别于其他法条的关键词。搜索关键词示例：“刑法 盗窃罪”、“最高人民法院关于适用〈民事诉讼法〉的解释 第501条”。
   - 如果你搜索了三次依然没有找到相关条文，请直接承认未找到，修改思考思路，搜索其他条文。不要编造内容。
   - **类案检索**：用来检索相似刑事案件的判例报告（案例库只有刑事），以预测判决结果或量刑，提高置信度。
3. **如何调用工具**：如果你决定检索，**必须**输出一个严格的 JSON 字符串，并用 `<search>` 和 `</search>` 标签包裹。
   - 调用【法律检索】示例：
     <search>
     {
       "检索类型": "法律检索",
       "关键词": "刑法 第xxx条 盗窃罪",
       "检索目的": "找到刑法中盗窃罪的刑期判定条文"
     }
     </search>
   - 调用【类案检索】示例：
     <search>
     {
       "检索类型": "类案检索",
       "检索案情": "张三蒙面进入邻居家，偷走现金5000元并持刀威胁屋主。",
       "罪名": ["盗窃罪", "抢劫罪"],
       "其他情节": "自首悔过"
     }
     </search>
4. **三段论推理**：接收到检索结果后，如果事实与法条匹配，必须使用 `<syllogism>` 标签生成三段论 JSON 分析。
   - 大前提 (Major Premise)：指代适用的具体罪名或法条。绝对不要重复输出法条原文，使用占位符 [法条参考 X]。
   - 小前提 (Minor Premise)：将用户平常的语言表述转化为专业的法言法语。
   - 结论 (Conclusion)：案件事实是否符合该法条，如何定罪或量刑。
5. **多轮迭代检索**：每次只输出一个 `<search>` 标签。
6. **最终回答**：收集充分后，必须将最终推理结论包裹在 `<answer>` 和 `</answer>` 标签中。

以下是需要回答的问题：{question_text}
"""

import torch.nn.functional as F

class StreamStopper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.buffer = "" 
        self.done = False
        self.action = None
        self.output_text = ""
        self.generated_ids = []

    def process_token(self, token_id):
        self.generated_ids.append(token_id.item())
        text = self.tokenizer.decode(self.generated_ids, skip_special_tokens=False)
        new_text = text[len(self.buffer):]
        self.buffer += new_text

        if "</search>" in self.buffer and not self.done:
            self.done = True
            self.action = "search"
            self.output_text = self.tokenizer.decode(self.generated_ids, skip_special_tokens=False).split("</search>")[0] + "</search>"
        elif "</answer>" in self.buffer and not self.done:
            self.done = True
            self.action = "answer"
            self.output_text = self.tokenizer.decode(self.generated_ids, skip_special_tokens=False).split("</answer>")[0] + "</answer>"

        return new_text

@torch.no_grad()
def stream_until_search(model, tokenizer, input_ids, max_new_tokens=1500, temperature=0.3, repetition_penalty=1.2):
    stopper = StreamStopper(tokenizer)
    generated = input_ids
    past_key_values = None

    for _ in range(max_new_tokens):
        outputs = model(generated if past_key_values is None else generated[:, -1:], use_cache=True, past_key_values=past_key_values)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        for token_id in set(generated[0].tolist()):
            logits[0, token_id] /= repetition_penalty

        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        stopper.process_token(next_token[0])
        generated = torch.cat((generated, next_token), dim=1)

        if stopper.done:
            break

    return stopper.action, stopper.output_text

class LLM_retriever:
    def __init__(self, model_path, retrieve_path, max_turn=5, topk=3):
        self.model_path = model_path
        self.topk = topk
        self.retrieve_path = retrieve_path
        self.max_turn = max_turn
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

        if 'Qwen' in model_path:
            self.tokenizer.pad_token_id = 151643
            self.tokenizer.eos_token_id = 151643
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '</s>'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    def _extract_query(self, text):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1] if matches else None
    
    def _extract_answer(self, text):
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1] if matches else None

    def _search(self, query):
        if not query or not query.strip(): return ""
        payload = {"queries": [query], "topk": self.topk, "return_scores": True}
        try:
            response = requests.post(self.retrieve_path, json=payload, timeout=10)
            response.raise_for_status()
            results = response.json().get("result", [])
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return ""

        if not results: return ""
        
        format_reference = ''
        for idx, doc_item in enumerate(results[0]):
            content = doc_item['document']['content']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    def gen(self, query, history=[], model_prompt=""):
        question = query.strip()
        if len(model_prompt): question = model_prompt + question
        
        prompt = NEW_SYSTEM_PROMPT.format(question_text=question)
        cnt = 0
        search_word_before = []

        while True:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            action, output_text = stream_until_search(self.model, self.tokenizer, input_ids, max_new_tokens=1500, temperature=0.4, repetition_penalty=1.2)

            instruct = ''
            search_results = ''
            history.append(output_text)

            if action == "answer" or cnt > self.max_turn:
                response = output_text
                break
            elif action == "search":
                tmp_query = self._extract_query(output_text)
                if search_word_before == tmp_query:
                    instruct = "如果我想给出最终回答，应该把答案放在 <answer> 和 </answer>之间。如果需要继续搜索，应该把新的关键词放在<search> 和 </search>之间。重新思考。"
                elif tmp_query and (cnt < self.max_turn):
                    search_word_before = tmp_query
                    search_results = self._search(tmp_query)
                elif cnt == self.max_turn:
                    instruct = "跳过检索阶段。注意：接下来总结以上思考，必须给出最终回答！把最终答案放在 <answer> 和 </answer>之间。"
                else:
                    instruct = "检索失败。重新检索或直接回答。"
            else:
                instruct = "\n我先前的操作有问题。如果我想搜索，应该把关键词放在<search> 和 </search>之间。如果我想给出最终回答，应该把答案放在 <answer> 和 </answer>之间。让我重新思考。\n"

            if len(search_results): history.append(search_results)
            
            search_text = self.search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            prompt += instruct
            cnt += 1

        history_str = '\n'.join(history)
        ans = self._extract_answer(response)
        return ans if ans else response, history_str

# 新增总结调用逻辑
def get_vllm_summary(query, history, port=8008):
    prompt = (
        "你是一个专业、严谨的法律总结助手。请根据下方提供的用户【原问题】与系统的【完整思维链及初步回答】，"
        "将其提炼成一份逻辑通顺、法理清晰、言简意赅的最终解答。\n"
        "核心要求：\n"
        "1. 剔除所有内部检索调用记录和 JSON 标签（如 <search>, <syllogism>, <answer>）。\n"
        "2. 隐藏所有的推理和试错过程，直接向用户呈现最终结论。\n"
        "3. 必须保留并明确引用判定所依据的核心法律条文或类案。\n\n"
        f"【原问题】：{query}\n"
        f"【思维链及初步回答】：{history}\n\n"
        "请直接输出总结后的标准回答，无需任何开头客套话："
    )
    
    url = f"http://127.0.0.1:{port}/v1/completions"
    payload = {
        "model": "Qwen3-8B",
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0.2
    }
    
    try:
        res = requests.post(url, json=payload, timeout=60)
        res.raise_for_status()
        return res.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[ERROR] Summarization API Call Failed: {e}")
        return history # 失败则回退返回全量 history

def mt_dialogue_gen(data_path, llm, result_path, summary_port):
    accelerator = Accelerator()
    tmp_path = result_path + ".tmp"
    dataset = TestDataset(data_path)
    val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)
    val_dataloader = accelerator.prepare(val_dataloader)

    processed = sum(1 for _ in open(tmp_path, "r", encoding="utf-8")) if os.path.exists(tmp_path) else 0

    with torch.no_grad():
        dataloader_iterator = tqdm(val_dataloader, total=len(val_dataloader)) if accelerator.is_main_process else val_dataloader
        idx = 0
        for batch in dataloader_iterator:
            for data in batch:
                if idx < processed:
                    idx += 1
                    continue

                query = data["needs"]
                model_prompt = data["model_prompt"]
                
                # 推理 + 检索
                res, history = llm.gen(query, [], model_prompt)
                res = res.strip()
                
                # 调用总结模型
                summary = get_vllm_summary(query, history, port=summary_port)

                dialogue = f"用户：{query}\nAI助手：{summary}\n"

                output = {
                    "dialogue": dialogue,
                    "model": llm.model_path,
                    "thinking": history,
                    "summary": summary
                }
                output.update(data)

                if accelerator.is_main_process:
                    with open(tmp_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(output, ensure_ascii=False) + "\n")
                idx += 1
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        data_list = [json.loads(line) for line in open(tmp_path, "r", encoding="utf-8")]
        data_list = list_to_dict(data_list)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        os.remove(tmp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_turn', default=4, type=int)
    parser.add_argument('--topk', default=4, type=int)
    parser.add_argument('--retrieve_path', default="http://127.0.0.1:8006/retrieve", type=str)
    parser.add_argument('--summary_port', default=8008, type=int)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    llm = LLM_retriever(args.model_path, retrieve_path=args.retrieve_path, max_turn=args.max_turn, topk=args.topk)
    mt_dialogue_gen(args.data_path, llm, args.result_path, args.summary_port)