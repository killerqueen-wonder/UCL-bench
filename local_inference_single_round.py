import os
import json
import torch
import logging
import argparse
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
import transformers
from transformers import set_seed
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel 
import re
import requests

from utils.user_simulator import GPTPerson



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
        self.query_prompt = ""
        raw_data = {}
        ### dataset 已经 load json 文件了
        with open(data_path) as f:
            raw_data = json.load(f)
        self.dataset = [item for sublist in raw_data.values() for item in sublist]  

    def __getitem__(self ,  index):
        # 在这里实现获取单个样本的逻辑
        sample = self.dataset[index]
        return sample

    def __len__(self):
        return len(self.dataset)
    
    def collate_fn(self, batch):
        return batch

class LLM:
    def __init__(self,model_path):
        self.model_path = model_path
        print("--------------加载模型路径为：---------------\n",model_path)
        ## 定义 model 变量 
        if "huatuo" in args.model_path.lower():
            model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
            model.generation_config = GenerationConfig.from_pretrained(args.model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        if "Qwen-72B-Chat" not in args.model_path and "deepseek" not in args.model_path and "DISC-LawLLM" not in args.model_path:
            model.cuda().eval()
        

        if 'PMC_LLaMA_13B' in args.model_path or 'zhongjing' in args.model_path:
            left_tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_path, padding_side='left')
        else:
            left_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')

        if 'Qwen' in args.model_path:
            left_tokenizer.pad_token_id = 151643
            left_tokenizer.eos_token_id = 151643
        if  left_tokenizer.pad_token is None:
            # left_tokenizer.pad_token = '<PAD>'
            left_tokenizer.pad_token = '</s>'
        
        self.model = model
        self.left_tokenizer = left_tokenizer
    
    def gen(self, query , history = [], model_prompt=""):
        
        
        if "qwen" in self.model_path.lower():
            if history == [] :
                history.append({"role":"system","content":model_prompt})
            history.append({"role": "user", "content": query})

            text = self.left_tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.left_tokenizer(text, return_tensors="pt")
            inputs = inputs.to('cuda')
            response_ids = self.model.generate(**inputs, max_new_tokens=8000)[0][len(inputs.input_ids[0]):].tolist()
            response = self.left_tokenizer.decode(response_ids, skip_special_tokens=False)
            # response = self.left_tokenizer.decode(response_ids, skip_special_tokens=True)

            # Update history
            history.append({"role": "assistant", "content": response})
            return response, history
    


# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False


class LLM_retriever:
    """
    实现一个“思考-检索-再思考-回答”的闭环生成系统。
    """
    def __init__(self, model_path, api_key=None, api_url=None, max_turn=5,retrieve_path="http://127.0.0.1:8006/retrieve"):
        self.model_path = model_path
        
        self.api_key = api_key
        self.api_url = api_url
        self.retrieve_path = retrieve_path
        self.max_turn=max_turn

        print("--------------加载模型路径为：---------------\n", model_path)

        
        # Initialize the tokenizer and model
    
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")


        # 特殊token设置
        if 'Qwen' in model_path:
            self.tokenizer.pad_token_id = 151643
            self.tokenizer.eos_token_id = 151643
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '</s>'

        # 停止条件定义
        self.target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
        self.stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(self.target_sequences, self.tokenizer)])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.curr_eos = [151645, 151643]
        self.search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    def _extract_query(self, text):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1] if matches else None

    def _search(self, query):
        """调用本地检索服务"""
        if not query or not query.strip():
            print("[WARNING] Empty query passed to search function.")
            return ""

        payload = {"queries": [query], "topk": 3, "return_scores": True}
        try:
            response = requests.post(
                self.retrieve_path,
                json=payload,
                proxies={"http": None, "https": None},
                timeout=10
            )
            response.raise_for_status()
            json_data = response.json()
            results = json_data.get("result", [])
        except requests.exceptions.Timeout:
            print("[ERROR] Search request timed out.")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed: {e}")
            return ""
        except ValueError as e:
            print(f"[ERROR] Failed to decode JSON: {e}")
            return ""

        if not results:
            print("[INFO] No results returned from search.")
            return ""

        def _passages2string(retrieval_result):
            format_reference = ''
            for idx, doc_item in enumerate(retrieval_result):
                            
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            return format_reference

        return _passages2string(results[0])

    def gen(self, question, history=None):
        """执行完整的思考-检索-再思考-回答流程"""
        if history is None:
            history = []

        question = question.strip()
        # if question[-1] != '?':
        #     question += '?'

        system_prompt = (
            "根据要求，回答问题。你必须遵守思考-检索-思考-回答的推理模式。\n"
            "思考：对问题进行推理，尝试解答。推理过程中，如果你发现涉及某些法律条文，则进入检索步骤。\n"
            "检索：请把需要检索的关键词放在 <search> 和 </search> 标签之间，调用搜索引擎。例如：<search> 民法典 盗窃罪 </search>。\n"
            "系统将返回最相关的搜索结果，并置于 <information> 和 </information> 标签之间。根据返回的结果，继续下一步思考。\n"
            "再次思考：基于检索结果，继续对问题进行推理。如果没有帮助，则修改关键词重新检索；如果有把握得到最终答案，则进入回答。\n"
            "回答：在 <answer> 和 </answer> 标签内提供最终答案。\n"
            f"以下是需要回答的问题：{question}\n"
        )

        # 构造prompt
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": system_prompt}],
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = system_prompt

        cnt = 0
        print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
        print(f'**[prompt]:{prompt}')

        while True:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=2500,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.3
            )

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f'[debug] output_text="{output_text}"')

            if outputs[0][-1].item() in self.curr_eos or cnt > self.max_turn:
                response = output_text
                break

            tmp_query = self._extract_query(output_text)
            print(f'[debug] search query="{tmp_query}"')
            if tmp_query and( cnt < self.max_turn):
                
                search_results = self._search(tmp_query)
            
            elif cnt==self.max_turn:
                search_results = "到达检索次数限制，接下来直接输出回答。"

            else:
                search_results = "检索失败。重新检索或直接回答。"

            print(f'**[debug]searching result :\n"{search_results}"')

            search_text = self.search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            cnt += 1

        # 更新历史记录
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})

        return response, history
        



        


def mt_dialogue_gen(data_path,llm,result_path):
    accelerator = Accelerator()
    # torch.cuda.set_device(accelerator.process_index)
    
    ## dataset 
    dataset = TestDataset(data_path)
    # 注意 batch_size
    val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)
    val_dataloader = accelerator.prepare(val_dataloader)
    accelerator.wait_for_everyone()

    with torch.no_grad():
        dataloader_iterator = tqdm(val_dataloader, total=len(val_dataloader)) if accelerator.is_main_process else val_dataloader
        print(dataloader_iterator)

        data_list = []  ## 多个数据的对话列表。
        for batch in dataloader_iterator:
            for data in batch:
                dialogue = ""  ## 用于记录对话
                model_prompt = data["model_prompt"]
                
                print("\n----------------------------------- model_prompt\n" + model_prompt)
                query=data["needs"]
                dialogue = dialogue + "用户：" + query + "\n"
                print("\n-----------------------------------first query\n"+query)
                history = []
                for round in range(1,2):#单轮问答
                    data["dialogue_round"] = round  ## 用于记录轮次 
                    ## 本地AI 助手生成回复
                    res,history = llm.gen(query,history,model_prompt)
                    res = res.strip()
                    print("\n-----------------------------------res\n"+res)
                    dialogue = dialogue + "AI助手：" + res + "\n"
                    
                    

                ## 结果 data
                data['dialogue'] = dialogue
                data['model'] = llm.model_path
                data_list.append(data)

        ### 保存结果
        data_list = list_to_dict(data_list)
        with open(result_path, 'w', encoding='utf-8') as file:
            json.dump(data_list, file,indent=4,ensure_ascii=False)

        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')
    # Model Args
    parser.add_argument('--data_path', default='/mntcephfs/lab_data/ganruoli/UC_bench/dataset/legal_data/legal_data_sample.json', type=str)
    parser.add_argument('--model_path', default='/mntcephfs/data/med/zhanghongbo/yaojishi/cjy/ckpts/huatuo2_7B_v2/checkpoint-0-30346/tfmr32', type=str)
    parser.add_argument('--result_path', default='/mntcephfs/lab_data/ganruoli/UC_bench/experiment/legal/chatglm3-6b_legal.json', type=str)
    # parser.add_argument("--model_name", type=str, default="gpt-4o-2024-11-20", help="Name of the GPT model to use.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("--api_url", type=str, default="https://api.openai.com/v1/chat/completions", help="OpenAI API URL.")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--retriever', default=False, type=bool)
    
    args = parser.parse_args()
    set_seed(args.seed)
    model_path=args.model_path

    if args.retriever:
        llm=LLM_retriever(model_path)
    else:
        llm= LLM(model_path)

    mt_dialogue_gen(args.data_path,llm,args.result_path)
