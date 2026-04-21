
#################################
import os
import json
import logging
import argparse
import requests
import re
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

os.umask(0)
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

def list_to_dict(data_list):
    classified_dict = {}
    for item in data_list:
        task_name = item.get("task_name", "unknown")
        if task_name not in classified_dict:
            classified_dict[task_name] = [item]
        else:
            classified_dict[task_name].append(item)
    return classified_dict

NEW_SYSTEM_PROMPT = """
你是一个严谨且专业的法律AI助手。你的任务是通过逐步思考用户请求并回答法律问题。回答必须基于事实，严禁编造法律条文或案例。

### 核心指令：
1. **判断是否需要检索**：你可以使用检索工具。如果问题基础且你非常有把握，也可以不使用工具直接作答。

2. **支持的检索工具**（两种）：
   - **法律检索**：需要确认某项罪名的具体刑期、适用条件、或者某一司法解释的原文时使用。
   - 不要试图一次性把所有关键词都搜完。每次最多只查找两项最相关的法律。
   - 先搜索最核心的概念。不要用缩写词，尽量用完整的，最有特点的，区别于其他法条的关键词。搜索关键词示例：“刑法 盗窃罪”、“最高人民法院关于适用〈民事诉讼法〉的解释 第501条”。
   -如果你搜索了三次依然没有找到相关条文，请直接承认未找到，修改思考思路，搜索其他条文。不要编造内容。
   - **类案检索**：用来检索刑事相似案件的判例报告，以预测判决结果或量刑，提高置信度。
   

3. **如何调用工具**：如果你决定检索，**必须**输出一个严格的 JSON 字符串，并用 `<search>` 和 `</search>` 标签包裹。
   - **调用【法律检索】的 JSON 格式示例**：
     <search>
     {{
       "检索类型": "法律检索",
       "关键词": "刑法 第xxx条 盗窃罪（你想要找的法条的编号和法条原文中包含的关键词）",
       "检索目的": "找到刑法中盗窃罪的刑期判定条文"
     }}
     </search>
   - **调用【类案检索】的 JSON 格式示例**：
     <search>
     {{
       "检索类型": "类案检索",
       "检索案情": "张三蒙面进入邻居家，偷走现金5000元并持刀威胁屋主。",
       "罪名": ["盗窃罪", "抢劫罪"],
       "其他情节": "自首悔过"
     }}
     </search>

4. **三段论推理（关键）**：
   在你接收到【法律检索结果】后，如果用户的案例事实与某条检索到的法律法规能够匹配，你**必须**在接下来的思考中，首先使用 `<syllogism>` 标签生成一个三段论 JSON 进行法理分析。
   - **大前提 (Major Premise)**：指代适用的具体罪名或法条。**注意：绝对不要重复输出法条原文，必须严格使用占位符 `[法条参考 X]`**（X为检索结果给出的序号，例如 `[法条参考 1]`）。
   - **小前提 (Minor Premise)**：将用户平常的语言表述转化为专业的**法言法语**。
   - **结论 (Conclusion)**：案件事实是否符合该法条，当事人是否适用该法条，以及根据法条应该如何定罪或量刑。
   
   **三段论 JSON 格式示例**：
   <syllogism>
   {{
     "Major Premise": "刑法第264条 盗窃罪 [法条参考 1]",
     "Minor Premise": "张三于某日以非法占有为目的，入室秘密窃取他人财物，共计金额5000元...",
     "Conclusion": "张三的行为符合盗窃罪的构成要件，适用该法条，判处..."
   }}
   </syllogism>
   *(注意：如果检索结果无法匹配事实，则不要生成该三段论标签和内容)*

5. **多轮迭代检索**：每次只输出一个 `<search>` 标签。接收结果后分析是否充足。最多允许检索 **{max_turn}** 次。

6. **最终回答**：收集充分后，必须将最终推理结论包裹在 `<answer>` 和 `</answer>` 标签中。
### 回答流程：
- 遇到问题 -> 分析问题复杂度，发现不需要查询资料 -> 思考问题 -> 回答
- 遇到问题 -> 分析问题复杂度，发现需要查询资料 -> **调用工具** (此时你会暂停) -> 接收工具结果 -> 分析结果 -> 发现还需要查别的 -> **再次调用工具** ... -> 最终整合信息回答。

"""

class VLLM_Retriever_Agent:
    def __init__(self, vllm_url, retrieve_url, model_name="Qwen3-8B", max_turn=5, topk=3):
        self.vllm_url = f"{vllm_url}/v1/completions"
        self.retrieve_url = retrieve_url
        self.model_name = model_name
        self.max_turn = max_turn
        self.topk = topk
        self.search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    def _extract_tag(self, text, tag):
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else None

    def _search(self, query):
        if not query or not query.strip() or not self.retrieve_url: return ""
        payload = {"queries": [query], "topk": self.topk, "return_scores": True}
        try:
            res = requests.post(self.retrieve_url, json=payload, timeout=15)
            res.raise_for_status()
            results = res.json().get("result", [])
        except Exception as e:
            return ""

        if not results: return ""
        format_reference = ''
        for idx, doc_item in enumerate(results[0]):
            content = doc_item['document']['content']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    def gen(self, query, instruction=""):
        question = query.strip()
        if instruction: question = f"{instruction}\n{question}"
        
        prompt = NEW_SYSTEM_PROMPT.format(question_text=question)
        cnt = 0
        search_word_before = ""
        history = []

        # ================= 推理阶段性能统计 =================
        sys_tool_latency = 0.0        
        sys_rag_count = 0             
        sys_user_prompt_tokens = 0    
        sys_total_prompt_tokens = 0   
        sys_total_completion_tokens = 0 
        # ================================================

        while True:
            payload = {
                "model": self.model_name,
                "prompt": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "max_tokens": 1500,
                "temperature": 0.3,
                "stop": ["</search>", "</answer>", "<|im_end|>"]
            }
            
            output_text = ""
            current_prompt_tokens = 0
            current_completion_tokens = 0
            
            try:
                
                res = requests.post(self.vllm_url, json=payload, timeout=200)
                res.raise_for_status()
                data = res.json()
                
                output_text = data["choices"][0].get("text", "")
                usage = data.get("usage", {})
                current_prompt_tokens = usage.get("prompt_tokens", 0)
                current_completion_tokens = usage.get("completion_tokens", 0)
                
            except Exception as e:
                logger.error(f"vLLM 请求异常: {e}")
                break

            # --- 累加 Token 指标 ---
            if cnt == 0:
                sys_user_prompt_tokens = current_prompt_tokens # 记录用户的首轮干净输入
            
            sys_total_prompt_tokens += current_prompt_tokens
            sys_total_completion_tokens += current_completion_tokens

            # --- 标签补齐 ---
            if "<search>" in output_text and "</search>" not in output_text:
                output_text += "</search>"; action = "search"
            elif "<answer>" in output_text and "</answer>" not in output_text:
                output_text += "</answer>"; action = "answer"
            else:
                action = "answer"

            history.append(output_text)
            instruct = ''
            search_results = ''

            # --- 动作路由 ---
            if action == "answer" or cnt >= self.max_turn:
                break
                
            elif action == "search":
                tmp_query = self._extract_tag(output_text, "search")
                if tmp_query == search_word_before:
                    instruct = "请勿重复检索。尝试直接回答或使用新关键词。"
                elif tmp_query and cnt < self.max_turn:
                    search_word_before = tmp_query
                    # 捕获检索阻塞耗时
                    t0 = time.time()
                    search_results = self._search(tmp_query)
                    sys_tool_latency += (time.time() - t0)
                    sys_rag_count += 1
                elif cnt == self.max_turn:
                    instruct = "跳过检索。请立即给出最终回答，包裹在 <answer> 中。"
                else:
                    instruct = "检索格式错误。请重新思考。"

            if search_results: history.append(search_results)
            prompt += self.search_template.format(output_text=output_text, search_results=search_results)
            prompt += instruct
            cnt += 1

        agent_metrics = {
            "tool_latency_sec": sys_tool_latency,
            "rag_count": sys_rag_count,
            "user_prompt_tokens": sys_user_prompt_tokens,
            "main_total_prompt_tokens": sys_total_prompt_tokens,
            "main_total_comp_tokens": sys_total_completion_tokens
        }

        return "\n".join(history), agent_metrics

def get_universal_vllm_summary(query, history, port, model_name="Qwen3-8B"):
    prompt = (
        "你是一个专业、严谨的法律文书与答案整理助手。请根据下方提供的【原问题】与系统的【思维链解析】，提取出最终答案。\n\n"
        "【核心规则】（请严格根据原问题的内容自动调整你的输出策略）：\n"
        "1. **如果这是选择题**（原问题中明确包含了 A, B, C, D 或其他选项）：请你**直接且只输出最终的选项大写字母**（例如 A、C 、 ABCD或者其他预设选项）。绝对不要包含任何解释、分析过程、客套话，连标点符号都不要有。\n"
        "2. **如果这是主观题/问答题**（原问题中没有选项）：请你整理出一份逻辑严密、法理清晰的最终解答。必须剔除所有 `<search>`, `<syllogism>`, `<answer>` 等内部标签及机器检索痕迹，保留相关法条和类案的核心内容，直接向用户呈现流畅专业的最终回答。\n\n"
        f"【原问题】：{query}\n"
        f"【思维链解析】：{history}\n\n"
        "最终答案："
    )
    url = f"http://127.0.0.1:{port}/v1/completions"
    payload = {"model": model_name, "prompt": prompt, "max_tokens": 1024, "temperature": 0.1}
    
    try:
        res = requests.post(url, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        
        summary_text = data["choices"][0]["text"].strip()
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        comp_tokens = usage.get("completion_tokens", 0)
        
        return summary_text, prompt_tokens, comp_tokens
    except Exception as e:
        logger.error(f"Summary API 调用失败: {e}")
        return history, 0, 0 

def process_single_item(item, agent, summary_port):
    # --- 记录整体起止时间 ---
    start_time = time.time()
    
    query = item["needs"]
    model_prompt = item.get("model_prompt", "")
    full_prompt = f"{model_prompt}\n{query}" if model_prompt else query
    
    # 1. 主代理推理
    history, agent_metrics = agent.gen(query=query, instruction=model_prompt)
    
    # 2. 总结代理脱水
    summary, sum_prompt_tok, sum_comp_tok = get_universal_vllm_summary(full_prompt, history, summary_port, agent.model_name)
    
    # --- 结束整体计时 ---
    total_time_sec = time.time() - start_time
    
    # 3. 终极 Token 清算 
    user_prompt_tokens = agent_metrics["user_prompt_tokens"]
    completion_tokens = sum_comp_tok  # 给用户看的最终回答
    
    # 全流程系统总消耗 Token = (主Agent Prompt + 主Agent Comp) + (总结Agent Prompt + 总结Agent Comp)
    total_tokens = (agent_metrics["main_total_prompt_tokens"] + agent_metrics["main_total_comp_tokens"]) + (sum_prompt_tok + sum_comp_tok)
    
    # 中间损耗 Token
    inter_agent_tokens = total_tokens - user_prompt_tokens - completion_tokens

    # 4. 构建输出结果
    out_item = item.copy()
    out_item["dialogue"] = f"用户：{query}\nAI助手：{summary}\n"
    out_item["model"] = agent.model_name
    out_item["thinking"] = history
    out_item["summary"] = summary
    
    # 性能与成本指标保存
    out_item["total_time_sec"] = total_time_sec
    out_item["tool_latency_sec"] = agent_metrics["tool_latency_sec"]
    out_item["rag_count"] = agent_metrics["rag_count"]
    
    out_item["total_tokens"] = total_tokens
    out_item["user_prompt_tokens"] = user_prompt_tokens
    out_item["completion_tokens"] = completion_tokens
    out_item["inter_agent_tokens"] = inter_agent_tokens
    
    return out_item

def mt_dialogue_gen(data_path, result_path, agent, summary_port, workers=32):
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    dataset = [item for sublist in raw_data.values() for item in sublist]
    results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_item, item, agent, summary_port) for item in dataset]
        for future in tqdm(as_completed(futures), total=len(futures), desc="UCL-Bench Inferencing"):
            results.append(future.result())

    final_dict = list_to_dict(results)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="Qwen3-8B")
    parser.add_argument('--vllm_url', type=str, default="http://127.0.0.1:8007")
    parser.add_argument('--summary_port', type=int, default=8008)
    parser.add_argument('--retrieve_path', default="http://127.0.0.1:8005/retrieve", type=str)
    parser.add_argument('--retriever', type=bool, default=True)
    parser.add_argument('--max_turn', default=12, type=int)
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--workers', default=32, type=int)
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.result_path) or ".", exist_ok=True)

    agent = VLLM_Retriever_Agent(
        vllm_url=args.vllm_url,
        retrieve_url=args.retrieve_path if args.retriever else None,
        model_name="Qwen3-8B",
        max_turn=args.max_turn,
        topk=args.topk
    )

    mt_dialogue_gen(args.data_path, args.result_path, agent, args.summary_port, args.workers)