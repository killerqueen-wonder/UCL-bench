import argparse
import json
import random 
import re
import time
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def call_vllm_judge(prompt, port=8009):
    """通过API并发调用评测模型"""
    url = f"http://127.0.0.1:{port}/v1/completions"
    payload = {
        "model": "Qwen2.5-8B-Instruct", # 动态获取或写死皆可
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0.2
    }
    try:
        response = requests.post(url, json=payload, timeout=200)
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"API调用失败: {e}")
        return "-1"

def compute_score(evaluation_result):
    try:
        label_content = evaluation_result.strip()
        label = re.findall(r"\[\[(\d)\]\]", label_content)
        if label:
            label = label[-1]
            if label == "1": return [10, 0]
            elif label == "2": return [0, 10]
            elif label == "3": return [5, 5]
        return [-1, -1]
    except Exception:
        return [-1, -1]

def process_single_evaluation(chatgpt_item, model_item, data_info, port):
    """处理单条评测任务供线程池使用"""
    evaluation_prompt = data_info["evaluation_prompt"]
    evaluation_hints = data_info["evaluation_hints"]
    information = data_info["information"]
    needs = data_info["needs"]
    
    # 核心变更：使用 summary 字段进行评测。如果缺失则回退使用 thinking。
    model_content = model_item.get("summary", model_item.get("thinking", ""))

    if random.random() < 0.5:
        input_text = evaluation_prompt.format(
            information=information, needs=needs, evaluation_hints=evaluation_hints,
            dialogue1=chatgpt_item["dialogue"].strip(), dialogue2=model_content.strip()
        ).strip()
        evaluate_result = call_vllm_judge(input_text, port)
        evaluate_score = compute_score(evaluate_result)
    else:
        input_text = evaluation_prompt.format(
            information=information, needs=needs, evaluation_hints=evaluation_hints,
            dialogue1=model_content.strip(), dialogue2=chatgpt_item["dialogue"].strip()
        ).strip()
        evaluate_result = call_vllm_judge(input_text, port)
        score = compute_score(evaluate_result)
        evaluate_score = [score[1], score[0]] # 翻转分数

    return {
        "task_name": chatgpt_item["task_name"],
        "id": chatgpt_item["id"],
        "evaluation_hints": evaluation_hints,
        "information": information,
        "needs": needs,
        "chatgpt_dialogue": chatgpt_item["dialogue"],
        "model_dialogue": model_content,
        "evaluation_prompt": input_text,
        "evaluation_result": evaluate_result,
        "evaluation_score": evaluate_score
    }

def format_time(seconds):
    if seconds < 60: return f"{seconds:.0f}秒"
    elif seconds < 3600: return f"{seconds//60:.0f}分{seconds%60:.0f}秒"
    else: return f"{seconds//3600:.0f}时{(seconds%3600)//60:.0f}分{seconds%60:.0f}秒"

def generate_evaluate_results_parallel(chatgpt_result, model_result, datasource, port, workers=32):
    start_time = time.time()
    total_result = {task: [] for task in datasource.keys()}
    
    # 构建所有任务队列
    tasks_queue = []
    for task_name in datasource.keys():
        for chatgpt_item in chatgpt_result.get(task_name, []):
            for model_item in model_result.get(task_name, []):
                if chatgpt_item["id"] == model_item["id"]:
                    data_info = next((d for d in datasource[task_name] if d["id"] == chatgpt_item["id"]), None)
                    if data_info:
                        tasks_queue.append((chatgpt_item, model_item, data_info))

    total_tasks = len(tasks_queue)
    print(f"总评测任务数: {total_tasks}，并发数: {workers}")

    # 并发执行
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_evaluation, c, m, d, port) for c, m, d in tasks_queue]
        
        with tqdm(total=total_tasks, desc="评测进度") as pbar:
            for future in as_completed(futures):
                res = future.result()
                total_result[res["task_name"]].append(res)
                pbar.update(1)

    total_elapsed = time.time() - start_time
    print(f"评测总耗时: {format_time(total_elapsed)}")
    return total_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chatgpt_result_path', type=str, required=True)
    parser.add_argument('--model_result_path', type=str, required=True)
    parser.add_argument('--datasource_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--judge_port', type=int, default=8009)
    args = parser.parse_args()

    with open(args.chatgpt_result_path, 'r', encoding='utf-8') as f: chatgpt_result = json.load(f)
    with open(args.model_result_path, 'r', encoding='utf-8') as f: model_result = json.load(f)
    with open(args.datasource_path, 'r', encoding='utf-8') as f: datasource = json.load(f)

    total_result = generate_evaluate_results_parallel(chatgpt_result, model_result, datasource, port=args.judge_port)

    with open(args.result_path, 'w', encoding='utf-8') as f:
        json.dump(total_result, f, indent=4, ensure_ascii=False)