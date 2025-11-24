import argparse
import json
import random 
import re
import threading
from retrying import retry
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载本地模型和tokenizer
model_name = "Qwen/Qwen2.5-8B-Instruct"  # 根据实际路径修改
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """加载本地模型"""
    global tokenizer, model
    if tokenizer is None or model is None:
        print(f"Loading model {model_name} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        print("Model loaded successfully!")

## 定义调用本地模型的函数 
@retry(wait_fixed=2000, stop_max_attempt_number=10)
def call_local_model(messages, max_length=8192):
    class InterruptableThread(threading.Thread):
        def __init__(self, messages, max_length):
            threading.Thread.__init__(self)
            self.result = None
            self.messages = messages
            self.max_length = max_length

        def run(self):
            try:
                # 构建Qwen格式的输入
                if isinstance(self.messages, str):
                    prompt = self.messages
                else:
                    # 如果是消息列表，转换为Qwen的对话格式
                    prompt = ""
                    for msg in self.messages:
                        if msg["role"] == "user":
                            prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n<|im_start|>assistant\n"
                        elif msg["role"] == "assistant":
                            prompt += f"{msg['content']}<|im_end|>\n"
                
                # 确保模型已加载
                load_model()
                
                # Tokenize输入
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length-512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 生成响应
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.2,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                # 解码响应
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                self.result = response.strip()
                
            except Exception as e:
                print(f"Error in local model inference: {e}")
                self.result = None
    
    it = InterruptableThread(messages, max_length)
    it.start()
    
    # 设置超时时间
    timeout_duration = 200
    it.join(timeout_duration)
    
    if it.is_alive() or it.result is None:
        print('本地模型调用超时或出错')
        raise Exception("本地模型调用超时")
    else:
        return it.result

def response(message, api_key=None, gpt_url=None):
    """兼容原有接口，但忽略api_key和gpt_url参数"""
    messages = []
    messages.append({"role": "user", "content": message})
    response_text = call_local_model(messages)
    return response_text

def compute_score(evaluation_result):
    review = evaluation_result
    try:
        label_content = review.strip()
        label = re.findall(r"\[\[(\d)\]\]", label_content)
        if label:
            label = label[-1]
            if label == "1" :
                return [10,0]
            elif label == "2" :
                return [0,10]
            elif label == "3" :
                return [5,5]
            else:
                return [-1,-1]
        else:
            return [-1,-1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]

## score 中 ，第一个分数是 chatgpt ， 第二个分数是 model 
def call_evaluate(chatgpt_dialogue, model_dialogue, information, needs, evaluation_hints, evaluation_prompt, api_key=None, gpt_url=None):
    """评测函数，忽略api_key和gpt_url参数"""
    
    if random.random() < 0.5:
        input_text = evaluation_prompt.format(
            information=information,
            needs=needs,
            evaluation_hints=evaluation_hints,
            dialogue1=chatgpt_dialogue,
            dialogue2=model_dialogue
        )
        input_text = input_text.strip()
        evaluation_result = response(input_text)
        score = compute_score(evaluation_result)
    else:
        input_text = evaluation_prompt.format(
            information=information,
            needs=needs,
            evaluation_hints=evaluation_hints,
            dialogue1=model_dialogue,
            dialogue2=chatgpt_dialogue
        )
        input_text = input_text.strip()
        evaluation_result = response(input_text)
        score = compute_score(evaluation_result)
        score[0], score[1] = score[1], score[0]
    
    return input_text, evaluation_result, score

def generate_evaluate_results(chatgpt_result, model_result, datasource, api_key=None, gpt_url=None):
    """生成评测结果"""
    task_names = datasource.keys()

    total_result = {}
    for task_name in task_names:
        print(task_name)
        task_evaluate_temp = []
        for chatgpt_item in chatgpt_result[task_name]:
            for model_item in model_result[task_name]:
                ## 匹配到对应数据。
                if chatgpt_item["id"] == model_item["id"] and chatgpt_item["task_name"] == model_item["task_name"]:
                    ## 输出 结果。 score 为 [x,y] 前一个为 chatgpt ，后一个为 model 分数。
                    evaluation_temp = {}  ## 存储一条数据的评估结果

                    ## 找到源数据的 evaluation_points , information , needs 
                    for data in datasource[task_name]:
                        if data["id"] == chatgpt_item["id"]:
                            evaluation_prompt = data["evaluation_prompt"]
                            evaluation_hints = data["evaluation_hints"]
                            information = data["information"]
                            needs  = data["needs"]
                            break
                    
                    ## 第一个是 chatgpt 
                    evaluate_prompt, evaluate_result, evaluate_score = call_evaluate(
                        chatgpt_item["dialogue"].strip(),
                        model_item["thinking"].strip(),
                        information,
                        needs,
                        evaluation_hints,
                        evaluation_prompt
                    )
                    print("--------------------------------------------------")
                    print(evaluate_result)
                    print("--------------------------------------------------")
                    evaluation_temp["task_name"] = chatgpt_item["task_name"]
                    evaluation_temp["id"] = chatgpt_item["id"]
                    evaluation_temp["evaluation_hints"] = evaluation_hints
                    evaluation_temp["information"] = information
                    evaluation_temp["needs"] = needs
                    evaluation_temp["chatgpt_dialogue"] = chatgpt_item["dialogue"]
                    evaluation_temp["model_dialogue"] = model_item["thingking"]
                    evaluation_temp["evaluation_prompt"] = evaluate_prompt
                    evaluation_temp["evaluation_result"] = evaluate_result
                    evaluation_temp["evaluation_score"] = evaluate_score

                    task_evaluate_temp.append(evaluation_temp)

        total_result[task_name] = task_evaluate_temp
    return total_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # Model Args
    parser.add_argument('--chatgpt_result_path', type=str)
    parser.add_argument('--model_result_path', type=str)
    parser.add_argument('--datasource_path', type=str)
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-8B-Instruct", help="Path to local model")

    args = parser.parse_args()
    chatgpt_result_path = args.chatgpt_result_path
    model_result_path = args.model_result_path
    datasource_path = args.datasource_path
    result_path = args.result_path
    
    # 设置模型路径
    if args.model_path:
        model_name = args.model_path

    ## base_line
    with open(chatgpt_result_path, 'r', encoding='utf-8') as file1:
        chatgpt_result = json.load(file1)
    
    ## 对比模型
    with open(model_result_path, 'r', encoding='utf-8') as file2:
        model_result = json.load(file2)

    ## 源数据，主要用它的 evaluation_hints
    with open(datasource_path, 'r', encoding='utf-8') as file3:
        datasource = json.load(file3)

    ## 预加载模型
    print("Preloading model...")
    load_model()
    
    ## 生成评测结果
    total_result = generate_evaluate_results(chatgpt_result, model_result, datasource)
    
    evaluation_result_path = result_path
    with open(evaluation_result_path, 'w', encoding='utf-8') as file:
        json.dump(total_result, file, indent=4, ensure_ascii=False)