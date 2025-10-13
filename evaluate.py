import argparse
import json
import random 
import requests
import threading
from retrying import retry
import re
from openai import OpenAI

## 定义调用 gpt4 的函数 
@retry(wait_fixed=2000, stop_max_attempt_number=10)
def call_api_timelimit(messages,api_key,gpt_url):
    class InterruptableThread(threading.Thread):
        def __init__(self,messages,api_key,gpt_url):
            threading.Thread.__init__(self)
            self.result = None
            self.messages = messages
            self.api_key = api_key
            self.gpt_url = gpt_url
            self.model_name='gpt-4o-2024-11-20'#裁判模型

        def run(self):
            try:
                # parameters = {
                # "model": "gpt-4-0613",
                # "messages": self.messages
                # }
                # headers = {
                #     "Content-Type": "application/json",
                #     "Authorization": "Bearer sk-IlhmAWpQFIfc5a0IF566F7Fe93A04522A255422c68158fD7"
                # }
                # response = requests.post(
                #     "https://api.ai-gaochao.cn/v1/chat/completions",
                #     headers=headers,
                #     json=parameters,
                # ).json()

                # if 'choices' not in response and 'error' in response:
                #     raise Exception(response['error']['message'] + '\n')
                
                # response_text = response["choices"][0]["message"]["content"].strip()
                # self.result = response_text
                client = OpenAI(api_key=self.api_key, base_url=self.gpt_url)

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    
                    temperature=0.2,
                    stream=False
                )
                response_text = response.choices[0].message.content.strip()
                self.result = response_text
            except Exception as e:
                print(e)
    it = InterruptableThread(messages,api_key,gpt_url)
    it.start()
    # 时间
    timeout_duration = 200
    it.join(timeout_duration)
    if it.is_alive() or it.result is None:
        print('时间进程出错')
        raise Exception("API调用超时")
    else:
        return it.result

def response(message,api_key,gpt_url):
    messages = []
    messages.append({"role": "user", "content": message})
    response_text = call_api_timelimit(messages,api_key,gpt_url)
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
def call_evaluate(chatgpt_dialogue,model_dialogue,information,needs,evaluation_hints,evaluation_prompt,api_key,gpt_url):
    # if evaluation_points["answer"] == "":
    #     evaluation_points["answer"] = "无"
    # if evaluation_points["must_contain"] == "":
    #     evaluation_points["must_contain"] = "无"
    # if evaluation_points["at_least_contain"] == "":
    #     evaluation_points["at_least_contain"] = "无"
    # if evaluation_points["should_contain"] == "":
    #     evaluation_points["should_contain"] = "无"
    # if evaluation_points["encourage_contain"] == "":
    #     evaluation_points["encourage_contain"] = "无"
    # if evaluation_points["forbid_contain"] == "":
    #     evaluation_points["forbid_contain"] = "无"

    if random.random() < 0.5:
        input = evaluation_prompt.format(information=information,needs=needs,\
            # hints_answer=evaluation_points["answer"],hints_must=evaluation_points["must_contain"],\
            #     hints_least=evaluation_points["at_least_contain"],hints_should=evaluation_points["should_contain"],\
            #         hints_encourage=evaluation_points["encourage_contain"],hints_forbid=evaluation_points["forbid_contain"],\
                        evaluation_hints=evaluation_hints,dialogue1=chatgpt_dialogue,dialogue2=model_dialogue)
        input = input.strip()
        evaluation_result = response(input,api_key,gpt_url)
        score = compute_score(evaluation_result)
    else:
        input = evaluation_prompt.format(information=information,needs=needs,\
            # hints_answer=evaluation_points["answer"],hints_must=evaluation_points["must_contain"],\
            #     hints_least=evaluation_points["at_least_contain"],hints_should=evaluation_points["should_contain"],\
            #         hints_encourage=evaluation_points["encourage_contain"],hints_forbid=evaluation_points["forbid_contain"],\
                        evaluation_hints=evaluation_hints,dialogue1=model_dialogue,dialogue2=chatgpt_dialogue)
        input = input.strip()
        evaluation_result = response(input,api_key,gpt_url)
        score = compute_score(evaluation_result)
        score[0],score[1] = score[1] , score[0]
    return input , evaluation_result ,  score

def generate_evaluate_results(chatgpt_result,model_result,datasource,api_key,gpt_url):
    task_names = datasource.keys()

    total_result = {}
    for task_name in task_names:
        # if task_name != "Nursing Knowledge Inquiry":
        #     continue
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
                    evaluate_prompt, evaluate_result,evaluate_score = call_evaluate(chatgpt_item["dialogue"].strip(),model_item["dialogue"].strip(),\
                                                                information,needs,evaluation_hints,evaluation_prompt,api_key,gpt_url)
                    print("--------------------------------------------------")
                    print(evaluate_result)
                    print("--------------------------------------------------")
                    evaluation_temp["task_name"] = chatgpt_item["task_name"]
                    evaluation_temp["id"] = chatgpt_item["id"]
                    evaluation_temp["evaluation_hints"] = evaluation_hints
                    evaluation_temp["information"] = information
                    evaluation_temp["needs"] = needs
                    evaluation_temp["chatgpt_dialogue"] = chatgpt_item["dialogue"]
                    evaluation_temp["model_dialogue"] = model_item["dialogue"]
                    evaluation_temp["evaluation_prompt"] = evaluate_prompt
                    evaluation_temp["evaluation_result"] = evaluate_result
                    evaluation_temp["evaluation_score"] = evaluate_score

                    task_evaluate_temp.append(evaluation_temp)

        total_result[task_name] = task_evaluate_temp
    return total_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args of sft')
    # Model Args
    parser.add_argument('--chatgpt_result_path', type=str)
    parser.add_argument('--model_result_path', type=str)
    parser.add_argument('--datasource_path', type=str)
    parser.add_argument('--result_path', type=str)
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("--api_url", type=str, default="https://api.openai.com/v1/chat/completions", help="OpenAI API URL.")

    args = parser.parse_args()
    chatgpt_result_path = args.chatgpt_result_path
    model_result_path = args.model_result_path
    datasource_path = args.datasource_path
    result_path = args.result_path
    api_key=args.api_key
    gpt_url=args.gpt_url

    ## base_line
    with open(chatgpt_result_path, 'r', encoding='utf-8') as file1:
        chatgpt_result = json.load(file1)
    
    ## 对比模型
    with open(model_result_path, 'r', encoding='utf-8') as file2:
        model_result = json.load(file2)

    ## 源数据，主要用它的 evaluation_hints
    with open(datasource_path, 'r', encoding='utf-8') as file3:
        datasource = json.load(file3)

    ## 
    total_result = generate_evaluate_results(chatgpt_result,model_result,datasource,api_key,gpt_url)
    
    evaluation_result_path = result_path
    with open(evaluation_result_path, 'w', encoding='utf-8') as file:
        json.dump(total_result, file,indent=4,ensure_ascii=False)
    
                

