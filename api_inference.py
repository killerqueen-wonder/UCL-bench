import argparse
from utils.user_simulator import GPTPerson,GPTTest
import json


def test_gpt(model_name,total_data):
    test_result = {}
    for task_name in total_data.keys():

        test_task_temp = []
        for data in total_data[task_name]:

            gpt_user = GPTPerson(data=data,model_name="gpt-4-0613")
            gpt_test = GPTTest(model_name = model_name,data=data)
            query = gpt_user.initial_response()
            dialogue =  "用户：" + query + "\n"

            turn_num = 5
            for round in range(1,turn_num):
                data["dialogue_round"] = round
                ## AI 助手生成回复
                res = gpt_test.response(query)
                res = res.strip()
                print(res)
                dialogue = dialogue + "AI助手：" + res + "\n"

                
                ### 最多生成三轮对话
                if round == 3:
                    break

                ## user 继续提问
                query = gpt_user.response(res)
                query = query.strip()
                print(query)
                if "咨询结束" in query:
                    break
                dialogue = dialogue + "用户：" + query + "\n"

            ## 结果 data 
            data['dialogue'] = dialogue
            data['model'] = model_name
            test_task_temp.append(data)
        test_result[task_name] = test_task_temp
    return test_result

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Args of sft')
    # Model Args
    parser.add_argument('--data_path', default='/mntcephfs/lab_data/ganruoli/UC_bench/dataset/legal_data/legal_data_sample.json', type=str)
    parser.add_argument('--model_path', default='/mntcephfs/data/med/zhanghongbo/MOSS/junying_models/chatglm3-6b', type=str)
    args = parser.parse_args()

    ## 数据地址
    data_path = args.data_path
    ## 模型地址
    model_name = args.model_path

    ## 读取数据
    with open(data_path, 'r', encoding='utf-8') as file:
        total_data = json.load(file)

    data_list = test_gpt(model_name,total_data)

    if "legal" in args.data_path:
        with open(model_name +"_legal.json", 'w', encoding='utf-8') as file:
            json.dump(data_list, file,indent=4,ensure_ascii=False)