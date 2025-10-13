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
        elif "chatglm-med" in args.model_path.lower():
            model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
        elif 'chatGLM'in args.model_path or 'chatglm' in args.model_path.lower() or 'bianque' in args.model_path.lower() or 'fuzi-mingcha' in args.model_path.lower():
            model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).half()
        elif 'PMC_LLaMA_13B' in args.model_path or 'zhongjing' in args.model_path:
            model = transformers.LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        elif "baichuan" in args.model_path.lower():
            model = AutoModelForCausalLM.from_pretrained(args.model_path,revision="v2.0",torch_dtype=torch.float16,trust_remote_code=True)
        elif "deepseek-llm-67b" in args.model_path.lower():
            model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True)
            model.generation_config = GenerationConfig.from_pretrained(args.model_path)
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
        elif "Qwen-72B-Chat" in args.model_path:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", trust_remote_code=True).eval()
            # model = AutoModelForCausalLM.from_pretrained("/mntcephfs/lab_data/ganruoli/UC_bench/models/Qwen-72B-Chat", device_map="auto", trust_remote_code=True).eval()
        elif "disc-medllm" in args.model_path.lower() or "DISC-LawLLM" in args.model_path:
            model = AutoModelForCausalLM.from_pretrained("/mntcephfs/lab_data/ganruoli/UC_bench/models/DISC-MedLLM", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        elif "luwen" in args.model_path:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", trust_remote_code=True).half()
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        if "Qwen-72B-Chat" not in args.model_path and "deepseek" not in args.model_path and "DISC-LawLLM" not in args.model_path:
            model.cuda().eval()
        

        if 'PMC_LLaMA_13B' in args.model_path or 'zhongjing' in args.model_path:
            left_tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_path, padding_side='left')
        elif "baichuan" in args.model_path.lower():
            left_tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision="v2.0",trust_remote_code=True, padding_side='left')
        # elif "Yi-34B" in args.model_path:
        #     left_tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        elif "huatuo" in args.model_path.lower():
            left_tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
        elif "Qwen-72B-Chat" in args.model_path:
            left_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        else:
            left_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')

        if 'Qwen-' in args.model_path:
            left_tokenizer.pad_token_id = 151643
            left_tokenizer.eos_token_id = 151643
        if  left_tokenizer.pad_token is None:
            # left_tokenizer.pad_token = '<PAD>'
            left_tokenizer.pad_token = '</s>'
        
        self.model = model
        self.left_tokenizer = left_tokenizer

    def gen(self, query , history = [], model_prompt=""):
        # if history and len(history) == 0:
        #         history = None

        if "deepseek-llm" in args.model_path.lower():
            if history == [] :
                history.append({"role":"system","content":model_prompt})
            history.append({"role": "user", "content": query})
            input_tensor = self.left_tokenizer.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
            outputs = self.model.generate(input_tensor.to(self.model.device), max_new_tokens=1024)
            res = self.left_tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
            history.append({"role": "assistant", "content":res })
            return res , history , ""
        
        if "qwen3" in self.model_path.lower():
            if history == [] :
                history.append({"role":"system","content":model_prompt})
            history.append({"role": "user", "content": query})

            text = self.tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(text, return_tensors="pt")
            response_ids = self.model.generate(**inputs, max_new_tokens=8000)[0][len(inputs.input_ids[0]):].tolist()
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Update history
            history.append({"role": "assistant", "content": response})
            return res, history, ''
    
        if  'qwen'  in args.model_path.lower() or 'internlm' in args.model_path.lower() or 'chatglm' in args.model_path.lower()\
                or "bianque" in args.model_path.lower() or "fuzi-mingcha" in args.model_path.lower():
            if history == [] :
                query = model_prompt + "\n" + query
            res, history = self.model.chat(self.left_tokenizer, query, history=history,max_length=8000)
            return res, history, ''
        
        elif "luwen" in args.model_path:
            prompt = ''
            if history == []:
                query = model_prompt + query
                prompt += f'</s>Human:{query} </s>Assistant:'
            else:
                for i ,(old_query, old_response) in enumerate(history):
                    prompt += f'</s>Human:{old_query} </s>Assistant:{old_response}'
                prompt += f'</s>Human:{query} </s>Assistant:'
            input = prompt
            inputs = self.left_tokenizer(input, return_tensors='pt')
            inputs = inputs.to('cuda')
            pred = self.model.generate(**inputs, max_new_tokens=800, repetition_penalty=1.2)
            response = self.left_tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
            res = response.split("Assistant:")[-1]
            history.append((query,res))
            return res,history,input
        
        elif "baichuan" in args.model_path.lower() or "disc-medllm" in args.model_path.lower() or "huatuo" in args.model_path.lower() or "DISC-LawLLM" in args.model_path:
            if history ==[]:
                history.append({"role": "system", "content": model_prompt})
            history.append({"role": "user", "content": query})
            res = self.model.chat(self.left_tokenizer, history)
            history.append({"role": "assistant", "content":res })
            return res , history , ""

        elif 'Yi' in args.model_path or 'deepseek' in args.model_path.lower():
            if history == []:
                history = [{"role": "system", "content": model_prompt}]
                print(history)
            history.append({"role": "user", "content": query})
            input_ids = self.left_tokenizer.apply_chat_template(conversation=history, tokenize=True, add_generation_prompt=True, return_tensors='pt')
            output_ids = self.model.generate(input_ids.to('cuda'),max_new_tokens=1600)
            res = self.left_tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            history.append({"role": "assistant", "content": res})
            return res,history,""
        
        elif 'mistral-' in args.model_path:
            input = self.dataset.generate_prompt(query,history,model_prompt)
            # input_ids = left_tokenizer(input, return_tensors='pt', add_special_tokens= False)['input_ids']
            input_ids = self.left_tokenizer(input, return_tensors="pt")
            attention_mask = self.left_tokenizer(input, return_tensors='pt', add_special_tokens= False)["attention_mask"]
            # Generate
            generate_ids = self.model.generate(input_ids.input_ids, max_length=1000)
            res = self.left_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            if history:
                history.append((query,res))
            else:
                history = [(query,res)]
            return res,history,input


def mt_dialogue_gen(data_path,model_path,result_path):
    accelerator = Accelerator()
    torch.cuda.set_device(accelerator.process_index)
    ## 模型
    llm = LLM(model_path)
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
                gpt = GPTPerson(data=data,model_name=args.model_name,gpt_url=args.api_url,api_key=args.api_key)
                query = gpt.initial_response()#role prompt要求GPT角色扮演
                dialogue = dialogue + "用户：" + query + "\n"
                print("\n-----------------------------------query\n"+query)
                history = []
                for round in range(1,4):
                    data["dialogue_round"] = round  ## 用于记录轮次 
                    ## AI 助手生成回复
                    res,history,input = llm.gen(query,history,model_prompt)
                    res = res.strip()
                    print("\n-----------------------------------res\n"+res)
                    dialogue = dialogue + "AI助手：" + res + "\n"
                    
                    ##### 防止最后一轮 GPT 提问
                    if round == 3:
                        break

                    ## user 继续提问
                    query = gpt.response(res)
                    query = query.strip()
                    if "咨询结束" in query:
                        break
                    dialogue = dialogue + "用户：" + query + "\n"

                ## 结果 data
                data['dialogue'] = dialogue
                data['model'] = model_path
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
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-11-20", help="Name of the GPT model to use.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("--api_url", type=str, default="https://api.openai.com/v1/chat/completions", help="OpenAI API URL.")
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    set_seed(args.seed)
    mt_dialogue_gen(args.data_path,args.model_path,args.result_path)
