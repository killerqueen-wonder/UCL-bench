import requests
import threading
from retrying import retry
import json
from openai import OpenAI

class GPTPerson():
    def __init__(self, data ,model_name,gpt_url,api_key,temperature=0.1):
        
        self.api_key = api_key
        self.model_name = model_name
        self.role = data["role_prompt"]
        self._initial_person()
        self.gpt_url = gpt_url
        self.temperature = temperature
        print(f"Using model: {self.model_name}")

    def _initial_person(self):
        print(f'[debug]init person')
        self.temp_messages = [{"role": "system", "content": self.role}]

    @retry(wait_fixed=200, stop_max_attempt_number=10)
    def call_api(self):
        # parameters = {
        #     "model": self.model_name,
        #     "messages": self.temp_messages
        # }
        # headers = {
        #     "Content-Type": "application/json",
        #     "Authorization": f"Bearer {self.api_key}"
        # }
        # response = requests.post(
        #     self.gpt_url,
        #     headers=headers,
        #     json=parameters
        # ).json()

        # if 'choices' not in response and 'error' in response:
        #     raise Exception(response['error']['message'] + '\n' + 'apikey:'+self.api_key)
        
        client = OpenAI(api_key=self.api_key, base_url=self.api_url)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": self.temp_messages}],
            # messages=[{"role": "user", "content": self.temp_messages}],
            temperature=self.temperature,
            stream=False
        )
        # response_text = response["choices"][0]["message"]["content"].strip()
        response_text = response.choices[0].message.content.strip()

        if 'error' in response_text:
            raise ValueError(f"API Error: {response_text}")

        return response_text
    

    @retry(wait_fixed=2000, stop_max_attempt_number=50)
    def call_api_timelimit(self):
        class InterruptableThread(threading.Thread):
            def __init__(self,temp_messages,api_key,model_name,gpt_url):
                threading.Thread.__init__(self)
                self.result = None
                self.temp_messages = temp_messages
                self.api_key = api_key
                self.model_name = model_name
                self.gpt_url=gpt_url

            def run(self):
                try:
                    # parameters = {
                    # "model": self.model_name,
                    # "messages": self.temp_messages
                    # }
                    # headers = {
                    #     "Content-Type": "application/json",
                    #     "Authorization": f"Bearer {self.api_key}"
                    # }
                    # response = requests.post(
                    #     self.gpt_url,
                    #     headers=headers,
                    #     json=parameters,
                    # ).json()
                    # if 'choices' not in response and 'error' in response:
                    #     raise Exception(response['error']['message'] + '\n' + 'apikey:'+self.api_key)
                    client = OpenAI(api_key=self.api_key, base_url=self.gpt_url)

                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": self.temp_messages}],
                        # messages=[{"role": "user", "content": self.temp_messages}],
                        temperature=0.2,
                        stream=False
                    )
                    # response_text = response["choices"][0]["message"]["content"].strip()
                    response_text = response.choices[0].message.content.strip()

                    self.result = response_text
                except Exception as e:
                    print(e)

        it = InterruptableThread(self.temp_messages,self.api_key,self.model_name,self.gpt_url)
        it.start()
        # 时间
        timeout_duration = 200
        it.join(timeout_duration)
        if it.is_alive() or it.result is None:
            print('时间进程出错')
            raise Exception("API调用超时")
        else:
            return it.result

    def response(self, message):
        self.temp_messages.append({"role": "user", "content": message})
        # response_text = self.call_api()
        try:
            # response_text = self.call_api()
            response_text = self.call_api_timelimit()
        except Exception as e:
            response_text = ""
        self.temp_messages.append({"role": "assistant", "content": response_text})
        return response_text
    
    def initial_response(self):
        try:
            # response_text = self.call_api()
            response_text = self.call_api_timelimit()
        except Exception as e:
            response_text = ""
        self.temp_messages.append({"role": "assistant", "content": response_text})
        return response_text

    def characters(self):
        return self.temp_messages
    
class GPTTest():   ## 需要多轮对话的 
    def __init__(self, model_name = "gpt-4",data={}):
        self.model_name = model_name
        self.data = data
        with open("config/configuration.json", 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.api_key = data["gpt_key"]
        if "{information}" in self.data["model_prompt"]:
            self.temp_messages = [{"role": "system", "content": self.data["model_prompt"].format(information=self.data["information"])}]
        else :
            self.temp_messages = [{"role": "system", "content": self.data["model_prompt"]}]
    @retry(wait_fixed=2000, stop_max_attempt_number=10)
    def call_api_timelimit(self):
        class InterruptableThread(threading.Thread):
            def __init__(self,temp_messages,api_key,model_name):
                threading.Thread.__init__(self)
                self.result = None
                self.temp_messages = temp_messages
                self.api_key = api_key
                self.model_name = model_name

            def run(self):
                try:
                    parameters = {
                    "model": self.model_name,
                    "messages": self.temp_messages
                    }
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    response = requests.post(
                        self.gpt_url,
                        headers=headers,
                        json=parameters,
                    ).json()
                    if 'choices' not in response and 'error' in response:
                        raise Exception(response['error']['message'] + '\n' + 'apikey:'+self.api_key)
                    
                    response_text = response["choices"][0]["message"]["content"].strip()
                    self.result = response_text
                except Exception as e:
                    print(e)

        it = InterruptableThread(self.temp_messages,self.api_key,self.model_name)
        it.start()
        # 时间
        timeout_duration = 200
        it.join(timeout_duration)
        if it.is_alive() or it.result is None:
            print('时间进程出错')
            raise Exception("API调用超时")
        else:
            return it.result

    def response(self, message):
        self.temp_messages.append({"role": "user", "content": message})
        try:
            response_text = self.call_api_timelimit()
        except Exception as e:
            response_text = ""
        self.temp_messages.append({"role": "assistant", "content": response_text})
        return response_text

    def characters(self):
        return self.temp_messages

if __name__ == "__main__":
    test = GPTTest(model_name ="gpt-4")
    res = test.response("who are you?")
    print(res)


