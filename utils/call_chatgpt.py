import requests, json

def call_chatgpt(prompt, system="",model:str="gpt-4"):
    url = "xxx"
    api_key = "xxx"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {
                'role': 'system',
                'content': system
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.8,
    }

    raw_response = requests.post(url, headers=headers, json=payload, verify=False)
    response = json.loads(raw_response.content.decode("utf-8"))

    return response['choices'][0]['message']['content']

if __name__ == "__main__":

    string1 = \
    """hello"""
    response = call_chatgpt(string1)
    print(response)


