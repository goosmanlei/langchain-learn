import os
from openai import OpenAI

client = OpenAI(
    api_key = os.environ.get('DEEPSEEK_API_KEY'),
    base_url= 'https://api.deepseek.com',
)

def get_completion(input, model = 'deepseek-chat'):
    response = client.chat.completions.create(
        model = model,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant'},
            {'role': 'user', 'content': input},
        ],
        stream=False,
    )
    return response.choices[0].message.content

print(get_completion('What is 1+1?'))