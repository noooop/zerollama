from ollama import chat


messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

response = chat('Qwen/Qwen1.5-0.5B-Chat-AWQ', messages=messages)
print(response['message']['content'])
