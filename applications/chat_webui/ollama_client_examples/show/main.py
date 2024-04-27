from ollama import show

response = show('Qwen/Qwen1.5-0.5B-Chat-AWQ')
print(response)
response = show('llama2:latest')
print(response)
