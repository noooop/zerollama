from applications.chat_webui.openai_client_examples.tiny_openai import OpenAI

client = OpenAI()


completion = client.chat.completions.create(
  model='Qwen/Qwen1.5-0.5B-Chat-AWQ',
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  stream=True
)

for chunk in completion:
  print(chunk.choices[0].delta.content, end="", flush=True)