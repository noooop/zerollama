from openai import OpenAI

client = OpenAI(base_url='http://localhost:8080/v1/', api_key="empty")

completion = client.chat.completions.create(
  model="Qwen/Qwen2-7B-Instruct-AWQ",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message.content)