from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8080/v1',
    api_key='...', # required, but unused
)

response = client.chat.completions.create(
  model="llama2",
  messages=[
      {
          "role": "system",
          "content": "You are a helpful assistant."
      },
      {
          "role": "user",
          "content": "Hello!"
      }
  ]
)
print(response.choices[0].message.content)