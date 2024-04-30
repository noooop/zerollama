from applications.chat_webui.openai_client_examples.tiny_openai import OpenAI
client = OpenAI()

print(client.embeddings.create(
  model="BAAI/bge-m3",
  input="The food was delicious and the waiter...",
  encoding_format="float"
))