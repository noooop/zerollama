from pprint import pprint
from applications.chat_webui.openai_client_examples.tiny_openai import OpenAI

client = OpenAI()

pprint(client.models.retrieve('Qwen/Qwen1.5-0.5B-Chat-AWQ'))
pprint(client.models.retrieve('llama2:latest'))