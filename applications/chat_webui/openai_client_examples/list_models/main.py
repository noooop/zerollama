from pprint import pprint

from applications.chat_webui.openai_client_examples.tiny_openai import OpenAI

client = OpenAI()

pprint(client.models.list())