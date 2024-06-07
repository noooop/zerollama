
import re
from string import Template
from zerollama.microservices.workflow.modular_rag.x001.prompt_template import template


def prompting_messages(t, query):
    content = t.replace('{query}', query)

    messages = [{
        "role": "user",
        "content": content,
    }]
    return messages


if __name__ == '__main__':
    from zerollama.tasks.chat.engine.client import ChatClient
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    client = ChatClient()
    client.wait_service_available(model_name)


    for k, t in template.items():
        print(k)

        messages = prompting_messages(t, query="What kind of university is the school where Rey Ramsey was educated an instance of?")
        response = client.chat(model_name, messages)
        print(response.msg.content)


