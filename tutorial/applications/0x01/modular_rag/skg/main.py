
import re
from string import Template
from zerollama.microservices.workflow.modular_rag.skg.prompt_template import template


def direct_prompting_messages(question):

    content = Template(template.DirectPromptingTemplate).safe_substitute(question=question)

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

    test_direct_prompting = True

    if test_direct_prompting:
        messages = direct_prompting_messages(question="Hello, today is a good day!")
        response = client.chat(model_name, messages)
        print(response.msg.content)

        messages = direct_prompting_messages(question="where is the capital of france?")
        response = client.chat(model_name, messages)
        print(response.msg.content)

        messages = direct_prompting_messages(question="What kind of university is the school where Rey Ramsey was educated an instance of?")
        response = client.chat(model_name, messages)
        print(response.msg.content)


