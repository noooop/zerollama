
import gradio as gr
from zerollama.tasks.chat.engine.client import ChatClient
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponseDone

chat_client = ChatClient()

response = chat_client.get_service_names()
services = response.msg["service_names"]

model = services[0]


def predict(message, history):
    messages = []
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    partial_message = ""
    for rep in chat_client.stream_chat(model, messages):
        if not isinstance(rep.msg, ChatCompletionStreamResponseDone):
            partial_message += rep.msg.delta_content
            yield partial_message


gr.ChatInterface(predict).launch()