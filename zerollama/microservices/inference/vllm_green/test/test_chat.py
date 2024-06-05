
import time
import shortuuid
from gevent.pool import Pool
from zerollama.microservices.inference.vllm_green.chat import VLLMChat, ChatCompletionStreamResponseDone


model_name = "Qwen/Qwen1.5-0.5B-Chat"
model = VLLMChat(model_name, local_files_only=False)
model.load()


def worker(prompt):
    request_id = f"{shortuuid.random(length=22)}"

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = model.chat(messages, stream=True)

    generated_text = ""
    for output in response:
        if not isinstance(output, ChatCompletionStreamResponseDone):
            generated_text += output.delta_content
            print(f"ID:{request_id}, Generated text: {generated_text!r}")


print("="*10)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


p = Pool(2)
for x in p.imap_unordered(worker, prompts):
    pass

time.sleep(10)

p = Pool(2)
for x in p.imap_unordered(worker, prompts):
    pass