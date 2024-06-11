
import yaml
import os
from pathlib import Path
import shortuuid
from gevent.pool import Pool
from zerollama.tasks.chat.engine.client import ChatClient
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponseDone

path = Path(os.path.dirname(__file__))

config_path = path.parent / "deploy.yml"

with open(config_path, 'r', encoding="utf-8") as f:
    config = yaml.safe_load(f)

for config in config["chat"]["models"]:
    if isinstance(config, str):
        model_name = config
    else:
        model_name = config["name"]

    client = ChatClient()
    print("=" * 80)
    print(f"Wait {model_name} available")
    client.wait_service_available(model_name)
    print(client.get_services(model_name))


    def worker(prompt):
        request_id = f"{shortuuid.random(length=22)}"
        messages = [
            {"role": "user", "content": prompt}
        ]
        generated_text = ""
        for output in client.stream_chat(model_name, messages):
            if not isinstance(output, ChatCompletionStreamResponseDone):
                generated_text += output.delta_content
                print(f"ID:{request_id}, Generated text: {generated_text!r}")


    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    p = Pool(32)
    for x in p.imap_unordered(worker, prompts * 100):
        pass