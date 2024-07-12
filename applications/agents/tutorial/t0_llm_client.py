import os
import sys
from pathlib import Path
pwd = Path(os.getcwd())
sys.path.append(str(pwd.parent.parent.parent))

from zerollama.agents import ConversableAgent

model = "Qwen/Qwen2-7B-Instruct-GPTQ-Int4"
prompt = "给我介绍一下大型语言模型。"

for llm_config in [
    {"type": "zerollama", "model": model, "global_priority": False},
    {"type": "openai", "model": model, "base_url": 'http://localhost:8080/v1/', "global_priority": False},
    {"type": "ollama", "model": model, "global_priority": False},

]:
    print("-" * 80)
    print(llm_config)
    agent = ConversableAgent(
        name="chatbot",
        llm_config=llm_config,
    )

    print("stream=False")
    reply = agent.generate_reply(
        messages=[{"content": prompt, "role": "user"}],
        stream=False,
        options={"max_tokens": 1000}
    )

    print(reply)

    print("stream=True")
    for part in agent.generate_reply(
        messages=[{"content": prompt, "role": "user"}],
        stream=True,
        options={"max_tokens": 1000}
    ):
        print(part, end="", flush=True)
    print()
