from zerollama.agents.core.conversable_agent import ConversableAgent

model = "Qwen/Qwen2-7B-Instruct-GPTQ-Int4"
prompt = "给我介绍一下大型语言模型。"

messages = [
    {"role": "user", "content": prompt}
]

for llm_config in [
    {"type": "zerollama", "model": model},
    {"type": "openai", "model": model, "base_url": 'http://localhost:8080/v1/'},
    {"type": "ollama", "model": model}

]:
    print("-" * 80)
    print(llm_config)
    agent = ConversableAgent(
        name="chatbot",
        llm_config=llm_config,
    )

    prompt = "给我介绍一下大型语言模型。"

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
