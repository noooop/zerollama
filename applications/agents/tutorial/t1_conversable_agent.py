from zerollama.agents.core.conversable_agent import ConversableAgent
from zerollama.agents.core.summary_agent import SummaryAgent
from zerollama.agents.core.session import Session

llm_config = {"model": "Qwen/Qwen2-7B-Instruct-GPTQ-Int4", "type": "openai", "base_url": 'http://localhost:8080/v1/'}


cathy = ConversableAgent(
    name="凯西",
    system_message=
    "你的名字是凯西，你是一位单口喜剧演员。",
    llm_config=llm_config,
)

joe = ConversableAgent(
    name="joe",
    system_message=
    "你的名字叫乔，你是一位单口喜剧演员。"
    "从上一个笑话的妙语开始下一个笑话。",
    llm_config=llm_config,
)


session = Session(participants=[joe, cathy])
session.append((joe, "我是乔。凯茜，让我们继续讲笑话吧。"))
session.chat(max_turns=2)

summary_agent = SummaryAgent(llm_config=llm_config)

print("总结:")
print(summary_agent.summary(session))

