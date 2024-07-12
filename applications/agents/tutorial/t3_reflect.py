from zerollama.agents import AssistantAgent
from zerollama.agents import Session

llm_config = {"model": "Qwen/Qwen2-7B-Instruct-GPTQ-Int4", "type": "openai", "base_url": 'http://localhost:8080/v1/'}


task = '''
    写一篇简洁但引人入胜的关于 DeepLearning.AI 博客文
       DeepLearning.AI. 确保文章在100字以内。
'''


writer = AssistantAgent(
    name="Writer",
    system_message="你是一名作家。"
                   "您就给定主题撰写引人入胜且简洁的博客文章（带标题）。"
                   "您必须根据收到的反馈完善您的写作并给出完善的版本。"
                   "仅返回您的最终作品，无需附加评论。仅返回您的最终作品，无需附加评论。仅返回您的最终作品，无需附加评论。",
    llm_config=llm_config,
)

reply = writer.generate_reply(task)
print("第一版")
print(reply)

critic = AssistantAgent(
    name="Critic",
    llm_config=llm_config,
    system_message="你是一个批评家。"
                   "您审查作者的作品并提供建设性的反馈，以帮助提高内容的质量。"
                   "指出亮点和不足并至少提三点意见。"
)


print("+"*80)
print("自我反省，迭代提高")
print("+"*80)


def discuss_and_improve(task):
    session = Session(participants=[writer, critic])
    session.append((critic, task))
    session.chat(max_turns=5)
    return session


discuss_and_improve(task)