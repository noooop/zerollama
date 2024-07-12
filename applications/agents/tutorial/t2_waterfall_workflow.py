from zerollama.agents import ConversableAgent
from zerollama.agents import SummaryAgent
from zerollama.agents import Session
from zerollama.agents import UserInput

llm_config = {"model": "Qwen/Qwen2-7B-Instruct-GPTQ-Int4", "type": "openai", "base_url": 'http://localhost:8080/v1/'}

onboarding_personal_information_agent = ConversableAgent(
    name="Onboarding Personal Information Agent",
    system_message='''您是一位乐于助人的客户引导代理，
    您在这里是为了帮助新客户开始使用我们的产品。
    您的工作仅收集客户姓名和目的地。不要询问其他信息。也不要跟顾客攀谈。
    您的工作仅收集客户姓名和目的地。不要询问其他信息。也不要跟顾客攀谈。
    您的工作仅收集客户姓名和目的地。不要询问其他信息。也不要跟顾客攀谈。
    
    一定要客户姓名和目的地，这是必须项。你需要反复更客户沟通收集客户姓名和目的地。
    一定要客户姓名和目的地，这是必须项。你需要反复更客户沟通收集客户姓名和目的地。
    一定要客户姓名和目的地，这是必须项。你需要反复更客户沟通收集客户姓名和目的地。
    
    当你收集完所有信息后，祝客户旅途愉快。返回“TERMINATE”。
    当你收集完所有信息后，祝客户旅途愉快。返回“TERMINATE”。
    当你收集完所有信息后，祝客户旅途愉快。返回“TERMINATE”。''',
    llm_config=llm_config,
)

onboarding_personal_information_summary_agent = SummaryAgent(
    system_message="Return the customer information "
                   "into as JSON object only: "
                   "{'name': '', 'location': ''}",
    llm_config=llm_config)

onboarding_topic_preference_agent = ConversableAgent(
    name="Onboarding Topic preference Agent",
    system_message='''您是一位乐于助人的客户引导代理，
    您在这里是为了帮助新客户开始使用我们的产品。
    
    您的工作仅收集客户感兴趣的内容。不要询问其他信息。不要跟用户聊客户感兴趣的内容细节，也不要跟顾客攀谈。
    您的工作仅收集客户感兴趣的内容。不要询问其他信息。不要跟用户聊客户感兴趣的内容细节，也不要跟顾客攀谈。
    您的工作仅收集客户感兴趣的内容。不要询问其他信息。不要跟用户聊客户感兴趣的内容细节，也不要跟顾客攀谈。
    
    当你收集完所有信息后，祝客户旅途愉快。返回“TERMINATE”。
    当你收集完所有信息后，祝客户旅途愉快。返回“TERMINATE”。
    当你收集完所有信息后，祝客户旅途愉快。返回“TERMINATE”。''',
    llm_config=llm_config,
)


onboarding_topic_preference_summary_agent = SummaryAgent(
    system_message="Return the customer information "
                   "into as JSON object only: "
                   "{'interests': ''}",
    llm_config=llm_config)


customer_engagement_agent = ConversableAgent(
    name="Customer Engagement Agent",
    system_message='''您是一位乐于助人的客户服务代理
    在这里根据用户的需求为客户提供乐趣
    个人信息和主题偏好。
    这可能包括有趣的事实、笑话或有趣的故事。
    确保让它引人入胜且有趣！''',
    llm_config=llm_config,
)

customer = ConversableAgent(
    name="Customer",
    system_message='''你是一名顾客。
    你的姓名叫小王，目的地是成都，你喜欢的动漫和电影。
    ''',
    llm_config=llm_config
)


def agent_pipeline(customer):
    session1 = Session(participants=[onboarding_personal_information_agent, customer],
                       termination_msg="terminate")
    session1.append(
        (onboarding_personal_information_agent, "您好，我是来帮助您开始使用我们的产品的。你能告诉我你的名字和目的地吗？"))
    session1.chat(max_turns=5)

    name_location = onboarding_personal_information_summary_agent.summary(session1)

    print("客户的姓名和目的地:")
    print(name_location)

    session2 = Session(participants=[onboarding_topic_preference_agent, customer],
                       termination_msg="terminate")
    session2.append((onboarding_topic_preference_agent, "棒!你能告诉我你对什么主题感兴趣。"))
    session2.chat(max_turns=5)

    interests = onboarding_topic_preference_summary_agent.summary(session2)

    print("客户的感兴趣的主题:")
    print(interests)

    session3 = Session(participants=[customer_engagement_agent, customer],
                       termination_msg="terminate")
    session3.append((customer, f"客户个人信息: \n {name_location} \n\n 客户感兴趣的主题: \n {interests} \n\n"))
    session3.chat(max_turns=5)


agent_pipeline(customer)

customer = UserInput()
agent_pipeline(customer)


