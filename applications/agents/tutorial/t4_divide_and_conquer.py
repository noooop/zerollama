
import gevent
from zerollama.agents import AssistantAgent
from zerollama.agents import Session
from zerollama.agents import SummaryAgent
from zerollama.agents import Agent


llm_config = {"model": "Qwen/Qwen2-7B-Instruct-GPTQ-Int4", "type": "openai", "base_url": 'http://localhost:8080/v1/'}


writer = AssistantAgent(
    name="Writer",
    system_message="你是一名作家。"
                   "您就给定主题撰写引人入胜且简洁的博客文章（带标题）。"
                   "您必须根据收到的反馈完善您的写作并给出完善的版本。"
                   "仅返回您的最终作品，无需附加评论。仅返回您的最终作品，无需附加评论。仅返回您的最终作品，无需附加评论。",
    llm_config=llm_config,
)


class CriticAgent(Agent):
    def __init__(self, name: str = "Critic"):
        super().__init__(name, None)

        self.SEO_reviewer = AssistantAgent(
            name="SEO Reviewer",
            llm_config=llm_config,
            system_message="您是一名 SEO 审核员，以优化搜索引擎内容的能力而闻名，"
                           "确保其排名良好并吸引自然流量。确保您的建议简洁（3 个要点以内）、具体且切题。"
                           "通过陈述您的角色来开始审查。",
        )

        self.legal_reviewer = AssistantAgent(
            name="Legal Reviewer",
            llm_config=llm_config,
            system_message="您是一名法律审核员，以确保内容合法且不存在任何潜在法律问题的能力而闻名。"
                           "确保您的建议简洁（3 个要点以内）、具体且切中要点。"
                           "通过陈述您的角色来开始审查。",
        )

        self.ethics_reviewer = AssistantAgent(
            name="Ethics Reviewer",
            llm_config=llm_config,
            system_message="您是一名道德审核员，以确保内容符合道德规范且不存在任何潜在道德问题的能力而闻名。"
                           "确保您的建议简洁（3 个要点以内）、具体且切中要点。"
                           "通过陈述您的角色来开始审查。",
        )

        self.summary_agent = SummaryAgent(
            system_message="Return review into as JSON object only:"
                           "{'审稿人': '', '评论': ''}.",
            llm_config=llm_config)

        self.meta_reviewer = AssistantAgent(
            name="Meta Reviewer",
            llm_config=llm_config,
            system_message="您是元审核员，您汇总和审阅其他审核员的工作，并对内容提出最终建议。",
        )

        self.critices = [self.SEO_reviewer, self.legal_reviewer, self.ethics_reviewer]

    def generate_reply(self, messages, stream=False, options=None):
        article = messages[-1]["content"]

        # divide
        def review_and_summary(critic, article):
            session = Session(participants=[self.summary_agent, critic])
            session.append((self.summary_agent, article))
            session.chat(max_turns=1, verbose=False, verbose_history=False)

            summary = self.summary_agent.summary(session)
            return summary

        review = gevent.joinall([gevent.spawn(review_and_summary, critic, article) for critic in self.critices])
        review = "\n".join([x.value for x in review])

        # conquer
        session2 = Session(participants=[self.summary_agent, self.meta_reviewer])
        session2.append((self.summary_agent, review))
        session2.chat(max_turns=1)

        return session2.history[-1].content


critic = CriticAgent(name="Critic")


def discuss_and_improve(task):
    session = Session(participants=[writer, critic])
    session.append((critic, task))
    session.chat(max_turns=5)
    return session


task = '''
    写一篇简洁但引人入胜的关于 DeepLearning.AI 博客文
       DeepLearning.AI. 确保文章在100字以内。
'''

discuss_and_improve(task)