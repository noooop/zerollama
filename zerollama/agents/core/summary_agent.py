from zerollama.agents.core.conversable_agent import ConversableAgent


class SummaryAgent(ConversableAgent):
    DEFAULT_SYSTEM_MESSAGE = "Summarize the takeaway from the conversation. Do not add any introductory phrases."

    def __init__(self,
                 name: str = "SummaryAgent",
                 system_message=DEFAULT_SYSTEM_MESSAGE,
                 llm_config=None,
                 description=None):
        super().__init__(name, system_message, llm_config, description)

    def summary(self, session):
        return session.summary(self)
