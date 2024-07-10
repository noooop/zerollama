
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

from zerollama.agents.core.chat_client import get_client
from zerollama.agents.core.agent import Agent
from zerollama.agents.core.session import Session


class ConversableAgent(Agent):
    def __init__(self,
                 name: str,
                 system_message: Optional[Union[str, List]] = "You are a helpful AI Assistant.",
                 llm_config: Optional[Union[Dict, Literal[False]]] = None,
                 description: Optional[str] = None,
                 human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",):
        super().__init__(name, description if description is not None else system_message)

        self.human_input_mode = human_input_mode

        self.llm_config = llm_config
        self._chat_client = get_client(self.llm_config)
        self._model_name = llm_config["model"]

        self._system_message = [{"content": system_message, "role": "system"}]

    def generate_reply(self, messages, stream=False, options=None):
        if isinstance(messages, list):
            response = self._chat_client.chat(self._system_message+messages, stream, options)
        elif isinstance(messages, str):
            messages = [{"content": messages, "role": "user"}]
            response = self._chat_client.chat(self._system_message + messages, stream, options)

        return response.content


if __name__ == '__main__':
    llm_config = {"type": "zerollama", "model": "Qwen/Qwen2-7B-Instruct-AWQ"}
    agent = ConversableAgent(
        name="chatbot",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    reply = agent.generate_reply(
        messages=[{"content": "给我讲个笑话。", "role": "user"}]
    )
    print("-" * 80)
    print(reply)
