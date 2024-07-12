
import inspect
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

from zerollama.agents.core.chat_client import get_client
from zerollama.agents.core.agent import Agent
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponseDone


class ConversableAgent(Agent):
    def __init__(self,
                 name: str,
                 system_message: Optional[Union[str, List]] = "You are a helpful AI Assistant.",
                 llm_config: Optional[Union[Dict, Literal[False]]] = None,
                 description: Optional[str] = None):
        super().__init__(name, description if description is not None else system_message)

        self.llm_config = llm_config
        self._chat_client = get_client(self.llm_config)
        self._model_name = self._chat_client.model_name

        self._system_message = [{"content": system_message, "role": "system"}]

    def generate_reply(self, messages, stream=False, options=None):
        if isinstance(messages, list):
            response = self._chat_client.chat(self._system_message+messages, stream, options)
        elif isinstance(messages, str):
            messages = [{"content": messages, "role": "user"}]
            response = self._chat_client.chat(self._system_message + messages, stream, options)

        if not inspect.isgenerator(response):
            return response.content
        else:
            def generator():
                for rep in response:
                    if not isinstance(rep, ChatCompletionStreamResponseDone):
                        yield rep.delta_content

            return generator()


if __name__ == '__main__':
    llm_config = {"type": "zerollama", "model": "Qwen/Qwen2-7B-Instruct-GPTQ-Int4"}
    agent = ConversableAgent(
        name="chatbot",
        llm_config=llm_config,
    )

    reply = agent.generate_reply(
        messages=[{"content": "给我讲个笑话。", "role": "user"}]
    )
    print("-" * 80)
    print(reply)
