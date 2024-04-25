from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Any, Union


class ChatCompletionResponse(BaseModel):
    model: str
    prompt_length: int
    response_length: int
    finish_reason: str
    content: str


class ChatCompletionStreamResponse(BaseModel):
    model: str
    prompt_length: int
    response_length: int
    finish_reason: Optional[str] = None
    content: Optional[str] = None
    done: bool


class ChatInterfaces(object):
    protocol = "chat"

    def load(self):
        """
        Load everything in memory.
        :return: None
        """
        raise NotImplementedError

    def chat(self, messages, options=None) -> ChatCompletionResponse:
        """

        :param messages:
        :param options:
        :return:
        """
        raise NotImplementedError

    def stream_chat(self, messages, options=None) -> ChatCompletionStreamResponse:
        """

        :param messages:
        :param options:
        :return:
        """
        raise NotImplementedError