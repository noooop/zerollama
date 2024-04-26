from pydantic import BaseModel, ConfigDict
from typing import Optional


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


class ChatModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = "chat"
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )


class ChatModel(object):
    family = ""
    protocol = "chat"
    model_kwargs = {}
    header = []
    info = []

    @classmethod
    def model_names(cls):
        return [x[0] for x in cls.info]

    @classmethod
    def get_model_config(cls, model_name):
        info_dict = {x[0]: {k: v for k, v in zip(cls.header, x)} for x in cls.info}

        if model_name not in info_dict:
            return None

        info = info_dict[model_name]
        info.update({"family": cls.family, "protocol": cls.protocol})

        chat_model_config = ChatModelConfig(**{
            "name": model_name,
            "info": info,
            "family": cls.family,
            "protocol": cls.protocol,
            "model_kwargs": cls.model_kwargs})

        return chat_model_config

    @classmethod
    def prettytable(cls):
        from prettytable import PrettyTable
        table = PrettyTable(cls.header + ["family", "protocol"], align='l')

        for x in cls.info:
            table.add_row(x+[cls.family, cls.protocol])

        return table


class ChatInterface(object):
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