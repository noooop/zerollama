from pydantic import BaseModel, ConfigDict
from typing import Optional
from zerollama.core.models.base import ModelBase


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


class ChatModel(ModelBase):
    protocol = "chat"
    inference_backend = "zerollama.inference_backend.hf_transformers.main:HuggingFaceTransformersChat"

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