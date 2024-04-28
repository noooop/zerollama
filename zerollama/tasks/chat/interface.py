
from zerollama.tasks.base.interface import ModelBase
from zerollama.tasks.chat.protocol import PROTOCOL
from zerollama.tasks.chat.protocol import ChatModelConfig, ChatCompletionResponse, ChatCompletionStreamResponse


class ChatModel(ModelBase):
    protocol = PROTOCOL
    inference_backend = "zerollama.inference_backend.hf_transformers.chat:HuggingFaceTransformersChat"

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
    protocol = PROTOCOL

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