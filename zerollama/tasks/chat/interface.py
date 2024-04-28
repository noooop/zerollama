
from zerollama.tasks.base.interface import ModelBase
from zerollama.tasks.chat.protocol import PROTOCOL
from zerollama.tasks.chat.protocol import ChatModelConfig, ChatCompletionResponse, ChatCompletionStreamResponse


class ChatModel(ModelBase):
    protocol = PROTOCOL
    inference_backend = "zerollama.inference_backend.hf_transformers.chat:HuggingFaceTransformersChat"

    @classmethod
    def get_model_config(cls, model_name):
        model_config = super().get_model_config(model_name)
        if model_config is None:
            return

        return ChatModelConfig(**model_config)


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