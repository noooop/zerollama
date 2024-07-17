
from zerollama.tasks.base.interface import ModelBase
from zerollama.tasks.chat.protocol import PROTOCOL
from zerollama.tasks.chat.protocol import ChatModelConfig


class ChatModel(ModelBase):
    protocol = PROTOCOL
    inference_backend = "zerollama.microservices.inference.transformers_green.chat:HuggingFaceTransformersChat"
    download_backend = "zerollama.microservices.inference.transformers_green.download:download"

    @classmethod
    def get_model_config(cls, model_name):
        model_config = super().get_model_config(model_name)
        if model_config is None:
            return

        return ChatModelConfig(**model_config)


class ChatGGUFModel(ChatModel):
    inference_backend = "zerollama.microservices.inference.llama_cpp_green.chat:LLamaCPPChat"
    download_backend = "zerollama.microservices.inference.llama_cpp_green.download:download"

    gguf = {"repo_id": [], "filename": []}

    @classmethod
    def model_names(cls):
        o = []
        for r in cls.gguf["repo_id"]:
            for f in cls.gguf["filename"]:
                o.append(f"{r}+{f}")
        return o

    @classmethod
    def get_model_config(cls, model_name):
        info_dict = dict()
        for r in cls.gguf["repo_id"]:
            for f in cls.gguf["filename"]:
                name = f"{r}+{f}"
                info_dict[name] = {
                    "name": name,
                    "repo_id": r,
                    "filename": f
                }

        if model_name not in info_dict:
            return None

        info = info_dict[model_name]
        info.update({"family": cls.family, "protocol": cls.protocol})

        model_config = {
            "name": model_name,
            "info": info,
            "family": cls.family,
            "protocol": cls.protocol,
            "model_kwargs": cls.model_kwargs}

        return ChatModelConfig(**model_config)

    @classmethod
    def prettytable(cls):
        from prettytable import PrettyTable
        table = PrettyTable(["name", "repo_id", "family", "filename", "protocol"], align='l')

        for r in cls.gguf["repo_id"]:
            for f in cls.gguf["filename"]:
                name = f"{r}+{f}"
                table.add_row([name, r, f, cls.family, cls.protocol])

        return table


class ChatInterface(object):
    protocol = PROTOCOL

    def load(self):
        """
        Load everything in memory.
        :return: None
        """
        raise NotImplementedError

    def chat(self, messages, tools=None, stream=False, options=None):
        raise NotImplementedError

    def stream_chat(self, messages, options=None):
        for item in self.chat(messages, stream=True, options=options):
            yield item