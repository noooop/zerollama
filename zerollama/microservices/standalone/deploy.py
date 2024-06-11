
import yaml
import importlib
from easydict import EasyDict as edict
from zerollama.tasks.base.download import download
from zerollama.core.framework.zero_manager.client import ZeroManagerClient
from zerollama.tasks.chat.protocol import Chat_MANAGER_NAME, Chat_ENGINE_CLASS
from zerollama.tasks.retriever.protocol import Retriever_MANAGER_NAME, Retriever_ENGINE_CLASS
from zerollama.tasks.reranker.protocol import Reranker_MANAGER_NAME, Reranker_ENGINE_CLASS
from zerollama.microservices.entrypoints.protocol import Entrypoint_PROTOCOL, Entrypoint_MANAGER_NAME, Entrypoint_ENGINE_CLASS


class Deploy(object):
    Root_MANAGER_NAME = "RootZeroManager"

    def __init__(self, config_path):
        with open(config_path, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.root_manager_client = None
        self.chat_manager_client = None
        self.retriever_manager_client = None

    def verify(self):
        for protocol in ["chat", "retriever", "reranker"]:
            if protocol not in self.config:
                continue

            for i, config in enumerate(self.config[protocol]["models"]):
                if isinstance(config, str):
                    self.config[protocol]["models"][i] = {"name": config}

        if "chat" in self.config:
            for config in self.config["chat"]["models"]:
                if "engine_kwargs" not in config:
                    continue

                engine_kwargs = config["engine_kwargs"]
                use_vllm = engine_kwargs.pop("use_vllm", False)
                if use_vllm:
                    engine_kwargs["inference_backend"] = "zerollama.microservices.inference.vllm_green.chat:VLLMChat"

        self.config = edict(self.config)

    def pull_models(self):
        print("=" * 80)
        print(f'pull models')
        for protocol in ["chat", "retriever", "reranker"]:
            if protocol not in self.config:
                continue

            for model in self.config[protocol]["models"]:
                module_name = f"zerollama.tasks.{protocol}.collection"
                module = importlib.import_module(module_name)
                get_model_by_name = getattr(module, "get_model_by_name")
                download(model.name, get_model_by_name)

    def root_manager_init(self):
        self.root_manager_client = ZeroManagerClient(self.Root_MANAGER_NAME)

    def _manage_init(self, protocol, manage_name, engine_class):
        print("=" * 80)
        print(f'start {manage_name}')
        self.root_manager_client.start(name=manage_name,
                                       engine_kwargs={"server_class": engine_class})

        self.root_manager_client.wait_service_available(manage_name)

        setattr(self, f"{protocol}_manager_client", ZeroManagerClient(manage_name))

    def model_init(self):
        for protocol, manage_name, engine_class in [
            ["chat", Chat_MANAGER_NAME, Chat_ENGINE_CLASS],
            ["retriever", Retriever_MANAGER_NAME, Retriever_ENGINE_CLASS],
            ["reranker", Reranker_MANAGER_NAME, Reranker_ENGINE_CLASS],
        ]:
            if protocol not in self.config:
                continue

            self._manage_init(protocol, manage_name, engine_class)

            for model in self.config[protocol]["models"]:
                client = getattr(self, f"{protocol}_manager_client")
                engine_kwargs = model.get("engine_kwargs", None)
                engine_kwargs = engine_kwargs if engine_kwargs is None else {"engine_kwargs": engine_kwargs}
                client.start(model.name, engine_kwargs=engine_kwargs)

    def http_entrypoint_init(self):
        if "entrypoints" not in self.config:
            return

        protocol, manage_name, engine_class = Entrypoint_PROTOCOL, Entrypoint_MANAGER_NAME, Entrypoint_ENGINE_CLASS
        self._manage_init(protocol, manage_name, engine_class)
        client = getattr(self, f"{protocol}_manager_client")

        if "ollama_compatible" in self.config.entrypoints:
            client.start(name="ollama_compatible",
                         engine_kwargs={
                                     "server_class": "zerollama.microservices.entrypoints.ollama_compatible.api:app",
                                     "server_kwargs": {"port": 11434},
                                })
        if "openai_compatible" in self.config.entrypoints:
            client.start(name="openai_compatible",
                         engine_kwargs={
                                     "server_class": "zerollama.microservices.entrypoints.openai_compatible.api:app",
                                     "server_kwargs": {"port": 8080},
                                })

    def __call__(self):
        self.verify()
        self.pull_models()
        self.root_manager_init()
        self.model_init()
        self.http_entrypoint_init()


if __name__ == '__main__':
    import os
    from pathlib import Path
    path = Path(os.path.dirname(__file__))

    config_path = path / "deploy.yml"
    Deploy(config_path)()
