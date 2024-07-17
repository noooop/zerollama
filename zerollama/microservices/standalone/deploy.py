import yaml
import importlib
from easydict import EasyDict as edict
from zerollama.tasks.base.download import get_pretrained_model_name
from zerollama.core.framework.zero_manager.client import ZeroManagerClient
from zerollama.tasks.chat.protocol import Chat_ENGINE_CLASS
from zerollama.tasks.retriever.protocol import Retriever_ENGINE_CLASS
from zerollama.tasks.reranker.protocol import Reranker_ENGINE_CLASS
from zerollama.microservices.entrypoints.protocol import Entrypoint_ENGINE_CLASS


class Deploy(object):
    MANAGER_NAME = "RootZeroManager"

    def __init__(self, config_path):
        with open(config_path, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.manager_client = None

    def root_manager_init(self):
        self.manager_client = ZeroManagerClient(self.MANAGER_NAME)

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
                get_pretrained_model_name(model.name, False, get_model_by_name)

    def model_init(self):
        for protocol, engine_class in [
            ["chat", Chat_ENGINE_CLASS],
            ["retriever", Retriever_ENGINE_CLASS],
            ["reranker", Reranker_ENGINE_CLASS],
        ]:
            if protocol not in self.config:
                continue

            for model in self.config[protocol]["models"]:
                engine_kwargs = model.get("engine_kwargs", {})
                self.manager_client.start(name=model.name,
                                          engine_kwargs={"server_class": engine_class,
                                                         "engine_kwargs": engine_kwargs})

    def http_entrypoint_init(self):
        if "entrypoints" not in self.config:
            return

        if "ollama_compatible" in self.config.entrypoints:
            self.manager_client.start(name="ollama_compatible",
                                      engine_kwargs={
                                          "server_class": Entrypoint_ENGINE_CLASS,
                                          "engine_class": "zerollama.microservices.entrypoints.ollama_compatible.api:app",
                                          "engine_kwargs": {"port": 11434},
                                      })
        if "openai_compatible" in self.config.entrypoints:
            self.manager_client.start(name="openai_compatible",
                                      engine_kwargs={
                                          "server_class": Entrypoint_ENGINE_CLASS,
                                          "engine_class": "zerollama.microservices.entrypoints.openai_compatible.api:app",
                                          "engine_kwargs": {"port": 8080},
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
