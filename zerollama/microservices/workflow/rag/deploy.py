
import yaml
from easydict import EasyDict as edict
from zerollama.core.framework.zero_manager.client import ZeroManagerClient
from zerollama.tasks.chat.protocol import Chat_MANAGER_NAME, Chat_ENGINE_CLASS
from zerollama.tasks.retriever.protocol import Retriever_MANAGER_NAME, Retriever_ENGINE_CLASS
from zerollama.tasks.reranker.protocol import Reranker_MANAGER_NAME, Reranker_ENGINE_CLASS
from zerollama.microservices.vector_database.protocol import VectorDatabase_PROTOCOL, VectorDatabase_MANAGER_NAME, VectorDatabase_ENGINE_CLASS
from zerollama.microservices.vector_database.engine.client import VectorDatabaseClient
from zerollama.microservices.workflow.rag.offline import text2vec


class Deploy(object):
    Root_MANAGER_NAME = "RootZeroManager"

    def __init__(self, config_path):
        with open(config_path, 'r', encoding="utf-8") as f:
            self.config = edict(yaml.safe_load(f))

        self.root_manager_client = None
        self.chat_manager_client = None
        self.retriever_manager_client = None

    def root_manager_init(self):
        self.root_manager_client = ZeroManagerClient(self.Root_MANAGER_NAME)

        #print("=" * 80)
        #print(self.root_manager_client.get_service_names())
        #print(f'{self.Root_MANAGER_NAME} support_methods')
        #print(self.root_manager_client.support_methods(self.Root_MANAGER_NAME))

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
            self._manage_init(protocol, manage_name, engine_class)

            for name in self.config[protocol]["models"]:
                client = getattr(self, f"{protocol}_manager_client")
                client.start(name)

    def _processing_document(self, collection):
        for embedding_model in self.config["retriever"]["models"]:
            text2vec(collection=collection, embedding_model=embedding_model)

    def processing_document(self):
        print("=" * 80)
        print(f'processing document')
        for filename in self.config["documents"]["collections"]:
            self._processing_document(filename)

    def vector_database_init(self):
        print("=" * 80)
        print(f'vector database init')

        protocol, manage_name, engine_class = VectorDatabase_PROTOCOL, VectorDatabase_MANAGER_NAME, VectorDatabase_ENGINE_CLASS
        self._manage_init(protocol, manage_name, engine_class)

        class_name = self.config.vector_database.backend

        for collection in self.config["documents"]["collections"]:
            for embedding_model in self.config["retriever"]["models"]:
                db_name = VectorDatabaseClient.get_db_name(collection, embedding_model)
                client = getattr(self, f"{protocol}_manager_client")
                client.start(name=db_name,
                             engine_kwargs={
                                 "class_name": class_name,
                                 "collection": collection,
                                 "embedding_model": embedding_model
                            })

    def __call__(self):
        self.root_manager_init()
        self.model_init()
        self.processing_document()
        self.vector_database_init()


if __name__ == '__main__':
    import os
    from pathlib import Path
    path = Path(os.path.dirname(__file__))

    config_path = path / "deploy.yml"
    Deploy(config_path)()
