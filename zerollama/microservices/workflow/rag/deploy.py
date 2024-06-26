from zerollama.microservices.vector_database.protocol import VectorDatabase_ENGINE_CLASS
from zerollama.microservices.vector_database.engine.client import VectorDatabaseClient
from zerollama.microservices.workflow.rag.protocol import RAG_ENGINE_CLASS
from zerollama.microservices.workflow.rag.offline import text2vec
from zerollama.microservices.entrypoints.protocol import Entrypoint_ENGINE_CLASS
from zerollama.microservices.standalone.deploy import Deploy as DeployBase


class Deploy(DeployBase):
    def _processing_document(self, collection):
        for embedding_model in self.config["retriever"]["models"]:
            text2vec(collection=collection, embedding_model=embedding_model.name)

    def processing_document(self):
        print("=" * 80)
        print(f'processing document')
        for filename in self.config["documents"]["collections"]:
            self._processing_document(filename)

    def vector_database_init(self):
        class_name = self.config.vector_database.backend

        for collection in self.config["documents"]["collections"]:
            for embedding_model in self.config["retriever"]["models"]:
                db_name = VectorDatabaseClient.get_db_name(collection, embedding_model.name)
                self.manager_client.start(name=db_name,
                                          engine_kwargs={"server_class": VectorDatabase_ENGINE_CLASS,
                                                         "class_name": class_name,
                                                         "collection": collection,
                                                         "embedding_model": embedding_model.name
                                                         })

    def rag_server_init(self):
        self.manager_client.start(name="rag",
                                  engine_kwargs={"server_class": RAG_ENGINE_CLASS})

    def http_entrypoint_init(self):
        super().http_entrypoint_init()

        self.manager_client.start(name="rag_http_entrypoint",
                                  engine_kwargs={
                                      "server_class": Entrypoint_ENGINE_CLASS,
                                      "engine_class": "zerollama.microservices.workflow.rag.http_entrypoint.api:app",
                                      "engine_kwargs": {"port": 8000},
                                  })

    def __call__(self):
        self.verify()
        self.pull_models()
        self.root_manager_init()
        self.model_init()

        self.processing_document()
        self.vector_database_init()
        self.rag_server_init()
        self.http_entrypoint_init()


if __name__ == '__main__':
    import os
    from pathlib import Path

    path = Path(os.path.dirname(__file__))

    config_path = path / "deploy.yml"
    Deploy(config_path)()
