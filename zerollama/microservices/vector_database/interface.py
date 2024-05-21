
import pickle
from zerollama.microservices.vector_database.protocol import PROTOCOL
from zerollama.microservices.vector_database.protocol import VectorDatabaseTopKResponse


class VectorDatabaseInterface(object):
    protocol = PROTOCOL
    n_concurrent = 1

    def __init__(self, embedding_model, nodes, embeddings, **kwargs):
        self.embedding_model = embedding_model
        self.embeddings = embeddings
        self.nodes = nodes
        self.meta = kwargs

        self.init()

    def init(self):
        """
        Load everything in memory.
        :return: None
        """
        raise NotImplementedError

    def top_k(self, query_dense_vecs, embedding_model, k=10) -> VectorDatabaseTopKResponse:
        raise NotImplementedError

    @classmethod
    def load_from_file(cls, filename):
        data = pickle.load(open(filename, "rb"))
        return cls(**data)



