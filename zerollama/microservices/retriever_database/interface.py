
import pickle
from zerollama.microservices.retriever_database.protocol import PROTOCOL
from zerollama.microservices.retriever_database.protocol import RetrieverDatabaseTopKResponse


class RetrieverDatabaseInterface(object):
    retriever_model = ""
    protocol = PROTOCOL
    n_concurrent = 1

    def __init__(self, collection, nodes, **kwargs):
        self.collection = collection
        self.nodes = nodes
        self.meta = kwargs

        self.init()

    def init(self):
        """
        Load everything in memory.
        :return: None
        """
        raise NotImplementedError

    def top_k(self, query, k=10) -> RetrieverDatabaseTopKResponse:
        raise NotImplementedError

    @classmethod
    def load_from_file(cls, filename):
        data = pickle.load(open(filename, "rb"))
        return cls(**data)