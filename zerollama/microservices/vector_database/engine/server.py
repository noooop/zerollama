
from hashlib import md5
from gevent.lock import Semaphore
from gevent.threadpool import ThreadPoolExecutor
from zerollama.core.framework.zero.server import Z_MethodZeroServer
from zerollama.microservices.vector_database.protocol import VectorDatabaseTopKRequest
from zerollama.microservices.vector_database.protocol import ZeroServerResponseOk
from zerollama.microservices.vector_database.collection import get_backend_by_name
from zerollama.core.config.main import config_setup


class ZeroVectorDatabaseEngine(Z_MethodZeroServer):
    def __init__(self, class_name, collection, embedding_model, **kwargs):
        self.class_name = class_name
        self.module_name = get_backend_by_name(class_name)

        if self.module_name is None:
            raise FileNotFoundError(f"VectorDatabase [{self.class_name}] not supported.")

        self.collection = collection
        self.embedding_model = embedding_model
        self.pickle_name = md5(f"zerollama:{self.collection}:{self.embedding_model}:embeddings".encode("utf-8")).hexdigest()

        print("Use Vector Database backend:")
        print(f"{self.module_name}:{self.class_name}")

        import importlib
        module = importlib.import_module(self.module_name)
        self.vdb_class = getattr(module, class_name)

        self.vdb = None
        self.semaphore = Semaphore(self.vdb_class.n_concurrent)

        kwargs.pop("name", None)
        Z_MethodZeroServer.__init__(self, name=self.pickle_name, protocol=self.vdb_class.protocol,
                                    port=None, do_register=True, **kwargs)

    def init(self):
        config = config_setup()

        pickle_file = f"{config.rag.path / self.collection / 'embeddings' / (self.pickle_name + '.pkl')}"

        self.vdb = self.vdb_class.load_from_file(pickle_file)
        print(f"{self.vdb.__class__.__name__}: is running!", "port:", self.port)

    def z_top_k(self, req):
        with self.semaphore:
            with ThreadPoolExecutor(1) as executor:
                f = executor.submit(self.worker, req)
                f.result()

    def worker(self, req):
        data = VectorDatabaseTopKRequest(**req.data)
        response = self.vdb.top_k(**data.dict())
        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)

    def z_info(self, req):
        rep = ZeroServerResponseOk(msg={
            "class_name": self.class_name,
            "collection": self.collection,
            "embedding_model": self.embedding_model})
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess

    collection = "test"
    embedding_model = "BAAI/bge-m3"

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess("zerollama.microservices.vector_database.engine.server:ZeroVectorDatabaseEngine",
                               server_kwargs={
                                   "class_name": "BruteForceVectorDatabase",
                                   "collection": collection,
                                   "embedding_model": embedding_model
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()



