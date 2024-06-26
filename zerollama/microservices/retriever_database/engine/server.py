
from hashlib import md5
from gevent.lock import Semaphore
from gevent.threadpool import ThreadPoolExecutor
from zerollama.core.framework.zero.server import Z_MethodZeroServer
from zerollama.microservices.retriever_database.protocol import RetrieverDatabaseTopKRequest
from zerollama.microservices.retriever_database.protocol import ZeroServerResponseOk
from zerollama.microservices.retriever_database.collection import get_backend_by_name
from zerollama.core.config.main import config_setup


class ZeroRetrieverDatabaseEngine(Z_MethodZeroServer):
    def __init__(self, class_name, collection, **kwargs):
        self.class_name = class_name
        self.module_name = get_backend_by_name(class_name)

        if self.module_name is None:
            raise FileNotFoundError(f"RetrieverDatabase [{self.class_name}] not supported.")

        self.collection = collection

        print("Use Retriever Database backend:")
        print(f"{self.module_name}:{self.class_name}")

        import importlib
        module = importlib.import_module(self.module_name)
        self.db_class = getattr(module, class_name)

        self.db = None
        self.semaphore = Semaphore(self.db_class.n_concurrent)

        self.retriever_model = self.db_class.retriever_model
        self.pickle_name = md5(f"zerollama:{self.collection}:{self.retriever_model}".encode("utf-8")).hexdigest()

        kwargs.pop("name", None)
        Z_MethodZeroServer.__init__(self, name=self.pickle_name, protocol=self.db_class.protocol,
                                    port=None, do_register=True, **kwargs)

    def init(self):
        config = config_setup()
        pickle_file = f"{config.rag.path / self.collection / 'chunk' / 'chunk.pkl'}"

        self.db = self.db_class.load_from_file(pickle_file)
        print(f"{self.db.__class__.__name__}: is running!", "port:", self.port)

    def z_top_k(self, req):
        with self.semaphore:
            with ThreadPoolExecutor(1) as executor:
                f = executor.submit(self.worker, req)
                f.result()

    def worker(self, req):
        data = RetrieverDatabaseTopKRequest(**req.data)
        response = self.db.top_k(**data.dict())
        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)

    def z_info(self, req):
        rep = ZeroServerResponseOk(msg={
            "class_name": self.class_name,
            "collection": self.collection,
            "retriever_model": self.db.retriever_model})
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess
    from zerollama.microservices.retriever_database.protocol import ENGINE_CLASS

    collection = "test_collection"

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess(ENGINE_CLASS,
                               server_kwargs={
                                   "class_name": "BM25sRetrieverDatabase",
                                   "collection": collection,
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()



