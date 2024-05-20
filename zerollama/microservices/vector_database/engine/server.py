
from hashlib import md5
from gevent.lock import Semaphore
from gevent.threadpool import ThreadPoolExecutor
from zerollama.core.framework.zero.server import Z_MethodZeroServer
from zerollama.microservices.vector_database.protocol import VectorDatabaseTopKRequest
from zerollama.microservices.vector_database.protocol import ZeroServerResponseOk
from zerollama.core.config.main import config_setup


class ZeroVectorDatabaseEngine(Z_MethodZeroServer):
    def __init__(self, vdb_class, filename, embedding_model, **kwargs):
        self.vdb_class = vdb_class
        self.filename = filename
        self.embedding_model = embedding_model
        self.pickle_name = f"zerollama:{filename}:{embedding_model}:embeddings"
        self.hash = md5(self.pickle_name.encode("utf-8")).hexdigest()

        print("Use Vector Database backend:")
        print(self.vdb_class)

        if isinstance(self.vdb_class, str):
            module_name, class_name = self.vdb_class.split(":")
            import importlib
            module = importlib.import_module(module_name)
            self.vdb_class = getattr(module, class_name)

        self.vdb = None
        self.semaphore = Semaphore(self.vdb_class.n_concurrent)

        Z_MethodZeroServer.__init__(self, name=self.hash, protocol=self.vdb_class.protocol,
                                    port=None, do_register=True, **kwargs)

    def init(self):
        config = config_setup()

        file = list((config.rag.path / self.filename).glob("*.txt"))[0]
        #book_name = file.stem.split("-")[0]

        self.vdb = self.vdb_class.load_from_file(f"{file.parent / (self.hash + '.pkl')}")
        print(f"{self.vdb.__class__.__name__}: is running!", "port:", self.port)

    def z_top_k(self, req):
        with self.semaphore:
            with ThreadPoolExecutor(1) as executor:
                f = executor.submit(self.worker, req)
                f.result()

    def worker(self, req):
        data = VectorDatabaseTopKRequest(**req.data)
        response = self.vdb.top_k(data)
        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)

    def z_info(self, req):
        rep = ZeroServerResponseOk(msg={"vector database class": self.vdb.__class__.__name__})
        self.zero_send(req, rep)


if __name__ == '__main__':
    from zerollama.core.framework.zero.server import ZeroServerProcess

    filename = "test"
    embedding_model = "BAAI/bge-m3"

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess("zerollama.microservices.vector_database.engine.server:ZeroVectorDatabaseEngine",
                               server_kwargs={
                                   "vdb_class": "zerollama.microservices.vector_database.use_bruteforce:BruteForceVectorDatabase",
                                   "filename": filename,
                                   "embedding_model": embedding_model
                               })

    nameserver.start()
    engine.start()

    engine.wait()
    nameserver.wait()


