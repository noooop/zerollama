
from gevent.lock import Semaphore
from gevent.threadpool import ThreadPoolExecutor
from zerollama.core.framework.zero.server import Z_MethodZeroServer
from zerollama.tasks.chat.protocol import ZeroServerResponseOk


class ZeroInferenceEngine(Z_MethodZeroServer):
    get_model_by_name = None

    def __init__(self, name, engine_kwargs=None, **kwargs):
        engine_kwargs = engine_kwargs or {}
        self.model_name = name
        self.model_class = self.get_model_by_name(name)

        if self.model_class is None:
            raise ValueError(f"[{self.model_name}] not support.")

        self.inference_backend = engine_kwargs.pop("inference_backend", None) or self.model_class.inference_backend

        print("use inference backend:")
        print(self.inference_backend)

        if isinstance(self.inference_backend, str):
            module_name, class_name = self.inference_backend.split(":")
            import importlib
            module = importlib.import_module(module_name)
            self.inference_backend = getattr(module, class_name)

        self.inference = self.inference_backend(model_name=self.model_name, **engine_kwargs)

        self.semaphore = Semaphore(self.inference.n_concurrent)

        Z_MethodZeroServer.__init__(self, name=self.model_name, protocol=self.inference.protocol,
                                    port=None, do_register=True, **kwargs)

    def init(self):
        self.inference.load()
        print(f"{self.__class__.__name__}: ", self.name, "is running!", "port:", self.port)

    def z_inference(self, req):
        if self.semaphore == 1:
            with self.semaphore:
                with ThreadPoolExecutor(1) as executor:
                    f = executor.submit(self.inference_worker, req)
                    f.result()
        else:
            self.inference_worker(req)

    def inference_worker(self, req):
        pass

    def z_info(self, req):
        if hasattr(self.inference, "info"):
            info = self.inference.info
        else:
            info = {}

        rep = ZeroServerResponseOk(msg=info)
        self.zero_send(req, rep)

