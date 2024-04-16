
def run_model_server():
    from zerollama.core.framework.inference_engine.server import ZeroInferenceEngine
    from zerollama.models.qwen.qwen1_5 import Qwen1_5

    engine = ZeroInferenceEngine(model_class=Qwen1_5,
                                 model_kwargs={
                                     "model_name": "Qwen/Qwen1.5-0.5B-Chat"
                                 })
    print("ZeroInferenceEngine: ", engine.name, "is running!")
    engine.run()


def run_gateway():
    import subprocess
    h = subprocess.Popen("python -m applications.chat_server.http_gateway", shell=True)
    print("gateway running!")
    h.wait()


def run_nameserver():
    from zerollama.core.framework.nameserver.server import nameserver
    nameserver()


if __name__ == '__main__':
    from multiprocess import Process

    h1 = Process(target=run_model_server)
    h2 = Process(target=run_gateway)
    h3 = Process(target=run_nameserver)

    handle = [h1, h2, h3]
    for h in handle:
        h.start()

    for h in handle:
        h.join()
