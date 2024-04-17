
def run_gateway():
    import subprocess
    h = subprocess.Popen("python -m applications.chat_server.http_gateway", shell=True)
    print("gateway running!")
    h.wait()


if __name__ == '__main__':
    from multiprocess import Process
    from zerollama.core.framework.zeroserver.server import ZeroServerProcess, Event

    h1 = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer", event=Event())
    h2 = ZeroServerProcess("zerollama.core.framework.inference_engine.server:ZeroInferenceEngine",
                               server_kwargs={
                                   "model_class": "zerollama.models.qwen.qwen1_5:Qwen1_5",
                                   "model_kwargs": {
                                     "model_name": "Qwen/Qwen1.5-0.5B-Chat"
                                   }
                               },
                               event=Event())
    h3 = Process(target=run_gateway)

    handle = [h1, h2, h3]
    for h in handle:
        h.start()

    for h in handle:
        h.join()
