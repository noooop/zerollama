
def run_gateway():
    import subprocess
    h = subprocess.Popen("python -m applications.chat_webui.http_gateway", shell=True)
    print("gateway running!")
    h.wait()


if __name__ == '__main__':
    from multiprocess import Process
    from zerollama.core.framework.zero.server import ZeroServerProcess

    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")
    engine = ZeroServerProcess("zerollama.core.framework.inference_engine.server:ZeroInferenceEngine",
                               server_kwargs={
                                   "model_class": "zerollama.models.qwen.qwen1_5:Qwen1_5",
                                   "model_kwargs": {
                                     "model_name": "Qwen/Qwen1.5-0.5B-Chat"
                                   }
                               },
                               ignore_warnings=True)
    h3 = Process(target=run_gateway)

    handle = [nameserver, engine, h3]
    for h in handle:
        h.start()

    try:
        for h in handle:
            h.join()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        engine.terminate()
        nameserver.terminate()
        print("quit gracefully")
