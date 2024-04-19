
def pull(model_name):
    from zerollama.inference_backend.hf_transformers.download import download
    download(model_name)


def start(model_name):
    from zerollama.core.framework.zero_manager.client import ZeroManagerClient
    name = "ZeroInferenceManager"
    manager_client = ZeroManagerClient(name)
    model_class = "zerollama.models.qwen.qwen1_5:Qwen1_5"
    model_kwargs = {"model_name": model_name}
    manager_client.start(model_name, model_class, model_kwargs)


def terminate(model_name):
    from zerollama.core.framework.zero_manager.client import ZeroManagerClient
    name = "ZeroInferenceManager"
    manager_client = ZeroManagerClient(name)
    manager_client.terminate(model_name)


def main(argv):
    method = argv[1]
    model_name = argv[2]

    if method == "pull":
        pull(model_name)

    if method == "start":
        start(model_name)

    if method == "terminate":
        terminate(model_name)


if __name__ == '__main__':
    import sys
    main(sys.argv)
