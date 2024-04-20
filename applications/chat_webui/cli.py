
def list_model():
    from prettytable import PrettyTable

    from zerollama.models.qwen.qwen1_5 import info_header, info
    table = PrettyTable(info_header, align='l')

    for x in info:
        table.add_row(x)

    print(table)


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

    if method == "list":
        list_model()
        return

    model_name = argv[2]

    if method == "pull":
        pull(model_name)
        return

    if method == "start":
        start(model_name)
        return

    if method == "terminate":
        terminate(model_name)
        return


if __name__ == '__main__':
    import sys
    main(sys.argv)
