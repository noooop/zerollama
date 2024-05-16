

from zerollama.tasks.reranker.collection import get_model_by_name


def download(model_name):
    model_class = get_model_by_name(model_name)

    if model_class is None:
        raise ValueError(f"[{model_name}] not supported.")

    download_backend = model_class.download_backend

    print("use download backend:")
    print(download_backend)

    if isinstance(download_backend, str):
        module_name, function_name = download_backend.split(":")
        import importlib
        module = importlib.import_module(module_name)
        download_backend = getattr(module, function_name)

    download_backend(model_name)


if __name__ == '__main__':
    download("bce-reranker-base_v1")


