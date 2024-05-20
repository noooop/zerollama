

def download(model_name, get_model_by_name):
    model_class = get_model_by_name(model_name)
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
    for size in ["0.5B"]:
        for k in ["q8_0"]:
            repo_id = f"Qwen/Qwen1.5-{size}-Chat-GGUF"
            filename = f"*{k}.gguf"
            print(repo_id)
            print(filename)
            download(model_name=f"{repo_id}+{filename}")

    from zerollama.tasks.chat.collection import get_model_by_name
    download("Qwen/Qwen1.5-1.8B-Chat", get_model_by_name)


