from functools import partial
from zerollama.core.config.main import config_setup


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

    download_backend(model_name=model_name, model_class=model_class)


def get_pretrained_model_name_or_path(model_name, local_files_only, get_model_by_name, get_model_config_by_name=None):
    config = config_setup()

    model = get_model_by_name(model_name)
    model_config = model.get_model_config(model_name)
    model_info = model_config.info

    if local_files_only:
        import huggingface_hub
        huggingface_hub.snapshot_download = partial(huggingface_hub.snapshot_download,
                                                    local_files_only=True)
    else:
        download(model_name, get_model_by_name)

    pretrained_model_name_or_path = model_name
    if config.use_modelscope:
        if "modelscope_name" in model_info:
            pretrained_model_name_or_path = config.modelscope.cache_dir / model_info["modelscope_name"]
    else:
        if "hf_name" in model_info:
            pretrained_model_name_or_path = model_info["hf_name"]

    return pretrained_model_name_or_path


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


