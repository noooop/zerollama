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
    model = get_model_by_name(model_name)
    model_config = model.get_model_config(model_name)
    model_info = model_config.info

    pretrained_model_name = get_pretrained_model_name(model_name, local_files_only, get_model_by_name)

    config = config_setup()

    if config.use_modelscope and not model_info.get("use_hf_only", False):
        pretrained_model_name_path = config.modelscope.cache_dir / pretrained_model_name.replace(".", "___")
        return pretrained_model_name_path
    else:
        return pretrained_model_name


def get_pretrained_model_name(model_name, local_files_only, get_model_by_name):
    config = config_setup()

    model = get_model_by_name(model_name)
    model_config = model.get_model_config(model_name)
    model_info = model_config.info

    if (config.use_modelscope and not model_info.get("use_hf_only", False) and
            not ("modelscope_name" in model_info and model_info["modelscope_name"] == "")):
        if "modelscope_name" in model_info:
            pretrained_model_name = model_info["modelscope_name"]
        else:
            pretrained_model_name = model_name

        import os
        os.environ["VLLM_USE_MODELSCOPE"] = "True"

        import modelscope
        if local_files_only:
            modelscope.snapshot_download = partial(modelscope.snapshot_download,
                                                   local_files_only=True)
            import transformers
            transformers.utils.hub.cached_file = partial(transformers.utils.hub.cached_file,
                                                         local_files_only=True)
        else:
            modelscope.snapshot_download(pretrained_model_name)
    else:
        if "hf_name" in model_info:
            pretrained_model_name = model_info["hf_name"]
        else:
            pretrained_model_name = model_name

        import huggingface_hub

        if local_files_only:
            huggingface_hub.snapshot_download = partial(huggingface_hub.snapshot_download,
                                                        local_files_only=True)

            import transformers
            transformers.utils.hub.cached_file = partial(transformers.utils.hub.cached_file,
                                                         local_files_only=True)
        else:
            huggingface_hub.snapshot_download(pretrained_model_name)

    return pretrained_model_name


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
