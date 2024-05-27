
from zerollama.core.config.main import config_setup


def download(model_name, model_class=None):
    config = config_setup()

    if model_class is not None:
        model_info = model_class.get_model_config(model_name).info
        if config.use_modelscope:
            if "modelscope_name" in model_info:
                model_name = model_info["modelscope_name"]
        else:
            if "hf_name" in model_info:
                model_name = model_info["hf_name"]

    if config.use_modelscope:
        from modelscope import snapshot_download

        snapshot_download(
            model_id=model_name
        )

    else:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_name,
            local_dir_use_symlinks=False,
        )


if __name__ == '__main__':
    download("Qwen/Qwen1.5-1.8B-Chat")