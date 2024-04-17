
from zerollama.core.config.main import config_setup


def download(model_name):
    config_setup()
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_name,
        local_dir_use_symlinks=False,
    )


if __name__ == '__main__':
    download("Qwen/Qwen1.5-1.8B-Chat")