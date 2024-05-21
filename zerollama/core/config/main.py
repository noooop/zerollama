
import os
import yaml
from pathlib import Path
from easydict import EasyDict as edict


def config_setup():
    home = Path.home()

    config_global_path = home / ".zerollama/config" / "global.yml"

    if config_global_path.exists():
        with open(config_global_path, 'r', encoding="utf-8") as f:
            config_global = yaml.safe_load(f)
    else:
        config_global = {}

    config = edict({})
    config.use_modelscope = False

    if "huggingface" in config_global:
        config_huggingface = config_global["huggingface"]

        if "HF_ENDPOINT" in config_huggingface:
            os.environ["HF_ENDPOINT"] = config_huggingface["HF_ENDPOINT"]

        if 'HF_HOME' in config_huggingface:
            os.environ['HF_HOME'] = config_huggingface["HF_HOME"]

    if "modelscope" in config_global:
        config_modelscope = config_global["modelscope"]
        if "USE_MODELSCOPE" in config_modelscope:
            if config_modelscope["USE_MODELSCOPE"]:
                config.use_modelscope = True

        if "MODELSCOPE_CACHE" in config_modelscope:
            os.environ["MODELSCOPE_CACHE"] = config_modelscope["MODELSCOPE_CACHE"]

    rag_path = Path.home() / ".zerollama/rag/documents"
    config.rag = edict({"path": rag_path})
    if "rag" in config_global:
        config_rag = config_global["rag"]
        if "path" in config_rag:
            config.rag.path = Path(config_rag["path"])

    return config


if __name__ == '__main__':
    from pprint import pprint

    pprint(config_setup())