
import os
import yaml
from pathlib import Path
from easydict import EasyDict as edict


def get_modelscope_cache_dir():
    import os
    default_cache_dir = Path.home().joinpath('.cache', 'modelscope')
    base_path = os.getenv('MODELSCOPE_CACHE', os.path.join(default_cache_dir, 'hub'))
    return base_path


def config_setup():
    home = Path.home()

    config_global_path = home / ".zerollama/config" / "global.yml"

    if config_global_path.exists():
        with open(config_global_path, 'r', encoding="utf-8") as f:
            config_global = yaml.safe_load(f)
    else:
        config_global = {}

    config = edict({})
    config.use_modelscope = True

    if "huggingface" in config_global:
        config_huggingface = config_global["huggingface"]

        if "HF_ENDPOINT" in config_huggingface:
            os.environ["HF_ENDPOINT"] = config_huggingface["HF_ENDPOINT"]

        if 'HF_HOME' in config_huggingface:
            os.environ['HF_HOME'] = config_huggingface["HF_HOME"]

    if "modelscope" in config_global:
        config_modelscope = config_global["modelscope"]
        if "USE_MODELSCOPE" in config_modelscope:
            config.use_modelscope = config_modelscope["USE_MODELSCOPE"]

        if "MODELSCOPE_CACHE" in config_modelscope:
            os.environ["MODELSCOPE_CACHE"] = config_modelscope["MODELSCOPE_CACHE"]

    config.modelscope = {}
    config.modelscope.cache_dir = Path(get_modelscope_cache_dir())

    if config.use_modelscope:
        os.environ["VLLM_USE_MODELSCOPE"] = "True"

    rag_path = Path.home() / ".zerollama/rag/documents"
    config.rag = edict({"path": rag_path})
    if "rag" in config_global:
        config_rag = config_global["rag"]
        if "path" in config_rag:
            config.rag.path = Path(config_rag["path"])

    if "vllm" in config_global:
        config_vllm = config_global["vllm"]
        if "VLLM_DO_NOT_TRACK" in config_vllm:
            os.environ["VLLM_DO_NOT_TRACK"] = str(config_vllm["VLLM_DO_NOT_TRACK"])

        if "DO_NOT_TRACK" in config_vllm:
            os.environ["DO_NOT_TRACK"] = str(config_vllm["DO_NOT_TRACK"])

        if "VLLM_NO_USAGE_STATS" in config_vllm:
            os.environ["VLLM_NO_USAGE_STATS"] = str(config_vllm["VLLM_NO_USAGE_STATS"])

    if "cuda" in config_global:
        config_cuda = config_global["cuda"]
        if "cudnn_path" in config_cuda:
            import platform
            plat = platform.system().lower()
            if plat == 'windows':
                os.environ["PATH"] = os.environ["PATH"] + ";" + config_cuda["cudnn_path"]

    if "agents" in config_global:
        config_agents = config_global["agents"]
        config.llm_config = config_agents.get("llm_config", {})
    return config


if __name__ == '__main__':
    from pprint import pprint

    pprint(config_setup())