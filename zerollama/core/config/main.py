import yaml
import os
from pathlib import Path


def config_setup():
    home = Path.home()

    config_global_path = home / ".zerollama/config" / "global.yml"

    if config_global_path.exists():
        with open(config_global_path, 'r') as f:
            config_global = yaml.safe_load(f)
    else:
        config_global = {}

    if "huggingface" in config_global:
        config_huggingface = config_global["huggingface"]

        if "HF_ENDPOINT" in config_huggingface:
            os.environ["HF_ENDPOINT"] = config_huggingface["HF_ENDPOINT"]

        if 'HF_HOME' in config_huggingface:
            os.environ['HF_HOME'] = config_huggingface["HF_HOME"]

    return config_global


if __name__ == '__main__':
    from pprint import pprint

    pprint(config_setup())