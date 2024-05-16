
from zerollama.models.bce.retriever import BCERetriever
from zerollama.models.bce.reranker import BCEReranker

MAP = {}

for name, hf_name, modelscope_name in BCERetriever.info:
    MAP[name] = {"hf_name": hf_name, "modelscope_name": modelscope_name}

for name, hf_name, modelscope_name in BCEReranker.info:
    MAP[name] = {"hf_name": hf_name, "modelscope_name": modelscope_name}


def download(model_name):
    if model_name not in MAP:
        raise ValueError(f"model {model_name} not support")

    from zerollama.core.config.main import config_setup

    config_setup()

    from modelscope import snapshot_download

    print(MAP[model_name]["modelscope_name"])

    snapshot_download(
        model_id=MAP[model_name]["modelscope_name"]
    )


if __name__ == '__main__':
    download("bce-embedding-base_v1")
