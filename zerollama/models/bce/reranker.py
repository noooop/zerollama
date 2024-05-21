

from zerollama.tasks.reranker.interface import Reranker


class BCEReranker(Reranker):
    family = "bce-reranker"
    model_kwargs = {}
    header = ["name", "hf_name", "modelscope_name"]
    info = [
        # name
        ["bce-reranker-base_v1", "maidalun1020/bce-reranker-base_v1", "maidalun/bce-reranker-base_v1"],

    ]
    inference_backend = "zerollama.models.bce.backend.reranker:BCEReranker"
    download_backend = "zerollama.models.bce.backend.download:download"


if __name__ == '__main__':
    def get_model(model_name):
        model_kwargs = {}

        model_class = BCEReranker.inference_backend
        module_name, class_name = model_class.split(":")
        import importlib

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        model = model_class(model_name=model_name, **model_kwargs)
        model.load()
        return model


    model_name = "bce-reranker-base_v1"

    model = get_model(model_name)

    sentence_pairs = [['what is panda?', 'hi'],
                      ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]

    scores = model.compute_score(sentence_pairs).vecs["scores"].tolist()
    print(scores)