

from zerollama.tasks.reranker.interface import Reranker


class BGEReranker(Reranker):
    family = "bge-reranker"
    model_kwargs = {}
    header = ["name", "modelscope_name", "type", "layerwise", "Base model", "Language", "feature"]
    info = [
        # name                                      modelscope_name                                     type          layerwise,   Base model               Language                    feature
        ["BAAI/bge-reranker-base",                  "AI-ModelScope/bge-reranker-base",                  "normal",     None,        "xlm-roberta-base",      "Chinese and English",      "Lightweight reranker model, easy to deploy, with fast inferenc"],
        ["BAAI/bge-reranker-large",                 "AI-ModelScope/bge-reranker-large",                 "normal",     None,        "xlm-roberta-large",     "Chinese and English",      "Lightweight reranker model, easy to deploy, with fast inference."],
        ["BAAI/bge-reranker-v2-m3",                 "AI-ModelScope/bge-reranker-v2-m3",                 "normal",     None,        "bge-m3",                "Multilingual",             "Lightweight reranker model, possesses strong multilingual capabilities, easy to deploy, with fast inference."],
        ["BAAI/bge-reranker-v2-gemma",              "AI-ModelScope/bge-reranker-v2-gemma",              "LLM-based",  None,        "gemma-2b",              "Multilingual",             "Suitable for multilingual contexts, performs well in both English proficiency and multilingual capabilities."],
        ["BAAI/bge-reranker-v2-minicpm-layerwise",  "AI-ModelScope/bge-reranker-v2-minicpm-layerwise",  "layerwise",  [8, 40],     "MiniCPM-2B-dpo-bf16",   "Multilingual",             "Suitable for multilingual contexts, performs well in both English and Chinese proficiency, allows freedom to select layers for output, facilitating accelerated inference."],
    ]
    inference_backend = "zerollama.models.baai.backend.reranker:BGEReranker"


if __name__ == '__main__':
    def get_model(model_name):
        model_kwargs = {"local_files_only": False}

        model_class = BGEReranker.inference_backend
        module_name, class_name = model_class.split(":")
        import importlib

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        model = model_class(model_name=model_name, **model_kwargs)
        model.load()
        return model


    model_name = "BAAI/bge-reranker-v2-m3"

    model = get_model(model_name)

    sentence_pairs = [['what is panda?', 'hi'],
                      ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]

    scores = model.compute_score(sentence_pairs,
                                 options={
                                     "batch_size": 12,
                                     "max_length": 512,
                                 }).vecs["scores"].tolist()
    print(scores)