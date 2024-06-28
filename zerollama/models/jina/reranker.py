

from zerollama.tasks.reranker.interface import Reranker


class JinaReranker(Reranker):
    family = "jina-reranker-v2"
    model_kwargs = {}
    header = ["name", "use_hf_only"]
    info = [
        ['jinaai/jina-reranker-v2-base-multilingual', True]
    ]
    inference_backend = "zerollama.models.jina.backend.reranker:JinaReranker"


if __name__ == '__main__':
    model_name = "BAAI/bge-reranker-v2-m3"

    for model_name, *_ in JinaReranker.info:
        model = JinaReranker.get_model(model_name, local_files_only=False)
        model.load()

        sentence_pairs = [['what is panda?', 'hi'],
                          ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]

        scores = model.compute_score(sentence_pairs).vecs["scores"].tolist()
        print(scores)