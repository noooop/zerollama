

from zerollama.tasks.reranker.interface import Reranker


class Zhinao360Reranker(Reranker):
    family = "360Zhinao-Reranking "
    model_kwargs = {}
    header = ["name", "use_hf_only"]
    info = [
        # name
        ["qihoo360/360Zhinao-1.8B-Reranking", True]
    ]
    inference_backend = "zerollama.models.qihoo360.backend.reranker:Zhinao360Reranker"


if __name__ == '__main__':
    for model_name, *_ in Zhinao360Reranker.info:
        model = Zhinao360Reranker.get_model(model_name, local_files_only=False)
        model.load()

        sentence_pairs = [['what is panda?', 'hi'],
                          ['what is panda?',
                           'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]

        scores = model.compute_score(sentence_pairs).vecs["scores"].tolist()
        print(scores)