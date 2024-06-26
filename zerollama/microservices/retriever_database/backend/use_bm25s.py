
from zerollama.microservices.retriever_database.interface import RetrieverDatabaseInterface
from zerollama.microservices.retriever_database.protocol import TopKNode, RetrieverDatabaseTopKResponse


class BM25sRetrieverDatabase(RetrieverDatabaseInterface):
    retriever_model = "BM25s"

    def init(self):
        import bm25s
        from zerollama.core.config.main import config_setup

        config_setup()

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        sentences = [node["text"] for node in self.nodes]

        corpus_tokens = self.tokenizer(sentences)["input_ids"]

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        self.retriever = retriever

    def top_k(self, query, k=10, retriever_model=None):
        if retriever_model is not None and retriever_model != self.retriever_model:
            raise ValueError(f"[{retriever_model}] not support")

        query_tokens = self.tokenizer.encode(query)
        results, scores = self.retriever.retrieve([query_tokens], k=k)

        data = [TopKNode(index=i, score=s, node=self.nodes[i]) for s, i in zip(scores[0], results[0])]
        return RetrieverDatabaseTopKResponse(retriever_model=self.retriever_model, data=data)


if __name__ == '__main__':
    from zerollama.core.config.main import config_setup

    config = config_setup()

    collection = "test_collection"
    pickle_file = f"{config.rag.path / collection / 'chunk' / 'chunk.pkl'}"

    rdb = BM25sRetrieverDatabase.load_from_file(pickle_file)

    query = rdb.nodes[10]["text"]

    top_k = rdb.top_k(query, k=10)

    for n in top_k.data:
        print(n.score)
        print(n.index)
        print(n.node["text"])