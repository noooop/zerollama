from gevent import spawn, joinall
from zerollama.tasks.retriever.engine.client import RetrieverClient
from zerollama.microservices.vector_database.engine.client import VectorDatabaseClient
from zerollama.microservices.retriever_database.engine.client import RetrieverDatabaseClient

retriever_client = RetrieverClient()
vector_database_client = VectorDatabaseClient()
retriever_database_client = RetrieverDatabaseClient()


def vector_database_worker(question, retriever_model, collection, n_retriever_candidate):
    # 1. embeddings
    embeddings = retriever_client.encode(retriever_model, [question]).vecs['dense_vecs'][0]

    # 2. retriever
    top_k_nodes = vector_database_client.top_k(collection, retriever_model, embeddings, k=n_retriever_candidate).data
    return top_k_nodes


def retriever_database_worker(question, retriever_model, collection, n_retriever_candidate):
    top_k_nodes = retriever_database_client.top_k(collection, retriever_model, question, k=n_retriever_candidate).data
    return top_k_nodes


def multi_recall(question, collection, n_retriever_candidate, retriever_models, retriever_database_models):
    jobs = ([spawn(vector_database_worker, question, retriever_model, collection, n_retriever_candidate)
             for retriever_model in retriever_models] +
            [spawn(retriever_database_worker, question, retriever_model, collection, n_retriever_candidate)
             for retriever_model in retriever_database_models])

    joinall(jobs, timeout=2)

    index = set()
    recall_nodes = []
    for job in jobs:
        for n in job.value:
            if n.index not in index:
                recall_nodes.append(n.node)
                index.add(n.index)

    return recall_nodes


if __name__ == '__main__':
    recall_nodes = multi_recall(question="作者是谁？",
                                collection="test_collection",
                                n_retriever_candidate=10,
                                retriever_models=["BAAI/bge-m3"],
                                retriever_database_models=["BM25s"])
    for i, node in enumerate(recall_nodes):
        print(i)
        print(node["text"])