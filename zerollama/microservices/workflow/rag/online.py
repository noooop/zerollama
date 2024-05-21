
import numpy as np
from zerollama.tasks.chat.engine.client import ChatClient
from zerollama.tasks.retriever.engine.client import RetrieverClient
from zerollama.tasks.reranker.engine.client import RerankerClient
from zerollama.microservices.vector_database.engine.client import VectorDatabaseClient

chat_client = ChatClient()
retriever_client = RetrieverClient()
reranker_client = RerankerClient()
vector_database_client = VectorDatabaseClient()

qa_prompt_tmpl_str = (
    "你是一个严谨的问答机器人，结合提供的资料回答问题, 如果发现资料无法得到答案，就回答不知道。 \n"
    "一步步的回答问题，输出的内容都从资料中直接引用，不要胡编乱造，回答尽可能全面清晰。 \n"
    "搜索的相关资料如下所示.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "问题: {query_str}\n"
    "答案: "
)


def rag(question, chat_model, retriever_model, reranker_model, collection,
        n_retriever_candidate=10, n_references=3, qa_prompt_tmpl_str=qa_prompt_tmpl_str,
        stream=False):
    # 1. embeddings
    embeddings = retriever_client.encode(retriever_model, [question]).vecs['dense_vecs'][0]

    # retriever
    top_k_nodes = vector_database_client.top_k(collection, retriever_model, embeddings, k=n_retriever_candidate).data

    # reranker
    sentence_pairs = [[question, c.node["text"]] for c in top_k_nodes]
    reranker_scores = reranker_client.compute_score(reranker_model, sentence_pairs).vecs["scores"]
    reranker_index = np.argsort(reranker_scores)
    rerank_nodes = [top_k_nodes[i] for i in reversed(reranker_index)]
    references = [n.node["text"] for n in rerank_nodes]

    # generation
    context_str = "\n".join(references[:n_references])
    prompt = qa_prompt_tmpl_str.format(context_str=context_str, query_str=question)

    messages = [
        {"role": "user", "content": prompt}
    ]

    if not stream:
        response = chat_client.chat(chat_model, messages)
        return response
    else:
        def generation():
            for msg in chat_client.stream_chat(chat_model, messages):
                yield msg
        return generation()


if __name__ == '__main__':
    from inspect import isgenerator

    response = rag(question="作者是谁？",
                   chat_model="Qwen/Qwen1.5-0.5B-Chat-AWQ",
                   retriever_model="BAAI/bge-m3",
                   reranker_model="BAAI/bge-reranker-v2-m3",
                   collection="test")

    print(response.msg.content)

    response = rag(question="作者是谁？",
                   chat_model="Qwen/Qwen1.5-0.5B-Chat-AWQ",
                   retriever_model="BAAI/bge-m3",
                   reranker_model="BAAI/bge-reranker-v2-m3",
                   collection="test",
                   stream=True)

    if isgenerator(response):
        for rep in response:
            if hasattr(rep.msg, "delta_content"):
                print(rep.msg.delta_content, end="")
            else:
                print("")



