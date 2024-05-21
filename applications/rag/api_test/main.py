import json
import requests
from zerollama.microservices.workflow.rag.protocol import RAGResponse
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponse, ChatCompletionStreamResponseDone

base_url = "http://localhost:8000"


def rag(question, chat_model, retriever_model, reranker_model, collection,
        n_retriever_candidate=10, n_references=3, qa_prompt_tmpl_str=None,
        stream=False, return_references=False):
    data = {"question": question,
            "chat_model": chat_model,
            "retriever_model": retriever_model,
            "reranker_model": reranker_model,
            "collection": collection,
            "n_retriever_candidate": n_retriever_candidate,
            "n_references": n_references,
            "qa_prompt_tmpl_str": qa_prompt_tmpl_str,
            "stream": stream,
            "return_references": return_references}

    if stream:
        def generate():
            r = requests.post(base_url + "/api/rag",
                              json=data,
                              stream=True)
            for chunk in r.iter_lines():
                msg = json.loads(chunk)
                if "references" in msg:
                    yield RAGResponse(**msg)
                elif msg["finish_reason"] is None:
                    yield ChatCompletionStreamResponse(**msg)
                else:
                    yield ChatCompletionStreamResponseDone(**msg)

        return generate()
    else:
        r = requests.post(base_url + "/api/rag",
                          json=data)
        return RAGResponse(**r.json())


if __name__ == '__main__':
    from inspect import isgenerator
    from pprint import pprint

    response = rag(question="作者是谁？",
                   chat_model="Qwen/Qwen1.5-0.5B-Chat-AWQ",
                   retriever_model="BAAI/bge-m3",
                   reranker_model="BAAI/bge-reranker-v2-m3",
                   collection="test_collection",
                   stream=False,
                   return_references=True)
    pprint(response.answer.dict())

    for i, r in enumerate(response.references):
        print(i)
        print(r)

    print("=" * 80)
    print("stream=True")

    response = rag(question="作者是谁？",
                   chat_model="Qwen/Qwen1.5-0.5B-Chat-AWQ",
                   retriever_model="BAAI/bge-m3",
                   reranker_model="BAAI/bge-reranker-v2-m3",
                   collection="test_collection",
                   stream=True,
                   return_references=True)

    references = None
    if isgenerator(response):
        for rep in response:
            if isinstance(rep, RAGResponse):
                references = rep.references
            else:
                if hasattr(rep, "delta_content"):
                    print(rep.delta_content, end="")
                else:
                    print("")