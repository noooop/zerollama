from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.microservices.workflow.rag.protocol import PROTOCOL
from zerollama.microservices.workflow.rag.protocol import RAGRequest, RAGResponse
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponse, ChatCompletionStreamResponseDone

CLIENT_VALIDATION = True


class RAGClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def rag(self, question, chat_model, retriever_model, reranker_model, collection,
            n_retriever_candidate=10, n_references=3, qa_prompt_tmpl_str=None,
            stream=False, return_references=False):

        name = "rag"
        method = "rag"
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

        if CLIENT_VALIDATION:
            data = RAGRequest(**data).dict()

        if not stream:
            rep = self.query(name, method, data)
            if rep is None:
                raise RuntimeError("RAG server not found.")

            if rep.state != "ok":
                raise RuntimeError(f"RAG error, with error msg [{rep.msg}]")

            return RAGResponse(**rep.msg)
        else:
            def generation():
                for rep in self.stream_query(name, method, data):
                    if rep is None:
                        raise RuntimeError(f"RAG [{name}] server not found.")

                    if rep.state != "ok":
                        raise RuntimeError(f"RAG [{name}] error, with error msg [{rep.msg}]")

                    if "references" in rep.msg:
                        yield RAGResponse(**rep.msg)
                    elif rep.msg["finish_reason"] is None:
                        yield ChatCompletionStreamResponse(**rep.msg)
                    else:
                        yield ChatCompletionStreamResponseDone(**rep.msg)

            return generation()

    def default_qa_prompt_tmpl(self):
        name = "rag"
        method = "default_qa_prompt_tmpl"
        client = self.get_client(name)
        if client is None:
            return None

        return self.query(name, method)


if __name__ == '__main__':
    from inspect import isgenerator
    from pprint import pprint

    client = RAGClient()

    print(client.default_qa_prompt_tmpl().msg)

    response = client.rag(question="作者是谁？",
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

    response = client.rag(question="作者是谁？",
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
