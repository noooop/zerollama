import json
import datetime
from fastapi import FastAPI
from inspect import isgenerator
from fastapi.responses import StreamingResponse, JSONResponse
from pprint import pprint
from zerollama.tasks.chat.engine.client import ChatClient
from zerollama.tasks.retriever.engine.client import RetrieverClient
from zerollama.tasks.reranker.engine.client import RerankerClient
from zerollama.microservices.vector_database.engine.client import VectorDatabaseClient
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponseDone
from zerollama.microservices.entrypoints.ollama_compatible.protocol import ChatCompletionRequest, EmbeddingsRequest
from zerollama.microservices.workflow.rag.engine.client import RAGClient
from zerollama.microservices.workflow.rag.protocol import RAGRequest, RAGResponse

chat_client = ChatClient()
retriever_client = RetrieverClient()
reranker_client = RerankerClient()
vector_database_client = VectorDatabaseClient()
rag_client = RAGClient()

app = FastAPI()


def get_timestamp():
    return datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')


@app.get("/")
def health():
    return "zerollama is running"


@app.get("/api/chat_models")
def chat_models():
    response = chat_client.get_service_names()
    services = response.msg["service_names"]

    models = []
    for s in services:
        models.append({
            'name': s,
            'model': s,
        })

    return {"models": models}


@app.get("/api/retriever_models")
def retriever_models():
    response = retriever_client.get_service_names()
    services = response.msg["service_names"]

    models = []
    for s in services:
        models.append({
            'name': s,
            'model': s,
        })

    return {"models": models}


@app.get("/api/reranker_models")
def reranker_models():
    response = reranker_client.get_service_names()
    services = response.msg["service_names"]

    models = []
    for s in services:
        models.append({
            'name': s,
            'model': s,
        })

    return {"models": models}


@app.get("/api/vector_databases/collections")
def vector_databases():
    response = vector_database_client.get_service_names()
    services = response.msg["service_names"]

    collections = []
    for s in services:
        info = vector_database_client.info(s)
        collections.append(info)

    return {"collections": collections}


@app.post("/api/chat")
async def chat(req: ChatCompletionRequest):
    if req.stream:
        def generate():
            for rep in chat_client.stream_chat(req.model, req.messages, req.options):
                if rep is None:
                    return JSONResponse(status_code=404,
                                        content={"error": f"model '{req.model}' not found"})

                rep = rep.msg
                if isinstance(rep, ChatCompletionStreamResponseDone):
                    response = json.dumps({
                        "model": req.model,
                        "created_at": get_timestamp(),
                        "message": {"role": "assistant", "content": ""},
                        "done": True
                    })
                    yield response
                    break
                else:
                    delta_content = rep.delta_content
                    response = json.dumps({"model": req.model,
                                           "created_at": get_timestamp(),
                                           "message": {"role": "assistant", "content": delta_content},
                                           "done": False
                                           })
                    yield response
                    yield "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")
    else:
        rep = chat_client.chat(req.model, req.messages, req.options)
        if rep is None:
            return JSONResponse(status_code=404,
                                content={"error": f"model '{req.model}' not found"})
        rep = rep.msg
        content = rep.content
        response = {"model": req.model,
                    "created_at": get_timestamp(),
                    "message": {"role": "assistant", "content": content},
                    "done": True}
        return response


@app.post("/api/embeddings")
def embeddings(req: EmbeddingsRequest):
    rep = retriever_client.encode(req.model, [req.prompt], req.options)
    if rep is None:
        return JSONResponse(status_code=404,
                            content={"error": f"model '{req.model}' not found"})

    return {"embedding": rep.vecs['dense_vecs'][0].tolist()}


@app.get("/api/default_qa_prompt_tmpl")
def reranker_models():
    return rag_client.default_qa_prompt_tmpl().msg


@app.post("/api/rag")
def rag(req: RAGRequest):
    response = rag_client.rag(**req.dict())

    if not isgenerator(response):
        return response.dict()

    def generate():
        for rep in response:
            yield json.dumps(rep.dict())
            yield "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, port=11434)
