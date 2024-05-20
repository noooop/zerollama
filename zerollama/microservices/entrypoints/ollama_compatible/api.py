
import json
import datetime
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

from zerollama.tasks.chat.engine.client import ChatClient
from zerollama.tasks.retriever.engine.client import RetrieverClient
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponseDone
from .protocol import ChatCompletionRequest, ShowRequest, EmbeddingsRequest


chat_client = ChatClient()
retriever_client = RetrieverClient()
app = FastAPI()


def get_timestamp():
    return datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')


@app.get("/")
def health():
    return "zerollama is running"


@app.get("/api/tags")
def tags():
    response = chat_client.get_service_names()
    services = response.msg["service_names"]
    response = retriever_client.get_service_names()
    services += response.msg["service_names"]

    models = []
    for s in services:
        #details = chat_client.info(s).msg
        models.append({
            'name': s,
            'model': s,
            'modified_at': "",
            'size': "",
            'digest': "",
            "details": {}
        })

    return {"models": models}


@app.post("/api/show")
def show(req: ShowRequest):
    name = req.name
    rep = chat_client.info(name)

    if rep is None:
        return JSONResponse(status_code=404,
                            content={"error": f"model '{name}' not found"})

    details = rep.msg
    out = {
        'name': name,
        'model': name,
        'modified_at': "",
        'size': "",
        'digest': "",
        "details": details
    }
    return out


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


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=11434)
