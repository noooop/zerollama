
import json
import inspect
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
    try:
        name = req.name
        rep = chat_client.info(name)
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
    except Exception:
        return JSONResponse(status_code=404,
                            content={"error": f"model '{req.name}' not found"})


@app.post("/api/chat")
def chat(req: ChatCompletionRequest):
    try:
        response = chat_client.chat(name=req.model, messages=req.messages, stream=req.stream, options=req.options)
        if not inspect.isgenerator(response):
            data = {"model": response.model,
                    "created_at": get_timestamp(),
                    "message": {"role": "assistant", "content": response.content},
                    "done": True,

                    "done_reason": response.finish_reason,
                    "eval_count": response.completion_tokens,
                    "prompt_eval_count": response.prompt_tokens}
            return data
        else:
            def generate():
                for rep in response:
                    if isinstance(rep, ChatCompletionStreamResponseDone):
                        data = json.dumps({
                            "model": req.model,
                            "created_at": get_timestamp(),
                            "message": {"role": "assistant", "content": ""},
                            "done": True,

                            "done_reason": rep.finish_reason,
                            "eval_count": rep.completion_tokens,
                            "prompt_eval_count": rep.prompt_tokens
                        })
                        yield data
                        yield "\n"
                        break
                    else:
                        delta_content = rep.delta_content
                        data = json.dumps({"model": req.model,
                                           "created_at": get_timestamp(),
                                           "message": {"role": "assistant", "content": delta_content},
                                           "done": False
                        })
                        yield data
                        yield "\n"
            return StreamingResponse(generate(), media_type="application/x-ndjson")
    except RuntimeError:
        return JSONResponse(status_code=404,
                            content={"error": f"model '{req.name}' not found"})


@app.post("/api/embeddings")
def embeddings(req: EmbeddingsRequest):
    try:
        rep = retriever_client.encode(req.model, [req.prompt], req.options)
        return {"embedding": rep.vecs['dense_vecs'][0].tolist()}
    except RuntimeError:
        return JSONResponse(status_code=404,
                            content={"error": f"model '{req.name}' not found"})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=11434)
