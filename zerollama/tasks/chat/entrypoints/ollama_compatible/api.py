
import json
import datetime
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

from zerollama.tasks.chat.inference_engine.client import ChatClient
from .protocol import ChatCompletionRequest, ShowRequest


chat_client = ChatClient()
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
async def chat(ccr: ChatCompletionRequest):
    if ccr.stream:
        def generate():
            for rep in chat_client.stream_chat(ccr.model, ccr.messages, ccr.options):
                if rep is None:
                    return JSONResponse(status_code=404,
                                        content={"error": f"model '{ccr.model}' not found"})

                rep = rep.msg
                if not rep.done:
                    content = rep.content
                    response = json.dumps({"model": ccr.model,
                                           "created_at": get_timestamp(),
                                           "message": {"role": "assistant", "content": content},
                                           "done": False
                                           })
                    yield response
                    yield "\n"
                else:
                    response = json.dumps({
                        "model": ccr.model,
                        "created_at": get_timestamp(),
                        "message": {"role": "assistant", "content": ""},
                        "done": True
                    })
                    yield response
                    break
        return StreamingResponse(generate(), media_type="application/x-ndjson")
    else:
        rep = chat_client.chat(ccr.model, ccr.messages, ccr.options)
        if rep is None:
            return JSONResponse(status_code=404,
                                content={"error": f"model '{ccr.model}' not found"})
        rep = rep.msg
        content = rep.content
        response = {"model": ccr.model,
                    "created_at": get_timestamp(),
                    "message": {"role": "assistant", "content": content},
                    "done": True}
        return response


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=11434)
