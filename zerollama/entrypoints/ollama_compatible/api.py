
import json
import datetime
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from zerollama.core.framework.inference_engine.client import ChatClient
from zerollama.entrypoints.ollama_compatible.protocol import ChatCompletionRequest


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

    msg = {
        "models": [
            {"name": s, "model": s}
            for s in services
        ]
    }
    return msg


@app.post("/api/chat")
async def chat(ccr: ChatCompletionRequest):
    if ccr.stream:
        def generate():
            for rep in chat_client.stream_chat(ccr.model, ccr.messages, ccr.options):
                if rep.state != "ok":
                    return
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
        if rep.state != "ok":
            return
        rep = rep.msg
        content = rep.content
        response = json.dumps({"model": ccr.model,
                               "created_at": get_timestamp(),
                               "message": {"role": "assistant", "content": content},
                               "done": True
                               })
        return response


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=11434)
