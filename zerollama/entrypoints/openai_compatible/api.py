
import json
import datetime
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

from zerollama.core.framework.inference_engine.client import ChatClient
from zerollama.entrypoints.openai_compatible.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorResponse,
    LogProbs,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
)

chat_client = ChatClient()
app = FastAPI()


def get_timestamp():
    return datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')


@app.get("/")
def index():
    return Response(status_code=200)


@app.get("/health")
def health():
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models() -> ModelList:
    response = chat_client.get_service_names()
    services = response.msg["service_names"]
    services.sort()

    model_cards = [
        ModelCard(id=s,
                  root=s,
                  permission=[ModelPermission()])
        for s in services
    ]
    return ModelList(data=model_cards)


@app.post("/v1/chat/completions")
async def chat(ccr: ChatCompletionRequest):
    if ccr.stream:
        async def generate():
            options = {}
            for res in chat_client.stream_chat(ccr.model, ccr.messages, options):

                content = res["content"]
                response = json.dumps({"model": msg.model,
                                       "created_at": get_timestamp(),
                                       "message": {"role": "assistant", "content": content},
                                       "done": False
                                       })
                yield response
                yield "\n"

            response = json.dumps({
                "model": msg.model,
                "created_at": get_timestamp(),
                "message": {"role": "assistant", "content": ""},
                "done": True
            })
            yield response

        return StreamingResponse(generator, media_type="text/event-stream")

    else:
        options = {}
        rep = chat_client.chat(ccr.model, ccr.messages, options)
        choices = [ChatCompletionResponseChoice(index=0,
                                                message=ChatMessage(**{"role": "assistant", "content": rep["content"]}),
                                                finish_reason=rep["finish_reason"])]
        return ChatCompletionResponse(model=ccr.model, choices=choices)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=11434)
