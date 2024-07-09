
import inspect
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

from zerollama.tasks.chat.engine.client import ChatClient
from zerollama.tasks.retriever.engine.client import RetrieverClient

from zerollama.tasks.chat.protocol import ChatCompletionStreamResponseDone
from .protocol import (
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
retriever_client = RetrieverClient()
app = FastAPI()


@app.get("/")
def health():
    return "zerollama is running"


@app.get("/v1/models")
def models():
    response = chat_client.get_service_names()
    services = response.msg["service_names"]
    response = retriever_client.get_service_names()
    services += response.msg["service_names"]
    return ModelList(data=[ModelCard(id=s) for s in services])


@app.get("/v1/models/{model_id:path}", name="path-convertor")
def model(model_id: str):
    rep = chat_client.info(model_id)
    return ModelCard(id=model_id)


@app.post("/v1/chat/completions")
def chat(req: ChatCompletionRequest):
    options = {}

    response = chat_client.chat(name=req.model, messages=req.messages, stream=req.stream, options=options)

    if not inspect.isgenerator(response):
        data = ChatCompletionResponse(**{
            "model": response.model,
            "choices": [ChatCompletionResponseChoice(**{
                "index": 0,
                "message": ChatMessage(role="assistant", content=response.content),
                "finish_reason": response.finish_reason
            })],
            "usage": UsageInfo(prompt_tokens=response.prompt_tokens,
                               total_tokens=response.total_tokens,
                               completion_tokens=response.completion_tokens)
        })
        return data
    else:
        def generate():
            for rep in response:
                if isinstance(rep, ChatCompletionStreamResponseDone):
                    data = ChatCompletionStreamResponse(**{
                        "model": rep.model,
                        "choices": [ChatCompletionResponseStreamChoice(**{
                            "index": 0,
                            "delta": DeltaMessage(),
                            "finish_reason": rep.finish_reason
                        })]
                    })
                    data = data.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"
                    break
                else:
                    data = ChatCompletionStreamResponse(**{
                        "model": rep.model,
                        "choices": [ChatCompletionResponseStreamChoice(**{
                            "index": 0,
                            "delta": DeltaMessage(role="assistant", content=rep.delta_content)
                        })]
                    })
                    data = data.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingsRequest):
    options = {}
    sentences = [req.input] if isinstance(req.input, str) else req.input

    rep = retriever_client.encode(req.model, sentences, options)

    vecs = rep.vecs['dense_vecs']
    if isinstance(req.input, str):
        vecs = vecs[0]

    return EmbeddingsResponse(**{
        "model": rep.model,
        "data": [{"index": 0, "object": "embedding", "embedding": vecs.tolist()}]
    })


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8080)
