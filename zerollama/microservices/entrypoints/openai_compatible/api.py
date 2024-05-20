
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

    if rep is None:
        return JSONResponse(status_code=404,
                            content={"error": f"model '{model_id}' not found"})

    return ModelCard(id=model_id)


@app.post("/v1/chat/completions")
def chat(req: ChatCompletionRequest):
    options = {}

    if req.stream:
        def generate():
            for rep in chat_client.stream_chat(req.model, req.messages, options):
                if rep is None:
                    return JSONResponse(status_code=404,
                                        content={"error": f"model '{req.model}' not found"})

                rep = rep.msg

                if isinstance(rep, ChatCompletionStreamResponseDone):
                    response = ChatCompletionStreamResponse(**{
                        "model": rep.model,
                        "choices": [ChatCompletionResponseStreamChoice(**{
                            "index": 0,
                            "delta": DeltaMessage(),
                            "finish_reason": rep.finish_reason
                        })]
                    })
                    yield response.model_dump_json()
                    break
                else:
                    response = ChatCompletionStreamResponse(**{
                        "model": rep.model,
                        "choices": [ChatCompletionResponseStreamChoice(**{
                            "index": 0,
                            "delta": DeltaMessage(role="assistant", content=rep.delta_content)
                        })]
                    })
                    yield response.model_dump_json()
                    yield "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")
    else:
        rep = chat_client.chat(req.model, req.messages, options)
        if rep is None:
            return JSONResponse(status_code=404,
                                content={"error": f"model '{req.model}' not found"})
        rep = rep.msg
        response = ChatCompletionResponse(**{
            "model": rep.model,
            "choices": [ChatCompletionResponseChoice(**{
                "index": 0,
                "message": ChatMessage(role="assistant", content=rep.content),
                "finish_reason": rep.finish_reason
            })],
            "usage": UsageInfo(prompt_tokens=rep.prompt_tokens,
                               total_tokens=rep.total_tokens,
                               completion_tokens=rep.completion_tokens)
        })
        return response


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingsRequest):
    options = {}
    sentences = [req.input] if isinstance(req.input, str) else req.input

    print(req)

    rep = retriever_client.encode(req.model, sentences, options)
    if rep is None:
        return JSONResponse(status_code=404,
                            content={"error": f"model '{req.model}' not found"})

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
