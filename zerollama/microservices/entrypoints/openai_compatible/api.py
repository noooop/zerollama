import inspect
import traceback

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

from zerollama.tasks.chat.engine.client import ChatClient
from zerollama.tasks.retriever.engine.client import RetrieverClient

from zerollama.tasks.chat.protocol import ChatCompletionStreamResponseDone
import zerollama.microservices.entrypoints.openai_compatible.protocol as protocol

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
    return protocol.ModelList(data=[protocol.ModelCard(id=s) for s in services])


@app.get("/v1/models/{model_id:path}", name="path-convertor")
def model(model_id: str):
    rep = chat_client.info(model_id)
    return protocol.ModelCard(id=model_id)


@app.post("/v1/chat/completions")
def chat(req: protocol.ChatCompletionRequest):
    options_key = {"temperature", "top_k", "top_p", "max_tokens", "presence_penalty", "frequency_penalty"}
    options = {k: v for k, v in req.model_dump().items() if k in options_key and v is not None}

    tool_dicts = None if req.tools is None else [tool.model_dump() for tool in req.tools]

    try:
        response = chat_client.chat(name=req.model, tools=tool_dicts, messages=req.messages, stream=req.stream,
                                    options=options)
    except Exception as e:
        traceback.print_exc()
        raise e

    if not inspect.isgenerator(response):
        data = protocol.ChatCompletionResponse(**{
            "model": response.model,
            "choices": [protocol.ChatCompletionResponseChoice(**{
                "index": 0,
                "message": protocol.ChatMessage(role="assistant", content=response.content),
                "finish_reason": response.finish_reason
            })],
            "usage": protocol.UsageInfo(prompt_tokens=response.prompt_tokens,
                                        total_tokens=response.total_tokens,
                                        completion_tokens=response.completion_tokens)
        })
        return data
    else:
        def generate():
            for rep in response:
                if isinstance(rep, ChatCompletionStreamResponseDone):
                    data = protocol.ChatCompletionStreamResponse(**{
                        "model": rep.model,
                        "choices": [protocol.ChatCompletionResponseStreamChoice(**{
                            "index": 0,
                            "delta": protocol.DeltaMessage(),
                            "finish_reason": rep.finish_reason
                        })]
                    })
                    data = data.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"
                    break
                else:
                    data = protocol.ChatCompletionStreamResponse(**{
                        "model": rep.model,
                        "choices": [protocol.ChatCompletionResponseStreamChoice(**{
                            "index": 0,
                            "delta": protocol.DeltaMessage(role="assistant", content=rep.delta_content)
                        })]
                    })
                    data = data.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, port=8080)
