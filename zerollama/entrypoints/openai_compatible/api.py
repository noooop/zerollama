
import time
import datetime
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

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


app = FastAPI()


@app.post("/v1/chat/completions")
async def chat(request: ChatCompletionRequest):
    if request.messages and request.messages[-1].role == 'user':
      resp_content = "As a mock AI Assitant, I can only echo your last message:" + request.messages[-1].content
    else:
      resp_content = "As a mock AI Assitant, I can only echo your last message, but there were no messages!"

    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{
            "message": ChatMessage(role="assistant", content=resp_content)
        }]
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8080)
