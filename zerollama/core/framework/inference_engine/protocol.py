from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

from zerollama.core.models.chat import ChatCompletionResponse, ChatCompletionStreamResponse


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list = Field(default_factory=list)
    options: dict = Field(default_factory=dict)
    stream: bool = True

