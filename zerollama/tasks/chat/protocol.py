from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list = Field(default_factory=list)
    options: dict = Field(default_factory=dict)
    stream: bool = True


class ChatCompletionResponse(BaseModel):
    model: str
    prompt_length: int
    response_length: int
    finish_reason: str
    content: str


class ChatCompletionStreamResponse(BaseModel):
    model: str
    prompt_length: int
    response_length: int
    finish_reason: Optional[str] = None
    content: Optional[str] = None
    done: bool


class ChatModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = "chat"
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )

