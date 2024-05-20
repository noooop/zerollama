from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

ENGINE_CLASS = "zerollama.tasks.chat.engine.server:ZeroChatInferenceEngine"
MANAGER_NAME = "ZeroChatInferenceManager"
PROTOCOL = "chat"


Chat_ENGINE_CLASS = ENGINE_CLASS
Chat_MANAGER_NAME = MANAGER_NAME
Chat_PROTOCOL = PROTOCOL


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list = Field(default_factory=list)
    options: dict = Field(default_factory=dict)
    stream: bool = True


class ChatCompletionResponse(BaseModel):
    model: str
    finish_reason: str
    content: str

    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatCompletionStreamResponse(BaseModel):
    model: str
    delta_content: str
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponseDone(BaseModel):
    model: str
    finish_reason: Optional[str] = None
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = PROTOCOL
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )

