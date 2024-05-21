from pydantic import BaseModel, Field, ValidationError, ConfigDict
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

from zerollama.tasks.chat.protocol import ChatCompletionResponse, ChatCompletionStreamResponse


class StartRequest(BaseModel):
    name: str
    engine_kwargs: dict = Field(default_factory=dict)


class TerminateRequest(BaseModel):
    name: str


class StatusRequest(BaseModel):
    name: str
