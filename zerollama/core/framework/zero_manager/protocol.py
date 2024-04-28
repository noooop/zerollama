from pydantic import BaseModel, Field, ValidationError, ConfigDict
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

from zerollama.core.models.chat import ChatCompletionResponse, ChatCompletionStreamResponse


class StartRequest(BaseModel):
    name: str
    protocol: str
    model_kwargs: dict = Field(default_factory=dict)

    model_config = ConfigDict(
        protected_namespaces=()
    )


class TerminateRequest(BaseModel):
    name: str


class StatusRequest(BaseModel):
    name: str
