from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

from zerollama.tasks.chat.protocol import ChatCompletionResponse, ChatCompletionStreamResponseDone, ChatCompletionStreamResponse

ENGINE_CLASS = "zerollama.tasks.qa.engine.server:ZeroQAInferenceEngine"
MANAGER_NAME = "ZeroQAInferenceManager"
PROTOCOL = "qa"


QA_ENGINE_CLASS = ENGINE_CLASS
QA_MANAGER_NAME = MANAGER_NAME
QA_PROTOCOL = PROTOCOL


class QACompletionRequest(BaseModel):
    model: str
    tools: Optional[list] = None
    messages: list = Field(default_factory=list)
    options: dict = Field(default_factory=dict)
    stream: bool = True


class QAModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = PROTOCOL
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )

