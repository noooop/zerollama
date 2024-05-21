import typing

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from pydantic.main import IncEx

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

ENGINE_CLASS = "zerollama.tasks.retriever.engine.server:ZeroRetrieverInferenceEngine"
MANAGER_NAME = "ZeroRetrieverInferenceManager"
PROTOCOL = "retriever"

Retriever_ENGINE_CLASS = ENGINE_CLASS
Retriever_MANAGER_NAME = MANAGER_NAME
Retriever_PROTOCOL = PROTOCOL


class RetrieverModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = PROTOCOL
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )


class RetrieverRequest(BaseModel):
    model: str
    sentences: list = Field(default_factory=list)
    options: dict = Field(default_factory=dict)


class RetrieverResponse(BaseModel):
    model: str
    vecs: dict = Field(default_factory=dict)

