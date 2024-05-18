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

MANAGER_NAME = "ZeroRerankerInferenceManager"
PROTOCOL = "reranker"


class RerankerModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = PROTOCOL
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )


class RerankerRequest(BaseModel):
    model: str
    sentence_pairs: list = Field(default_factory=list)
    options: dict = Field(default_factory=dict)


class RerankerResponse(BaseModel):
    model: str
    vecs: dict = Field(default_factory=dict)

