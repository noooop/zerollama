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

ENGINE_CLASS = "zerollama.tasks.super_resolution.engine.server:ZeroSRInferenceEngine"
MANAGER_NAME = "ZeroSRInferenceManager"
PROTOCOL = "super_resolution"

SR_ENGINE_CLASS = ENGINE_CLASS
SR_MANAGER_NAME = MANAGER_NAME
SR_PROTOCOL = PROTOCOL


class SRModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = PROTOCOL
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )


class SRRequest(BaseModel):
    model: str
    image: Any
    options: dict = Field(default_factory=dict)


class SRResponse(BaseModel):
    model: str
    image: Any

