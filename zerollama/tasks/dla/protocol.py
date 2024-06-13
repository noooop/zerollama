from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

ENGINE_CLASS = "zerollama.tasks.dla.engine.server:ZeroDLAInferenceEngine"
MANAGER_NAME = "ZeroDLAInferenceManager"
PROTOCOL = "dla"


DLA_ENGINE_CLASS = ENGINE_CLASS
DLA_MANAGER_NAME = MANAGER_NAME
DLA_PROTOCOL = PROTOCOL


class DLAModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = PROTOCOL
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )


class DLARequest(BaseModel):
    image: Any
    options: Optional[Dict] = None