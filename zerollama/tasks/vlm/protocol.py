from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

ENGINE_CLASS = "zerollama.tasks.vlm.engine.server:ZeroVLMInferenceEngine"
MANAGER_NAME = "ZeroVLMInferenceManager"
PROTOCOL = "vlm"


VLM_ENGINE_CLASS = ENGINE_CLASS
VLM_MANAGER_NAME = MANAGER_NAME
VLM_PROTOCOL = PROTOCOL


class VLMModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = PROTOCOL
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )

