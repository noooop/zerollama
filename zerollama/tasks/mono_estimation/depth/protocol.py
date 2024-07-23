from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

ENGINE_CLASS = "zerollama.tasks.mono_estimation.depth.engine.server:ZeroDepthEstimationInferenceEngine"
MANAGER_NAME = "ZeroDepthEstimationInferenceManager"
PROTOCOL = "depth_estimation"


DepthEstimation_ENGINE_CLASS = ENGINE_CLASS
DepthEstimation_MANAGER_NAME = MANAGER_NAME
DepthEstimation_PROTOCOL = PROTOCOL


class DepthEstimationModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = PROTOCOL
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )


class DepthEstimationRequest(BaseModel):
    model: str
    image: Any


class DepthEstimationResponse(BaseModel):
    model: str
    depth: Any
