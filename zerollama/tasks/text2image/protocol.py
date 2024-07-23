from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

ENGINE_CLASS = "zerollama.tasks.text2image.engine.server:ZeroText2ImageInferenceEngine"
MANAGER_NAME = "ZeroText2ImageInferenceManager"
PROTOCOL = "text2image"


Text2Image_ENGINE_CLASS = ENGINE_CLASS
Text2Image_MANAGER_NAME = MANAGER_NAME
Text2Image_PROTOCOL = PROTOCOL


class Text2ImageModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = PROTOCOL
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )


class Text2ImageRequest(BaseModel):
    model: str
    prompt: str
    negative_prompt: Optional[str] = None
    options: Optional[dict] = None


class Text2ImageResponse(BaseModel):
    model: str
    image: Any
