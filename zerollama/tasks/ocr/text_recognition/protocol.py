import copy
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic import field_validator, computed_field
from typing import Literal, Optional, List, Dict, Any, Union, Tuple
from zerollama.tasks.ocr.text_line_detection.protocol import Bbox, TextDetectionResult

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

ENGINE_CLASS = "zerollama.tasks.ocr.text_recognition.engine.server:ZeroTRInferenceEngine"
MANAGER_NAME = "ZeroTRInferenceManager"
PROTOCOL = "text_recognition"

TR_ENGINE_CLASS = ENGINE_CLASS
TR_MANAGER_NAME = MANAGER_NAME
TR_PROTOCOL = PROTOCOL


class TextRecognitionModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = PROTOCOL
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )


class TextRecognitionRequest(BaseModel):
    image: Any
    lang: List[str]
    lines: TextDetectionResult
    options: Optional[Dict] = None


class TextLine(Bbox):
    text: str


class TextRecognitionResult(BaseModel):
    text_lines: List[TextLine]
