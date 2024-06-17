import copy
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic import field_validator, computed_field
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

ENGINE_CLASS = "zerollama.tasks.ocr.text_line_detection.engine.server:ZeroTLDInferenceEngine"
MANAGER_NAME = "ZeroTLDInferenceManager"
PROTOCOL = "text_line_detection"

TLD_ENGINE_CLASS = ENGINE_CLASS
TLD_MANAGER_NAME = MANAGER_NAME
TLD_PROTOCOL = PROTOCOL


class TextLineDetectionModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = PROTOCOL
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )


class TextLineDetectionRequest(BaseModel):
    image: Any
    options: Optional[Dict] = None


class Bbox(BaseModel):
    bbox: List[float]

    @field_validator('bbox')
    @classmethod
    def check_4_elements(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError('bbox must have 4 elements')
        return v

    @property
    def polygon(self):
        return [[self.bbox[0], self.bbox[1]],
                [self.bbox[2], self.bbox[1]],
                [self.bbox[2], self.bbox[3]],
                [self.bbox[0], self.bbox[3]]]


class ColumnLine(Bbox):
    vertical: bool
    horizontal: bool


class BboxC(Bbox):
    confidence: Optional[float] = None


class TextDetectionResult(BaseModel):
    bboxes: List[BboxC]
    vertical_lines: List[ColumnLine]
    horizontal_lines: List[ColumnLine]
