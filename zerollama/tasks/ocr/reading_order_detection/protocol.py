import copy
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic import field_validator, computed_field
from typing import Literal, Optional, List, Dict, Any, Union, Tuple
from zerollama.tasks.ocr.text_line_detection.protocol import Bbox
from zerollama.tasks.ocr.document_layout_analysis.protocol import DocumentLayoutAnalysisResult

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

ENGINE_CLASS = "zerollama.tasks.ocr.reading_order_detection.engine.server:ZeroRODInferenceEngine"
MANAGER_NAME = "ZeroRODInferenceManager"
PROTOCOL = "reading_order_detection"

ROD_ENGINE_CLASS = ENGINE_CLASS
ROD_MANAGER_NAME = MANAGER_NAME
ROD_PROTOCOL = PROTOCOL


class ReadingOrderDetectionModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = PROTOCOL
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )


class ReadingOrderDetectionRequest(BaseModel):
    image: Any
    layout: Optional[DocumentLayoutAnalysisResult] = None
    options: Optional[Dict] = None


class OrderBox(Bbox):
    position: int


class ReadingOrderDetectionResult(BaseModel):
    bboxes: List[OrderBox]
