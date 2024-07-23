from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple
from zerollama.tasks.ocr.text_line_detection.protocol import BboxC, TextDetectionResult

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

ENGINE_CLASS = "zerollama.tasks.ocr.document_layout_analysis.engine.server:ZeroDLAInferenceEngine"
MANAGER_NAME = "ZeroDLAInferenceManager"
PROTOCOL = "document_layout_analysis"


DLA_ENGINE_CLASS = ENGINE_CLASS
DLA_MANAGER_NAME = MANAGER_NAME
DLA_PROTOCOL = PROTOCOL


class DocumentLayoutAnalysisModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = PROTOCOL
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )


class DocumentLayoutAnalysisRequest(BaseModel):
    image: Any
    lines: Optional[TextDetectionResult] = None
    options: Optional[Dict] = None


class LayoutBox(BboxC):
    label: str
    id: int


class DocumentLayoutAnalysisResult(BaseModel):
    bboxes: List[LayoutBox]
    class_names: List[str]