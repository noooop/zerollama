from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple
from zerollama.tasks.ocr.text_line_detection.protocol import BboxC
from zerollama.tasks.ocr.document_layout_analysis.protocol import LayoutBox

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerStreamResponseOk,
    ZeroServerResponseError
)

ENGINE_CLASS = "zerollama.tasks.ocr.table_structure_recognition.engine.server:ZeroTSRInferenceEngine"
MANAGER_NAME = "ZeroTSRInferenceManager"
PROTOCOL = "table_structure_recognition"


TSR_ENGINE_CLASS = ENGINE_CLASS
TSR_MANAGER_NAME = MANAGER_NAME
TSR_PROTOCOL = PROTOCOL


class TableStructureRecognitionModelConfig(BaseModel):
    name: str
    info: dict
    family: str
    protocol: str = PROTOCOL
    model_kwargs: dict

    model_config = ConfigDict(
        protected_namespaces=()
    )


class TableStructureRecognitionRequest(BaseModel):
    image: Any
    options: Optional[Dict] = None


class TableStructureRecognitionResult(BaseModel):
    bboxes: List[LayoutBox]