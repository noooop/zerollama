from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import ZeroServerResponseOk

MANAGER_NAME = "ZeroVectorDatabaseManager"
PROTOCOL = "vector_database"


class VectorDatabaseTopKRequest(BaseModel):
    embedding_model: str
    query_dense_vecs: Any  # pydantic does not support numpy types.
    k: int = 10


class TopKNode(BaseModel):
    index: int
    score: float
    node: dict


class VectorDatabaseTopKResponse(BaseModel):
    embedding_model: str
    data: List[TopKNode]





