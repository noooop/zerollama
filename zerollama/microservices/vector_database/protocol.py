from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import ZeroServerResponseOk

ENGINE_CLASS = "zerollama.microservices.vector_database.engine.server:ZeroVectorDatabaseEngine"
MANAGER_NAME = "ZeroVectorDatabaseManager"
PROTOCOL = "vector_database"

VectorDatabase_ENGINE_CLASS = ENGINE_CLASS
VectorDatabase_MANAGER_NAME = MANAGER_NAME
VectorDatabase_PROTOCOL = PROTOCOL


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





