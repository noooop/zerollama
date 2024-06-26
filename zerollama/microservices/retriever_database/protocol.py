from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple
from zerollama.microservices.vector_database.protocol import TopKNode
from zerollama.core.framework.zero.protocol import ZeroServerResponseOk

ENGINE_CLASS = "zerollama.microservices.retriever_database.engine.server:ZeroRetrieverDatabaseEngine"
MANAGER_NAME = "ZeroRetrieverDatabaseManager"
PROTOCOL = "retriever_database"

RetrieverDatabase_ENGINE_CLASS = ENGINE_CLASS
RetrieverDatabase_MANAGER_NAME = MANAGER_NAME
RetrieverDatabase_PROTOCOL = PROTOCOL


class RetrieverDatabaseTopKRequest(BaseModel):
    retriever_model: str
    query: str
    k: int = 10


class RetrieverDatabaseTopKResponse(BaseModel):
    retriever_model: str
    data: List[TopKNode]





