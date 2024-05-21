from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import ZeroServerResponseOk, ZeroServerStreamResponseOk
from zerollama.tasks.chat.protocol import ChatCompletionResponse

ENGINE_CLASS = "zerollama.microservices.workflow.rag.engine.server:ZeroRAGEngine"
MANAGER_NAME = "RAGManager"
PROTOCOL = "rag"

RAG_ENGINE_CLASS = ENGINE_CLASS
RAG_MANAGER_NAME = MANAGER_NAME
RAG_PROTOCOL = PROTOCOL


class RAGRequest(BaseModel):
    question: str
    chat_model: str
    retriever_model: str
    reranker_model: str
    collection: str
    n_retriever_candidate: int = 10
    n_references: int = 3
    qa_prompt_tmpl_str: Optional[str] = None
    stream: bool = False
    return_references: bool = False


class RAGResponse(BaseModel):
    answer: Optional[ChatCompletionResponse] = None
    references: list = Field(default_factory=list)





