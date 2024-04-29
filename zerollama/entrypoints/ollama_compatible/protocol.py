
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any, Union, Tuple


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list = []
    options: Optional[dict] = Field(default_factory=dict)
    stream: bool = True


class ShowRequest(BaseModel):
    name: str


class EmbeddingsRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[dict] = Field(default_factory=dict)