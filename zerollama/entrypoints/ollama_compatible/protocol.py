
from pydantic import BaseModel


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list = []
    options: dict = {}
    stream: bool = True


class ShowRequest(BaseModel):
    name: str