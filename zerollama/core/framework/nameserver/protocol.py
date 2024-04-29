from pydantic import BaseModel, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple

from zerollama.core.framework.zero.protocol import (
    ZeroServerRequest,
    ZeroServerResponse,
    ZeroServerResponseOk,
    ZeroServerResponseError
)


class ServerInfo(BaseModel):
    name: str
    host: str
    port: str | int
    protocol: str


class GetServicesRequest(BaseModel):
    name: str
    protocol: str


class GetServiceNamesRequest(BaseModel):
    protocol: str