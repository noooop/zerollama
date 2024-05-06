
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional, List, Dict, Any, Union, Tuple


def loc_to_dot_sep(loc: Tuple[Union[str, int], ...]) -> str:
    path = ''
    for i, x in enumerate(loc):
        if isinstance(x, str):
            if i > 0:
                path += '.'
            path += x
        elif isinstance(x, int):
            path += f'[{x}]'
        else:
            raise TypeError('Unexpected type')
    return path


def convert_errors(e: ValidationError) -> List[Dict[str, Any]]:
    new_errors: List[Dict[str, Any]] = e.errors()
    for error in new_errors:
        error['loc'] = loc_to_dot_sep(error['loc'])
    del error["url"]
    return new_errors


class ZeroServerRequest(BaseModel):
    method: str
    uuid: Optional[bytes] = None
    req_id: Optional[bytes] = None
    data: Optional[dict] = None
    payload: Optional[list] = None


class ZeroServerResponse(BaseModel):
    state: str = Literal["ok", "error"]
    msg: Any = ""

    @property
    def b(self):
        return [self.model_dump_json().encode('utf8')]


class ZeroServerResponseOk(ZeroServerResponse):
    state: str = "ok"


class ZeroServerStreamResponseOk(ZeroServerResponse):
    state: str = "ok"
    snd_more: bool
    rep_id: int


class ZeroServerResponseError(ZeroServerResponse):
    state: str = "error"


class Meta(BaseModel):
    name: str
    dtype: str
    shape: tuple


class ZeroServerResponseOkWithPayload(ZeroServerResponseOk):
    meta: List[Meta] = Field(default_factory=list)
    payload: Optional[list] = None
    tensor_field: str

    @classmethod
    def load(cls, m, tensor_field):
        msg = m.model_dump(exclude={tensor_field})

        meta = []
        payload = []

        data = getattr(m, tensor_field)

        for k, vecs in data.items():
            if vecs is None:
                continue
            meta.append(Meta(name=k, dtype=str(vecs.dtype), shape=vecs.shape))
            payload.append(np.ascontiguousarray(vecs))

        return cls(msg=msg, tensor_field=tensor_field, meta=meta, payload=payload)

    def unload(self):
        msg = self.msg.copy()

        msg[self.tensor_field] = {}
        for m, p in zip(self.meta, self.payload):
            msg[self.tensor_field][m.name] = p
        return msg

    @property
    def b(self):
        return [self.model_dump_json(exclude={"payload"}).encode('utf8')] + self.payload


class Timeout(Exception):
    pass


if __name__ == '__main__':
    print(ZeroServerResponseOk().b)