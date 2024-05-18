
import json
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


class ZeroServerResponse(BaseModel):
    state: str = Literal["ok", "error"]
    msg: Any = ""


class ZeroServerResponseOk(ZeroServerResponse):
    state: str = "ok"


class ZeroServerStreamResponseOk(ZeroServerResponse):
    state: str = "ok"
    snd_more: bool
    rep_id: int


class ZeroServerResponseError(ZeroServerResponse):
    state: str = "error"


class Meta(BaseModel):
    name: list
    dtype: str
    shape: tuple


class Timeout(Exception):
    pass


class ZeroMSQ(object):
    @staticmethod
    def load(data):
        payload = []
        meta = []
        msg = {}

        def _load(d, m, n):
            for k, v in d.items():
                if type(v) is np.ndarray:
                    meta.append(Meta(name=n+[k], dtype=str(v.dtype), shape=v.shape).dict())
                    payload.append(np.ascontiguousarray(v))
                elif isinstance(v, dict):
                    m[k] = {}
                    _load(v, m[k], n+[k])
                else:
                    m[k] = v

        _load(data, msg, [])

        req = {"msg": msg}

        if meta:
            req["meta"] = meta

        req = json.dumps(req).encode('utf8')
        return req, payload

    @staticmethod
    def unload(req, payload):
        req = json.loads(req)
        msg = req["msg"]

        if not payload:
            return msg

        meta = [Meta(**x) for x in req["meta"]]
        for m, p in zip(meta, payload):
            d = msg
            for i in range(len(m.name)-1):
                if m.name[i] not in d:
                    d[m.name[i]] = {}
                d = d[m.name[i]]

            d[m.name[-1]] = np.frombuffer(p, dtype=m.dtype).reshape(m.shape)
        return msg


if __name__ == '__main__':
    print(ZeroServerResponseOk().b)