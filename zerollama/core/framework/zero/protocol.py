
from pydantic import BaseModel, ValidationError
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
    payload: Optional[list] = None

    @property
    def b(self):
        return self.model_dump_json().encode('utf8')


class ZeroServerResponseOk(ZeroServerResponse):
    state: str = "ok"


class ZeroServerStreamResponseOk(ZeroServerResponse):
    state: str = "ok"
    snd_more: bool
    rep_id: int


class ZeroServerResponseError(ZeroServerResponse):
    state: str = "error"


if __name__ == '__main__':
    print(ZeroServerResponseOk().b)