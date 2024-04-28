

from zerollama.models.collection.chat import get_chat_model_by_name


PROTOCOL_MAP = {
    "chat": get_chat_model_by_name
}


def get_model(protocol, model_name):
    if protocol not in PROTOCOL_MAP:
        return None

    return PROTOCOL_MAP[protocol](model_name)