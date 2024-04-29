

from zerollama.models.baai.bge import BGEM3


MODELS = [BGEM3]
MODELS_NAME_MAP = dict()
MODELS_FAMILY_MAP = {m.family: m for m in MODELS}

for m in MODELS:
    MODELS_NAME_MAP.update({n: m for n in m.model_names()})


def families():
    return list(MODELS_FAMILY_MAP.keys())


def family(name):
    return MODELS_FAMILY_MAP.get(name, None)


def get_model_config_by_name(model_name):
    if model_name not in MODELS_NAME_MAP:
        return None

    return MODELS_NAME_MAP[model_name].get_model_config(model_name)


def get_model_by_name(model_name):
    if model_name not in MODELS_NAME_MAP:
        return None

    return MODELS_NAME_MAP[model_name]


if __name__ == '__main__':
    print(families())

    print(family("BGE"))

    model_name = "BAAI/bge-m3"
    config = get_model_config_by_name(model_name)
    print(config)

    for name in families():
        print("="*80)
        print(name)
        c = family(name)
        print(c.prettytable())

