

from zerollama.models.deepseek.model import DeepSeekLLM
from zerollama.models.openbmb.minicpm import MiniCPM
from zerollama.models.qwen.qwen1_5 import Qwen1_5
from zerollama.models.yi.model import Yi
from zerollama.models.llama.llama3 import Llama3
from zerollama.models.llama.llama3zh import Llama3ZH


CHAT_MODELS = [DeepSeekLLM, MiniCPM, Qwen1_5, Yi, Llama3, Llama3ZH]
CHAT_MODELS_NAME_MAP = dict()
CHAT_MODELS_FAMILY_MAP = {m.family: m for m in CHAT_MODELS}

for m in CHAT_MODELS:
    CHAT_MODELS_NAME_MAP.update({n: m for n in m.model_names()})


def chat_families():
    return list(CHAT_MODELS_FAMILY_MAP.keys())


def chat_family(name):
    return CHAT_MODELS_FAMILY_MAP.get(name, None)


def get_chat_model_config_by_name(model_name):
    if model_name not in CHAT_MODELS_NAME_MAP:
        return None

    return CHAT_MODELS_NAME_MAP[model_name].get_model_config(model_name)


if __name__ == '__main__':
    print(chat_families())

    print(chat_family("Qwen1.5"))

    model_name = "Qwen/Qwen1.5-0.5B-Chat"
    config = get_chat_model_config_by_name(model_name)
    print(config)

    for name in chat_families():
        print("="*80)
        print(name)
        c = chat_family(name)
        print(c.prettytable())

