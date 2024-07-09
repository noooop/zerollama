

from zerollama.models.deepseek.chat import DeepSeekLLM
from zerollama.models.openbmb.chat import MiniCPM
from zerollama.models.qwen.chat import Qwen1_5, Qwen1_5_GGUF, Qwen2
from zerollama.models.yi.chat import Yi, Yi_1_5
from zerollama.models.llama.llama3 import Llama3
from zerollama.models.llama.llama3zh import Llama3ZH
from zerollama.models.xverse.chat import XVERSE, XVERSE_GGUF
from zerollama.models.yuan.chat import Yuan, Yuan_GGUF
from zerollama.models.mistral.chat import Mistral
from zerollama.models.thudm.chat import GLM4
from zerollama.models.bilibili.chat import Index, IndexGGUF
from zerollama.models.qihoo360.chat import Zhinao360
from zerollama.models.lnternlm.chat import InternLM, InternLM_GGUF

MODELS = [DeepSeekLLM, MiniCPM, Qwen1_5, Qwen1_5_GGUF, Qwen2,
          Yi, Yi_1_5, Llama3, Llama3ZH, XVERSE, XVERSE_GGUF,
          Yuan, Yuan_GGUF, Mistral, GLM4, Index, IndexGGUF,
          Zhinao360, InternLM, InternLM_GGUF]

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

    print(family("Qwen1.5"))

    model_name = "Qwen/Qwen1.5-0.5B-Chat"
    config = get_model_config_by_name(model_name)
    print(config)

    for name in families():
        print("="*80)
        print(name)
        c = family(name)
        print(c.prettytable())

