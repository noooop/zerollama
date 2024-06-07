# Adapted from
# https://github.com/chanchimin/RQ-RAG/blob/main/data_curation/llm_agent/prompt_template/Template.py

import os
from pathlib import Path
from easydict import EasyDict as edict

rq_rag_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent.parent.parent / "RQ-RAG"

prompt_template_path = rq_rag_path / "data_curation/llm_agent/prompt_template"
cur_dir = prompt_template_path

with open(os.path.join(cur_dir, "QueryRewriterTemplate.txt")) as f:
    QueryRewriterTemplate = "".join(f.readlines())

with open(os.path.join(cur_dir, "QueryJudgerTemplate.txt")) as f:
    QueryJudgerTemplate = "".join(f.readlines())

with open(os.path.join(cur_dir, "QueryGeneratorTemplate.txt")) as f:
    QueryGeneratorTemplate = "".join(f.readlines())

with open(os.path.join(cur_dir, "UnambiguousGeneratorTemplateLong.txt")) as f:
    UnambiguousGeneratorTemplateLong = "".join(f.readlines())

with open(os.path.join(cur_dir, "UnambiguousGeneratorTemplateShort.txt")) as f:
    UnambiguousGeneratorTemplateShort = "".join(f.readlines())

with open(os.path.join(cur_dir, "DecomposeGeneratorTemplate.txt")) as f:
    DecomposeGeneratorTemplate = "".join(f.readlines())

with open(os.path.join(cur_dir, "MultiTurnGeneratorTemplate.txt")) as f:
    MultiTurnGeneratorTemplate = "".join(f.readlines())

with open(os.path.join(cur_dir, "MultiTurnGeneratorTemplateForAns.txt")) as f:
    MultiTurnGeneratorTemplateForAns = "".join(f.readlines())

with open(os.path.join(cur_dir, "QAGeneratorTemplate.txt")) as f:
    QAGeneratorTemplate = "".join(f.readlines())


template = edict({"QueryRewriterTemplate": QueryRewriterTemplate,
                  "QueryJudgerTemplate": QueryJudgerTemplate,
                  "QueryGeneratorTemplate": QueryGeneratorTemplate,
                  "UnambiguousGeneratorTemplateLong": UnambiguousGeneratorTemplateLong,
                  "UnambiguousGeneratorTemplateShort": UnambiguousGeneratorTemplateShort,
                  "DecomposeGeneratorTemplate": DecomposeGeneratorTemplate,
                  "MultiTurnGeneratorTemplate": MultiTurnGeneratorTemplate,
                  "MultiTurnGeneratorTemplateForAns": MultiTurnGeneratorTemplateForAns,
                  "QAGeneratorTemplate": QAGeneratorTemplate})