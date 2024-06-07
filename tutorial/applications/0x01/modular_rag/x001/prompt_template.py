
import os
import json
from pathlib import Path
from easydict import EasyDict as edict

rq_rag_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent.parent.parent / "RAG-query-rewriting"

prompt_template_path = rq_rag_path / "generate/inprompts"


promptlines = open(prompt_template_path / 'myprompt.jsonl', 'r', encoding="utf-8").readlines()

task = set()
# {'step1', 'wiki', 'rewrite2', 'searchre', 'step2', 'rewrite'}

template = edict({})

i = 0
for line in promptlines:
    line = json.loads(line)
    task.add(line["task"])
    if line["task"] == "rewrite":
        template[f"template-{i}"] = line["prompt"]
        i += 1

if __name__ == '__main__':
    print(len(template.keys()))
