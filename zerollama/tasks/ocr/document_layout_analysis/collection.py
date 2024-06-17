
from zerollama.models.qihoo360.document_layout_analysis import LayoutAnalysis360
from zerollama.models.moured.document_layout_analysis import YOLOv10DocumentLayoutAnalysis
from zerollama.models.surya.document_layout_analysis import SuryaDocumentLayoutAnalysis
from zerollama.models.ragflow.document_layout_analysis import DeepDocDocumentLayoutAnalysis

MODELS = [LayoutAnalysis360, YOLOv10DocumentLayoutAnalysis, SuryaDocumentLayoutAnalysis, DeepDocDocumentLayoutAnalysis]
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

    print(family("YOLOv10-Document-Layout-Analysis"))

    model_name = "YOLOv10-Document-Layout-Analysis/yolov10b_best.pt"
    config = get_model_config_by_name(model_name)
    print(config)

    for name in families():
        print("="*80)
        print(name)
        c = family(name)
        print(c.prettytable())

