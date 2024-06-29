
from zerollama.tasks.ocr.document_layout_analysis.interface import DocumentLayoutAnalysisModel
from zerollama.models.qihoo360.utils import paper_class_names, report_class_names, publaynet_class_names, general6_class_names


class LayoutAnalysis360(DocumentLayoutAnalysisModel):
    family = "360LayoutAnalysis"
    header = ["name", "weight_path", "class_names"]
    info = [
        # name                                weight_path
        ["360LayoutAnalysis/paper-8n.pt",     "paper-8n.pt",      paper_class_names],
        ["360LayoutAnalysis/report-8n.pt",    "report-8n.pt",     report_class_names],
        ["360LayoutAnalysis/publaynet-8n.pt", "publaynet-8n.pt",  publaynet_class_names],
        ["360LayoutAnalysis/general6-8n.pt",  "general6-8n.pt",   general6_class_names],
    ]
    inference_backend = "zerollama.models.qihoo360.backend.document_layout_analysis:LayoutAnalysis360"


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.tasks.ocr.document_layout_analysis.utils import get_annotated_image

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/dla"

    input_path = dla_test_path / "input_sample.png"
    image = Image.open(input_path)

    for model_name in [x[0] for x in LayoutAnalysis360.info]:
        model = LayoutAnalysis360.get_model(model_name)
        model.load()
        results = model.detection(image, options={"conf": 0.5, "line_width": 2})

        annotated_image = get_annotated_image(image, results)
        annotated_image.save(f'result-{model_name.replace("/", "-")}.jpg')