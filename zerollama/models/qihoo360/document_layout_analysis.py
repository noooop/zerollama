
from zerollama.tasks.ocr.document_layout_analysis.interface import DocumentLayoutAnalysisModel


class LayoutAnalysis360(DocumentLayoutAnalysisModel):
    family = "360LayoutAnalysis"
    header = ["name", "weight_path"]
    info = [
        # name                             weight_path
        ["360LayoutAnalysis/paper-8n.pt",  "paper-8n.pt"],
        ["360LayoutAnalysis/report-8n.pt", "report-8n.pt"],
    ]
    inference_backend = "zerollama.models.qihoo360.backend.document_layout_analysis:LayoutAnalysis360"


if __name__ == '__main__':
    import os
    import cv2
    from pathlib import Path
    from zerollama.tasks.ocr.document_layout_analysis.utils import get_annotated_image

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/dla"

    input_path = dla_test_path / "input_sample.png"
    img_BGR = cv2.imread(str(input_path))
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    for model_name in [x[0] for x in LayoutAnalysis360.info]:
        model = LayoutAnalysis360.get_model(model_name)
        model.load()
        results = model.detection(img_RGB, options={"conf": 0.5, "line_width": 2})

        annotated_image = get_annotated_image(img_RGB, results)
        cv2.imwrite(f'result-{model_name.replace("/", "-")}.jpg', annotated_image)