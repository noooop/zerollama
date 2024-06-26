
from zerollama.tasks.ocr.document_layout_analysis.interface import DocumentLayoutAnalysisModel


class YOLOv10DocumentLayoutAnalysis(DocumentLayoutAnalysisModel):
    family = "YOLOv10-Document-Layout-Analysis"
    header = ["name", "weight_path"]
    info = [
        # name                                                weight_path
        ["YOLOv10-Document-Layout-Analysis/yolov10b_best.pt", "yolov10b_best.pt"],
        ["YOLOv10-Document-Layout-Analysis/yolov10l_best.pt", "yolov10l_best.pt"],
        ["YOLOv10-Document-Layout-Analysis/yolov10m_best.pt", "yolov10m_best.pt"],
        ["YOLOv10-Document-Layout-Analysis/yolov10n_best.pt", "yolov10n_best.pt"],
        ["YOLOv10-Document-Layout-Analysis/yolov10s_best.pt", "yolov10s_best.pt"],
        ["YOLOv10-Document-Layout-Analysis/yolov10x_best.pt", "yolov10x_best.pt"],
    ]
    inference_backend = "zerollama.models.moured.backend.document_layout_analysis:YOLOv10DocumentLayoutAnalysis"


if __name__ == '__main__':
    import os
    import cv2
    from pathlib import Path
    from zerollama.tasks.ocr.document_layout_analysis.utils import get_annotated_image

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/dla"

    input_path = dla_test_path / "input_sample.png"
    img_BGR = cv2.imread(str(input_path))
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    for model_name in [x[0] for x in YOLOv10DocumentLayoutAnalysis.info]:
        model = YOLOv10DocumentLayoutAnalysis.get_model(model_name)
        model.load()
        results = model.detection(img_RGB, options={"conf": 0.2, "iou": 0.8})

        annotated_image = get_annotated_image(img_RGB, results)
        cv2.imwrite(f'result-{model_name.replace("/", "-")}.jpg', annotated_image)