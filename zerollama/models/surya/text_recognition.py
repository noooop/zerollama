
from zerollama.tasks.ocr.text_recognition.interface import TextRecognition


class SuryaTextRecognition(TextRecognition):
    family = "surya_tr"
    header = ["name"]
    info = [
        ["surya_tr"]
    ]
    inference_backend = "zerollama.models.surya.backend.text_recognition:SuryaTextRecognition"


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.models.surya.text_line_detection import SuryaTextLineDetection

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/dla"

    input_path = dla_test_path / "input_sample.png"
    image = Image.open(input_path)

    tld = SuryaTextLineDetection.get_model("surya_tld")
    tld.load()
    results = tld.detection(image)

    for model_name in [x[0] for x in SuryaTextRecognition.info]:
        model = SuryaTextRecognition.get_model(model_name)
        model.load()
        results = model.recognition(image, ["zh", "en"], [x.bbox for x in results.bboxes])
        print(results)

