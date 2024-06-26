
from zerollama.tasks.ocr.text_recognition.interface import TextRecognitionModel


class DeepDocTextRecognition(TextRecognitionModel):
    family = "deepdoc_tr"
    header = ["name"]
    info = [
        ["deepdoc_tr"]
    ]
    inference_backend = "zerollama.models.ragflow.backend.text_recognition:DeepDocTextRecognition"


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path
    from zerollama.models.ragflow.text_line_detection import DeepDocTextLineDetection
    from zerollama.tasks.ocr.text_recognition.utils import get_annotated_image

    dla_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/dla"

    input_path = dla_test_path / "input_sample.png"
    image = Image.open(input_path)

    tld = DeepDocTextLineDetection.get_model("deepdoc_tld")
    tld.load()
    lines = tld.detection(image)

    langs = ["zh", "en"]

    for model_name in [x[0] for x in DeepDocTextRecognition.info]:
        model = DeepDocTextRecognition.get_model(model_name)
        model.load()
        results = model.recognition(image, langs, lines)

        annotated_image = get_annotated_image(image, results, langs)
        annotated_image.save(f'result-{model_name.replace("/", "-")}.jpg')

