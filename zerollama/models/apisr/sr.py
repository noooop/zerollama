
from zerollama.tasks.super_resolution.interface import SuperResolutionModel


class APISR(SuperResolutionModel):
    family = "APISR"
    header = ["name", "scale", "weight_path"]
    info = [
        # name        scale    weight_path
        ["4xGRL",     4,       "4x_APISR_GRL_GAN_generator.pth"],
        ["4xDAT",     4,       "4x_APISR_DAT_GAN_generator.pth"],
        ["4xRRDB",    4,       "4x_APISR_RRDB_GAN_generator.pth"],
        ["2xRRDB",    2,       "2x_APISR_RRDB_GAN_generator.pth"],
    ]
    inference_backend = "zerollama.models.apisr.backend.sr:APISR"


if __name__ == '__main__':
    import os
    import cv2
    from PIL import Image
    from pathlib import Path

    def get_model(model_name):
        model_class = APISR.inference_backend
        module_name, class_name = model_class.split(":")
        import importlib

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        model = model_class(model_name=model_name)
        model.load()
        return model


    vlm_test_path = Path(os.path.dirname(__file__)).parent.parent.parent / "static/test_sample/vlm"

    input_path = vlm_test_path / "monday.jpg"
    img_lr = cv2.imread(str(input_path))
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)

    for model_name in [x[0] for x in APISR.info]:
        model = get_model(model_name)
        super_resolved_img = model.sr(img_lr)

        im = Image.fromarray(super_resolved_img.image)
        im.save(f"{model_name}-super_resolved_img.png", format="png")