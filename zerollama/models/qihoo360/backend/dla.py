
import os
import PIL.Image
from pathlib import Path
from zerollama.tasks.dla.interface import DLAInterface
from zerollama.tasks.dla.collection import get_model_by_name

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
weight_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent.parent / "models/360LayoutAnalysis"


class LayoutAnalysis360(DLAInterface):

    def __init__(self, model_name):
        model = get_model_by_name(model_name)
        model_config = model.get_model_config(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info

        self.model = None
        self.weight_dtype = None
        self.n_concurrent = 1

    def load(self):
        from ultralytics import YOLO
        self.model = YOLO(str(weight_path / self.model_name.split("/")[-1]))

    def detection(self, image, options=None):
        options = options or {}
        return self.model(source=PIL.Image.fromarray(image), **options)[0].summary()



