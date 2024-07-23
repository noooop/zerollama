
import torch
import requests
import traceback
from PIL import Image
from zerollama.tasks.mono_estimation.depth.protocol import DepthEstimationResponse
from zerollama.tasks.mono_estimation.depth.interface import DepthEstimationInterface
from zerollama.tasks.mono_estimation.depth.collection import get_model_by_name
from zerollama.tasks.base.download import get_pretrained_model_name


class DepthAnything(DepthEstimationInterface):
    def __init__(self, model_name, local_files_only=True, device="cuda"):
        model = get_model_by_name(model_name)
        model_config = model.get_model_config(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.device = device
        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info
        self.trust_remote_code = self.model_config.model_kwargs.get("trust_remote_code", False)
        self.model_name_or_path = get_pretrained_model_name(model_name=model_name,
                                                            local_files_only=local_files_only,
                                                            get_model_by_name=get_model_by_name)
        self.model = None
        self.processor = None
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup

        config_setup()

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        try:
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
            model.to(self.device)
        except requests.exceptions.HTTPError:
            traceback.print_exc()
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first") from None
        except EnvironmentError:
            traceback.print_exc()
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first") from None

        self.model = model
        self.processor = processor

    @torch.no_grad()
    def estimation(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)

        predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        return DepthEstimationResponse(model=self.model_name, depth=prediction.squeeze().cpu().numpy())



