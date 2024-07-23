
import torch
from zerollama.tasks.text2image.protocol import Text2ImageResponse
from zerollama.tasks.text2image.interface import Text2ImageInterface
from zerollama.tasks.text2image.collection import get_model_by_name


class Kolors(Text2ImageInterface):
    def __init__(self, model_name, local_files_only=True):
        model = get_model_by_name(model_name)
        model_config = model.get_model_config(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info

        self.pipe = None
        self.n_concurrent = 1

    def load(self):
        from zerollama.core.config.main import config_setup

        config_setup()

        from diffusers import KolorsPipeline

        self.pipe = KolorsPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            variant="fp16"
        )

        self.pipe.to("cuda")

    @torch.no_grad()
    def generate(self, prompt, negative_prompt=None, options=None):
        options = options or {}
        image = self.pipe(prompt=prompt, negative_prompt=negative_prompt, **options).images[0]

        return Text2ImageResponse(model=self.model_name, image=image)



