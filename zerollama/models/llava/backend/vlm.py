import torch
import requests
import traceback
from PIL import Image
from zerollama.tasks.vlm.interface import VLMInterface
from zerollama.tasks.vlm.collection import get_model_by_name
from zerollama.tasks.vlm.protocol import VLMChatCompletionResponse
from zerollama.tasks.base.download import get_pretrained_model_name


class Llava(VLMInterface):

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

        from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        try:
            processor = AutoProcessor.from_pretrained(self.model_name_or_path)
            model = LlavaNextForConditionalGeneration.from_pretrained(self.model_name_or_path,
                                                                      torch_dtype=torch.float16, low_cpu_mem_usage=True)
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
    def chat(self, messages, images, options=None):
        options = options or {}
        options["max_new_tokens"] = options.get("max_new_tokens", 128)

        image = images[0]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        prompt = messages[-1]["content"]

        prompt = f"USER: <image>\n {prompt} ASSISTANT:"

        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)

        output = self.model.generate(**inputs, **options)

        content = self.processor.decode(output[0], skip_special_tokens=True)

        return VLMChatCompletionResponse(model=self.model_name, content=content[len(prompt):])

    @property
    def info(self):
        return self.model_info
