
import torch
import requests
import PIL.Image
from functools import partial
from zerollama.core.config.main import config_setup
from zerollama.tasks.vlm.interface import VLMInterface
from zerollama.tasks.vlm.protocol import VLMChatCompletionResponse
from zerollama.tasks.vlm.collection import get_model_config_by_name


class MiniCPMV(VLMInterface):
    def __init__(self, model_name, local_files_only=True, device="cuda"):
        model_config = get_model_config_by_name(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.device = device
        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info
        self.local_files_only = local_files_only

        self.tokenizer = None
        self.model = None
        self.vl_chat_processor = None

        self.n_concurrent = 1

    def load(self):
        config_setup()

        if self.local_files_only:
            import huggingface_hub
            huggingface_hub.snapshot_download = partial(huggingface_hub.snapshot_download,
                                                        local_files_only=True)

        from transformers import AutoModel, AutoTokenizer

        try:
            model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
            model = model.to(device=self.device, dtype=torch.bfloat16)

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model.eval()

        except requests.exceptions.HTTPError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None
        except EnvironmentError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None

        self.tokenizer = tokenizer
        self.model = model

    @torch.no_grad()
    def chat(self, messages, images, options=None):
        pil_images = [PIL.Image.fromarray(image) for image in images]

        res, context, _ = self.model.chat(
            image=pil_images[0],
            msgs=messages,
            context=None,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.7
        )

        return VLMChatCompletionResponse(model=self.model_name, content=res)

    @property
    def info(self):
        return self.model_info

