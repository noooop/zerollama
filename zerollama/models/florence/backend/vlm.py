import torch
import requests
import PIL.Image
from zerollama.tasks.vlm.interface import VLMInterface
from zerollama.tasks.vlm.protocol import VLMChatCompletionResponse
from zerollama.tasks.vlm.collection import get_model_config_by_name, get_model_by_name
from zerollama.tasks.base.download import get_pretrained_model_name


class Florence(VLMInterface):
    def __init__(self, model_name, local_files_only=True, device="cuda"):
        model_config = get_model_config_by_name(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.device = device
        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info
        self.local_files_only = local_files_only
        self.pretrained_model_name = get_pretrained_model_name(model_name=model_name,
                                                               local_files_only=local_files_only,
                                                               get_model_by_name=get_model_by_name)

        self.processor = None
        self.model = None
        self.vl_chat_processor = None

        self.n_concurrent = 1

    def load(self):
        from transformers import AutoProcessor, AutoModelForCausalLM

        try:
            model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name,
                                                         trust_remote_code=True,
                                                         local_files_only=self.local_files_only)
            processor = AutoProcessor.from_pretrained(self.pretrained_model_name,
                                                      trust_remote_code=True,
                                                      local_files_only=self.local_files_only)
            model.eval()

        except requests.exceptions.HTTPError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None
        except EnvironmentError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None

        self.processor = processor
        self.model = model

    @torch.no_grad()
    def chat(self, messages, images, options=None):
        pil_images = [PIL.Image.fromarray(image) for image in images]

        image = pil_images[0]
        text_input = messages[-1].get("content", None)
        task_prompt = messages[-1]["task"]

        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.processor.post_process_generation(generated_text,
                                                               task=task_prompt,
                                                               image_size=(image.width, image.height))

        return VLMChatCompletionResponse(model=self.model_name, data=parsed_answer)

    @property
    def info(self):
        return self.model_info
