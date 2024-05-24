
import torch
import requests
import PIL.Image
from functools import partial
from zerollama.core.config.main import config_setup
from zerollama.tasks.vlm.interface import VLMInterface
from zerollama.tasks.vlm.protocol import VLMChatCompletionResponse
from zerollama.tasks.vlm.collection import get_model_config_by_name


class DeepseekVL(VLMInterface):
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

        from transformers import AutoModelForCausalLM
        from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM


        try:
            vl_chat_processor = VLChatProcessor.from_pretrained(self.model_name)
            tokenizer = vl_chat_processor.tokenizer

            model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
            model = model.to(torch.bfloat16).to(self.device).eval()
        except requests.exceptions.HTTPError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None
        except EnvironmentError:
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first.") from None

        self.tokenizer = tokenizer
        self.model = model
        self.vl_chat_processor = vl_chat_processor

    @torch.no_grad()
    def chat(self, messages, images, options=None):
        conversations = []
        for message in messages:
            conversations.append({"role": message["role"].capitalize(),
                                  "content": "<image_placeholder>"+message["content"]})

        pil_images = [PIL.Image.fromarray(image) for image in images]
        prepare_inputs = self.vl_chat_processor(
            conversations=conversations,
            images=pil_images,
            force_batchify=True
        ).to(self.model.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return VLMChatCompletionResponse(model=self.model_name, content=answer)

    @property
    def info(self):
        return self.model_info

