
import torch
import PIL.Image
from zerollama.tasks.vlm.interface import VLMInterface
from zerollama.tasks.vlm.collection import get_model_config_by_name
from zerollama.tasks.vlm.protocol import VLMChatCompletionResponse
from zerollama.microservices.inference.transformers_green.chat import HuggingFaceTransformers, TORCH_TYPE


class CogVLM(HuggingFaceTransformers, VLMInterface):
    get_model_config_by_name = staticmethod(get_model_config_by_name)

    @torch.no_grad()
    def chat(self, messages, images, options=None):
        options = options or {}

        if len(images) == 0:
            image = None
        else:
            pil_images = [PIL.Image.fromarray(image) for image in images]
            image = pil_images[0]

        query = messages[-1]["content"]

        history = []
        for i in range(0, len(messages)-1, 2):
            history.append((messages[i]["content"], messages[i+1]["content"]))

        if image is None:
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=query,
                history=history,
                template_version='chat'
            )
        else:
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=query,
                history=history,
                images=[image],
                template_version='chat'
            )

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[input_by_model['images'][0].to(self.device).to(self.torch_dtype)]] if image is not None else None,
        }

        gen_kwargs = {
            "max_new_tokens": options.get("max_new_tokens", 2048),
            "pad_token_id": 128002,  # avoid warning of llama3
        }

        outputs = self.model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        content = self.tokenizer.decode(outputs[0])
        content = content.split("<|end_of_text|>")[0]

        return VLMChatCompletionResponse(model=self.model_name, content=content)

    @property
    def info(self):
        return self.model_info

