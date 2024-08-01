

import torch
from threading import Thread
from zerollama.microservices.inference.transformers_green.chat import HuggingFaceTransformersChat
from zerollama.tasks.qa.protocol import ChatCompletionResponse
from zerollama.tasks.qa.protocol import ChatCompletionStreamResponse, ChatCompletionStreamResponseDone
from zerollama.tasks.qa.collection import get_model_by_name


def get_formatted_input(messages, context):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
        if item['role'] == "user":
            ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(
        ["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in
         messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + context + "\n\n" + conversation

    return formatted_input


class ChatQA(HuggingFaceTransformersChat):
    get_model_by_name = staticmethod(get_model_by_name)

    @torch.no_grad()
    def qa(self, messages, document=None, stream=False, options=None):
        options = options or dict()
        max_new_tokens = options.get("max_new_tokens", 512)

        formatted_input = get_formatted_input(messages, document)
        tokenized_prompt = self.tokenizer([self.tokenizer.bos_token + formatted_input], return_tensors="pt").to(self.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        prompt_tokens = len(tokenized_prompt.input_ids[0])

        if not stream:
            generation_kwargs = dict(tokenized_prompt, max_new_tokens=max_new_tokens, eos_token_id=terminators)
            if self.model_config.model_kwargs.get("use_generation_config", False):
                generation_kwargs["generation_config"] = self.generation_config

            generated_ids = self.model.generate(**generation_kwargs)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_prompt.input_ids, generated_ids)
            ]

            content = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            completion_tokens = len(generated_ids[0])

            return ChatCompletionResponse(**{"model": self.model_name,
                                             "content": content,
                                             "finish_reason": "stop" if completion_tokens < max_new_tokens else "length",

                                             "completion_tokens": completion_tokens,
                                             "prompt_tokens": prompt_tokens,
                                             "total_tokens": prompt_tokens+completion_tokens})
        else:
            def generator():
                generation_kwargs = dict(tokenized_prompt, streamer=self.streamer, max_new_tokens=max_new_tokens, eos_token_id=terminators)
                if self.model_config.model_kwargs.get("use_generation_config", False):
                    generation_kwargs["generation_config"] = self.generation_config

                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                completion_tokens = 0
                for content in self.streamer:
                    completion_tokens += 1

                    if not content:
                        continue

                    yield ChatCompletionStreamResponse(**{"model": self.model_name,
                                                          "delta_content": content})

                yield ChatCompletionStreamResponseDone(**{"model": self.model_name,
                                                          "finish_reason": "stop" if completion_tokens < max_new_tokens else "length",

                                                          "prompt_tokens": prompt_tokens,
                                                          "completion_tokens": completion_tokens,
                                                          "total_tokens": prompt_tokens + completion_tokens})
                thread.join()
            return generator()