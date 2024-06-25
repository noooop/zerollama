
import gc
import traceback

import torch
import requests
import shortuuid
from gevent.threadpool import ThreadPoolExecutor
from zerollama.tasks.chat.interface import ChatInterface
from zerollama.tasks.chat.protocol import ChatCompletionResponse
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponse, ChatCompletionStreamResponseDone
from zerollama.tasks.chat.collection import get_model_by_name
from zerollama.tasks.base.download import get_pretrained_model_name


class VLLMChat(ChatInterface):
    get_model_by_name = staticmethod(get_model_by_name)

    def __init__(self, model_name, local_files_only=True, device="cuda", **engine_args):
        model = self.get_model_by_name(model_name)
        model_config = model.get_model_config(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.device = device
        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info
        self.local_files_only = local_files_only
        self.trust_remote_code = self.model_config.model_kwargs.get("trust_remote_code", False)
        self.pretrained_model_name = get_pretrained_model_name(model_name=model_name,
                                                               local_files_only=local_files_only,
                                                               get_model_by_name=get_model_by_name)
        self.engine_args = engine_args
        self.engine = None
        self.SamplingParams = None
        self.TextTokensPrompt = None
        self.n_concurrent = 32

    def load(self):
        from vllm import SamplingParams
        from vllm.inputs import TextTokensPrompt
        from zerollama.microservices.inference.vllm_green.llm_engine_gevent import GeventLLMEngine, GeventEngineArgs

        engine_args = GeventEngineArgs(model=self.pretrained_model_name,
                                       trust_remote_code=self.trust_remote_code,
                                       device="cuda",
                                       **self.engine_args)

        try:
            engine = GeventLLMEngine.from_engine_args(engine_args)
        except requests.exceptions.HTTPError:
            traceback.print_exc()
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first") from None
        except EnvironmentError:
            traceback.print_exc()
            raise FileNotFoundError(f"model '{self.model_name}' not found, try pulling it first") from None

        self.engine = engine
        self.SamplingParams = SamplingParams
        self.TextTokensPrompt = TextTokensPrompt

    def __del__(self):
        self.engine.terminate()
        self.SamplingParams = None
        self.TextTokensPrompt = None
        self.engine = None

        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def info(self):
        return self.model_info

    def chat(self, messages, stream=False, options=None):
        options = options or {}
        skip_empty_delta_text = options.pop("skip_empty_delta_text", True)
        request_id = f"{shortuuid.random(length=22)}"
        sampling_params = self.SamplingParams(**options)

        def encode(messages):
            tokenizer = self.engine.get_tokenizer()
            prompt = tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompt_token_ids = tokenizer.encode(prompt)
            inputs = self.TextTokensPrompt(prompt=prompt, prompt_token_ids=prompt_token_ids)
            return inputs

        with ThreadPoolExecutor(1) as executor:
            f = executor.submit(encode, messages)
            inputs = f.result()

        result_generator = self.engine.generate(inputs=inputs,
                                                sampling_params=sampling_params,
                                                request_id=request_id)

        if not stream:
            final_res = None
            for res in result_generator:
                final_res = res

            output = final_res.outputs[0]

            num_prompt_tokens = len(final_res.prompt_token_ids)
            num_generated_tokens = len(output.token_ids)

            return ChatCompletionResponse(**{"model": self.model_name,
                                             "content": output.text,
                                             "finish_reason": output.finish_reason,

                                             "completion_tokens": num_generated_tokens,
                                             "prompt_tokens": num_prompt_tokens,
                                             "total_tokens": num_prompt_tokens + num_generated_tokens})
        else:
            def generator():
                previous_texts = ""
                prompt_tokens = 0
                completion_tokens = 0
                finish_reason = None

                for res in result_generator:
                    output = res.outputs[0]

                    delta_text = output.text[len(previous_texts):]
                    previous_texts = output.text
                    finish_reason = output.finish_reason
                    prompt_tokens = len(res.prompt_token_ids)
                    completion_tokens = len(output.token_ids)

                    if not delta_text and skip_empty_delta_text:
                        continue

                    yield ChatCompletionStreamResponse(**{"model": self.model_name,
                                                          "delta_content": delta_text})

                yield ChatCompletionStreamResponseDone(**{"model": self.model_name,
                                                          "finish_reason": finish_reason,

                                                          "prompt_tokens": prompt_tokens,
                                                          "completion_tokens": completion_tokens,
                                                          "total_tokens": prompt_tokens + completion_tokens})
            return generator()


def run_test(model_name, stream=False, **kwargs):
    import inspect
    print("=" * 80)

    model = VLLMChat(model_name, local_files_only=False, **kwargs)
    model.load()
    print(model.info)

    prompt = "给我介绍一下大型语言模型。"

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = model.chat(messages, options={"max_tokens": 512}, stream=stream)

    if inspect.isgenerator(response):
        for part in response:
            if isinstance(part, ChatCompletionStreamResponseDone):
                print()
                print("completion_tokens:", part.completion_tokens)
            else:
                print(part.delta_content, end="", flush=True)
    else:
        print(response.content)
        print("completion_tokens:", response.completion_tokens)


if __name__ == '__main__':
    model_name = "Qwen/Qwen2-0.5B-Instruct"

    run_test(model_name, stream=False)
    run_test(model_name, stream=True)