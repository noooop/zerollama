from zerollama.core.config.main import config_setup

config_setup()

model_name = "Qwen/Qwen1.5-0.5B-Chat"


from mii import pipeline
inference_pipeline = pipeline(model_name_or_path=model_name)

prompt = "给我介绍一下大型语言模型。"

messages = [
    {"role": "user", "content": prompt}
]

prompt = inference_pipeline.tokenizer.tokenizer.apply_chat_template(
    conversation=messages,
    tokenize=False,
    add_generation_prompt=True,
)

tokenizer = inference_pipeline.tokenizer

print(tokenizer.eos_token_id)
print(tokenizer.tokenizer.eos_token_id)

out = inference_pipeline(prompts=[prompt], max_new_tokens=5000)
print(out)

"""
https://github.com/microsoft/DeepSpeed/pull/5219
https://github.com/microsoft/DeepSpeed-MII/pull/408
https://github.com/microsoft/DeepSpeed-MII/issues/442
https://github.com/microsoft/DeepSpeed-MII/issues/457
https://github.com/microsoft/DeepSpeed/pull/5403
"""