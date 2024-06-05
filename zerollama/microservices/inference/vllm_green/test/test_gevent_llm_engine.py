
import time
from zerollama.core.config.main import config_setup
config_setup()

from gevent.pool import Pool
import shortuuid
from vllm import SamplingParams
from zerollama.microservices.inference.vllm_green.llm_engine_gevent import GeventLLMEngine, GeventEngineArgs

model_name = "Qwen/Qwen1.5-0.5B-Chat"

engine_args = GeventEngineArgs(model=model_name, device="cuda")

engine = GeventLLMEngine.from_engine_args(engine_args)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def worker(prompt):
    request_id = f"{shortuuid.random(length=22)}"
    outputs = engine.generate(inputs=prompt,
                              sampling_params=sampling_params,
                              request_id=request_id)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"ID:{request_id} Prompt: {prompt!r}, Generated text: {generated_text!r}")


print("="*10)

p = Pool(2)
for x in p.imap_unordered(worker, prompts):
    pass

time.sleep(10)

p = Pool(2)
for x in p.imap_unordered(worker, prompts):
    pass
