import time
import pickle
from gevent.pool import Pool
from pathlib import Path
from tqdm import trange
from zerollama.microservices.standalone.server import setup, run
from zerollama.core.framework.zero_manager.client import ZeroManagerClient
from zerollama.tasks.chat.protocol import Chat_ENGINE_CLASS
from zerollama.tasks.chat.engine.client import ChatClient
from zerollama.tasks.chat.protocol import ChatCompletionStreamResponseDone

server = setup()
run(server, waiting=False)

Root_MANAGER_NAME = "RootZeroManager"
manager_client = ZeroManagerClient(Root_MANAGER_NAME)
manager_client.wait_service_available(Root_MANAGER_NAME)

gptq_info = [
    # GPTQ-Int4
    ["Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",     "0.5B",   "GPTQ",                        "4bits"],
    ["Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4",     "1.8B",   "GPTQ",                        "4bits"],
    ["Qwen/Qwen1.5-4B-Chat-GPTQ-Int4",       "4B",     "GPTQ",                        "4bits"],
    ["Qwen/Qwen1.5-7B-Chat-GPTQ-Int4",       "7B",     "GPTQ",                        "4bits"],
    ["Qwen/Qwen1.5-14B-Chat-GPTQ-Int4",      "14B",    "GPTQ",                        "4bits"],
    ["Qwen/Qwen1.5-32B-Chat-GPTQ-Int4",      "32B",    "GPTQ",                        "4bits"],
]

for model_name, *_ in gptq_info:
    print(model_name)
    manager_client.start(name=model_name,
                         engine_kwargs={"server_class": Chat_ENGINE_CLASS,
                                        "engine_kwargs": {
                                            "inference_backend": "zerollama.microservices.inference.vllm_green.chat:VLLMChat",
                                            "device": "cuda",
                                            "disable_log_stats": True,
                                            "max_model_len": 1024,
                                            #"enable_chunked_prefill": True,
                                            #"max_num_batched_tokens": 64,
                                            #"max_num_seqs": 32
                                        }})

    client = ChatClient()
    client.wait_service_available(model_name)

    start_time = time.time()


    def worker(request_id):
        prompt = "给我介绍一下大型语言模型。"

        messages = [
            {"role": "user", "content": prompt}
        ]
        time_list = [start_time, time.time()]
        for output in client.stream_chat(model_name, messages, options={"ignore_eos": True, "max_tokens": 1000, "skip_empty_delta_text": False}):
            if not isinstance(output, ChatCompletionStreamResponseDone):
                time_list.append(time.time())

        return request_id, time_list


    for n in [1, 2, 4, 8, 16, 32, 64]:
        p = Pool(n)
        data = []
        for x in p.imap_unordered(worker, trange(min(128, n*4))):
            data.append(x)

        Path(f"./vllm/{n}/").mkdir(exist_ok=True, parents=True)
        pickle.dump(data, open(f"./vllm/{n}/{model_name.replace('/', '_')}.pkl", "wb"))

    manager_client.terminate(model_name)
    time.sleep(10)

for h in server:
    h.terminate()
