# 流式推理

这个功能用来支持流式推理，生成一个字，输出一个字，明显能提高用户体验

## 推理后端需要支持流式推理
```
# Transformers Streaming Mode
# Starting here, we add streamer for text generation.
from transformers import TextStreamer
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# This will print the output in the streaming mode.
generated_ids = model.generate(
    model_inputs,
    max_new_tokens=512,
    streamer=streamer,
)
```

## 服务框架需要支持流式传输
```
第一个窗口
$ python server.py

第二个窗口
$ python client.py 
{'model': 'Qwen/Qwen1.5-0.5B-Chat', 'created_at': '2024-04-14T16:21:08', 'message': {'role': 'assistant', 'content': '大型', 'images': None}, 'done': False}
{'model': 'Qwen/Qwen1.5-0.5B-Chat', 'created_at': '2024-04-14T16:21:08', 'message': {'role': 'assistant', 'content': '语言', 'images': None}, 'done': False}
{'model': 'Qwen/Qwen1.5-0.5B-Chat', 'created_at': '2024-04-14T16:21:08', 'message': {'role': 'assistant', 'content': '模型', 'images': None}, 'done': False}
...
{'model': 'Qwen/Qwen1.5-0.5B-Chat', 'created_at': '2024-04-14T16:21:13', 'message': {'role': 'assistant', 'content': '能够', 'images': None}, 'done': False}
{'model': 'Qwen/Qwen1.5-0.5B-Chat', 'created_at': '2024-04-14T16:21:13', 'message': {'role': 'assistant', 'content': '', 'images': None}, 'done': False}
{'model': 'Qwen/Qwen1.5-0.5B-Chat', 'created_at': '2024-04-14T16:21:13', 'done': True, 'total_duration': 0, 'load_duration': 0, 'prompt_eval_count': 0, 'prompt_eval_duration': 0, 'eval_count': 0, 'eval_duration': 0}
```
Amazing

# Reference
https://qwen.readthedocs.io/en/latest/inference/chat.html#streaming-mode




