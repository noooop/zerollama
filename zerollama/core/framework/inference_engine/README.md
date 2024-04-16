# 模型推理引擎
模型推理引擎启动后要把自己注册到服务中心，客户端就可以从服务中心获取到服务器地址端口(host:port)，以及能提供的服务。

## demo
```
第一个窗口
$ python -m zerollama.core.framework.inference_engine.server
ZeroNameServer: InMemoryNameServer running!
ZeroInferenceEngine:  Qwen/Qwen1.5-0.5B-Chat is running!


第二个窗口
$ python -m zerollama.core.framework.inference_engine.client
================================================================================
{'state': 'ok', 'msg': {'service_names': ['Qwen/Qwen1.5-0.5B-Chat']}}
================================================================================
stream == False
{'content': '大型语言模型是一种复杂的计算机程序，它可以生成大量的文本，例如故事、文章、诗歌等，并从中提取有价值的信息和知识。这些大模型通常使用深度学习技术，它们能够自动从大量数据中学习，从而实现更高的准确性和精度。\n'
            '\n'
            '大型语言模型可以应用于各种领域，例如自然语言处理、机器翻译、语音识别、图像分类等。它们也可以用于搜索引擎、聊天机器人、智能客服等领域。此外，大型语言模型还可以用于金融风控、医疗诊断、自动驾驶等领域。',
 'model': 'Qwen/Qwen1.5-0.5B-Chat'}
================================================================================
stream == True
{'content': '大型', 'done': False, 'model': 'Qwen/Qwen1.5-0.5B-Chat'}
{'content': '语言', 'done': False, 'model': 'Qwen/Qwen1.5-0.5B-Chat'}
{'content': '模型', 'done': False, 'model': 'Qwen/Qwen1.5-0.5B-Chat'}
...
{'content': '等', 'done': False, 'model': 'Qwen/Qwen1.5-0.5B-Chat'}
{'content': '。', 'done': False, 'model': 'Qwen/Qwen1.5-0.5B-Chat'}
{'content': '', 'done': True, 'model': 'Qwen/Qwen1.5-0.5B-Chat'}
```