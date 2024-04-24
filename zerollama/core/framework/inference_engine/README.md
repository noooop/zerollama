# 模型推理引擎
模型推理引擎启动后要把自己注册到服务中心，客户端就可以从服务中心获取到服务器地址端口(host:port)，以及能提供的服务。

## demo
```
第一个窗口
$ python -m zerollama.core.framework.inference_engine.server
ZeroNameServer: InMemoryNameServer running! port: 9527
ZeroInferenceEngine:  Qwen/Qwen1.5-0.5B-Chat is running! port: 59296


第二个窗口
$ python -m zerollama.core.framework.inference_engine.client
================================================================================
Wait Qwen/Qwen1.5-0.5B-Chat available
[{'name': 'Qwen/Qwen1.5-0.5B-Chat', 'host': 'localhost', 'port': 57220, 'protocol': 'chat'}]
================================================================================
ZeroInferenceEngine support_methods
{'state': 'ok', 'support_methods': ['inference', 'support_methods']}
================================================================================
stream == False
{'content': '大型语言模型是一种强大的计算模型，它可以理解和生成人类可以使用的文本，并且能够处理大量的数据。它们通常由多层神经网络组成，这些神经网络可以模拟人类的自然语言处理能力。\n'
            '\n'
            '在大型语言模型中，训练集是用于训练和优化模型的大量数据。训练集中的数据通常包含已知的句子，模型需要学习如何从这些句子中提取有用的特征，以产生有效的输出。模型会通过不断迭代和调整来提高其性能。\n'
            '\n'
            '大型语言模型在许多领域都有应用，包括但不限于语音识别、机器翻译、聊天机器人、自动问答系统等。它们也可以用于预测和分析复杂的模式，帮助人们更好地理解复杂的信息。',
 'finish_reason': 'stop',
 'model': 'Qwen/Qwen1.5-0.5B-Chat',
 'prompt_length': 24,
 'response_length': 143}
================================================================================
stream == True
{'content': '大型', 'done': False, 'model': 'Qwen/Qwen1.5-0.5B-Chat'}
{'content': '语言', 'done': False, 'model': 'Qwen/Qwen1.5-0.5B-Chat'}
{'content': '模型', 'done': False, 'model': 'Qwen/Qwen1.5-0.5B-Chat'}
...
{'content': '', 'done': False, 'model': 'Qwen/Qwen1.5-0.5B-Chat'}
{'content': '', 'done': False, 'model': 'Qwen/Qwen1.5-0.5B-Chat'}
{'content': '。', 'done': False, 'model': 'Qwen/Qwen1.5-0.5B-Chat'}
{'done': True,
 'finish_reason': 'stop',
 'model': 'Qwen/Qwen1.5-0.5B-Chat',
 'prompt_length': 24,
 'response_length': 172}

# 第一个窗口 Control-C 关闭 
ZeroNameServer clean_up!
ZeroInferenceEngine clean_up!
```