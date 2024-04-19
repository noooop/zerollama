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
Wait service available
{'state': 'ok', 'msg': {'service_names': ['Qwen/Qwen1.5-0.5B-Chat']}}
================================================================================
ZeroInferenceEngine support_methods
{'state': 'ok', 'support_methods': ['inference', 'support_methods']}
================================================================================
stream == False
{'content': '大型语言模型是一种具有大量文本数据的模型，它可以理解和生成高质量的语言文本。它们通常由多种不同的语言模型组成，包括基于规则、基于神经网络和基于深度学习等。大型语言模型可以用来处理大量的自然语言输入，并且在特定任务上表现出色，例如文本分类、语义分析和机器翻译等。',
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

# 第一个窗口 Control-C 关闭 
ZeroNameServer clean_up!
ZeroInferenceEngine clean_up!
```