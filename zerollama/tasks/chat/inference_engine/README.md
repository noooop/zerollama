# 模型推理引擎
模型推理引擎启动后要把自己注册到服务中心，客户端就可以从服务中心获取到服务器地址端口(host:port)，以及能提供的服务。

## demo
```
第一个窗口
$ python -m zerollama.tasks.chat.inference_engine.server
ZeroNameServer: InMemoryNameServer running! port: 9527
ZeroInferenceEngine:  Qwen/Qwen1.5-0.5B-Chat is running! port: 59296

第二个窗口
$ python -m zerollama.tasks.chat.inference_engine.client
================================================================================
Wait Qwen/Qwen1.5-0.5B-Chat available
[ServerInfo(name='Qwen/Qwen1.5-0.5B-Chat', host='localhost', port=55258, protocol='chat')]
================================================================================
ZeroInferenceEngine support_methods
state='ok' msg={'support_methods': ['inference', 'info', 'support_methods']} payload=[]
state='ok' msg={'name': 'Qwen/Qwen1.5-0.5B-Chat', 'size': '0.5B', 'quantization': '', 'bits': '', 'family': 'Qwen1.5', 'protocol': 'chat'} payload=[]
================================================================================
stream == False
大型语言模型通常是指能够理解大量文本，具有较高准确率和泛化能力的语言模型，它们可以用于回答问题、生成代码、聊天等任务。大型语言模型通常包括深度学习技术、预训练模型、半监督学习模型、无监督学习模型等多种技术。
response_length: 60
================================================================================
stream == True
ZeroServerResponse(state='ok', msg=ChatCompletionStreamResponse(model='Qwen/Qwen1.5-0.5B-Chat', prompt_length=24, response_length=1, finish_reason=None, content='大型', done=False), payload=[])
ZeroServerResponse(state='ok', msg=ChatCompletionStreamResponse(model='Qwen/Qwen1.5-0.5B-Chat', prompt_length=24, response_length=2, finish_reason=None, content='语言', done=False), payload=[])
ZeroServerResponse(state='ok', msg=ChatCompletionStreamResponse(model='Qwen/Qwen1.5-0.5B-Chat', prompt_length=24, response_length=3, finish_reason=None, content='模型', done=False), payload=[])...
...
ZeroServerResponse(state='ok', msg=ChatCompletionStreamResponse(model='Qwen/Qwen1.5-0.5B-Chat', prompt_length=24, response_length=63, finish_reason=None, content='等', done=False), payload=[])
ZeroServerResponse(state='ok', msg=ChatCompletionStreamResponse(model='Qwen/Qwen1.5-0.5B-Chat', prompt_length=24, response_length=64, finish_reason=None, content='。', done=False), payload=[])
ZeroServerResponse(state='ok', msg=ChatCompletionStreamResponse(model='Qwen/Qwen1.5-0.5B-Chat', prompt_length=24, response_length=64, finish_reason='stop', content=None, done=True), payload=[])

# 第一个窗口 Control-C 关闭 
ZeroNameServer clean_up!
ZeroInferenceEngine clean_up!
```