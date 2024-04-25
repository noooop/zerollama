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
[ServerInfo(name='Qwen/Qwen1.5-0.5B-Chat', host='localhost', port=51335, protocol='chat')]
================================================================================
ZeroInferenceEngine support_methods
state='ok' msg={'support_methods': ['inference', 'support_methods']}
================================================================================
stream == False
大型语言模型是指具有大量训练数据的深度学习模型，它能够处理和理解复杂的自然语言任务，包括生成文本、问答、对话系统等。

它们可以使用各种不同的技术来实现，例如卷积神经网络（CNN）、循环神经网络（RNN）、长短期记忆网络（LSTM）等。这些模型在处理复杂自然语言问题时通常可以比传统的机器学习模型更快、更准确地完成任务。

常见的大型语言模型有：

1. spaCy：这是一个开源的预训练深度学习模型，用于处理中文文本。它包含了大量的语料库，并且支持多种语言。

2. Gensim：一个基于深度强化学习的自然语言处理模型，它可以理解和生成人类写作风格的高质量文本。

3. Word2Vec：这是一个基于词向量的文本分类和识别模型，它可以从大规模的数据中提取出词语的表示模式，从而帮助我们理解文本的意义。

4. BERT：这是Bert的升级版本，它的学习能力更强，可以在大量的语言数据上进行训练。

5. Yann LeCun：这是LeCun教授开发的一种深度学习方法，可以帮助我们建立一个高效的自动问答系统。

以上只是一些大型语言模型的例子，实际上还有很多其他的大规模语言模型也在不断出现和发展。
response_length: 267
================================================================================
stream == True
ZeroServerResponse(state='ok', msg=ChatCompletionStreamResponse(model='Qwen/Qwen1.5-0.5B-Chat', prompt_length=24, response_length=1, finish_reason=None, content='大型', done=False))
ZeroServerResponse(state='ok', msg=ChatCompletionStreamResponse(model='Qwen/Qwen1.5-0.5B-Chat', prompt_length=24, response_length=2, finish_reason=None, content='语言', done=False))
ZeroServerResponse(state='ok', msg=ChatCompletionStreamResponse(model='Qwen/Qwen1.5-0.5B-Chat', prompt_length=24, response_length=3, finish_reason=None, content='模型', done=False))
...
ZeroServerResponse(state='ok', msg=ChatCompletionStreamResponse(model='Qwen/Qwen1.5-0.5B-Chat', prompt_length=24, response_length=152, finish_reason=None, content='利用', done=False))
ZeroServerResponse(state='ok', msg=ChatCompletionStreamResponse(model='Qwen/Qwen1.5-0.5B-Chat', prompt_length=24, response_length=153, finish_reason=None, content='。', done=False))
ZeroServerResponse(state='ok', msg=ChatCompletionStreamResponse(model='Qwen/Qwen1.5-0.5B-Chat', prompt_length=24, response_length=153, finish_reason='stop', content=None, done=True))

# 第一个窗口 Control-C 关闭 
ZeroNameServer clean_up!
ZeroInferenceEngine clean_up!
```