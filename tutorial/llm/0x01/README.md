# 模型
测试使用qwen1.5系列模型，包括8种型号尺寸，包括0.5B、1.8B、4B、7B、14B、32B和72B。

很方便体验到随着模型大小的增加，速度变化和模型能力变化

# 推理后端
huggingface transformers，最简单也兼容性最强

```
$ python model.py
{'finish_reason': 'stop',
 'prompt_length': 24,
 'response_length': 95,
 'response_text': '大型语言模型是一种能够处理大规模数据并进行自然语言处理的计算机程序。它们通常使用深度学习技术，包括卷积神经网络（CNN）、循环神经网络（RNN）和长短时记忆网络（LSTM）等。这些模型通常能够处理大量的文本数据，并能够提取文本中的重要信息和结构。大型语言模型在自然语言处理领域中具有广泛的应用，例如文本分类、情感分析、机器翻译、问答系统等。'}
```


> ## 注意
> 下载模型速度慢，可以使用代理 
> 
> 打开os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 注释
> 
> 默认模型下载到c盘，如果空间紧张
> 
> 打开 os.environ['HF_HOME'] = 'D:/.cache/' 注释


# 服务端和客户端

```
第一个窗口
$ python server.py

第二个窗口
$ python client.py 
{'created_at': '2024-04-11T15:30:44',
 'done': True,
 'message': {'content': '大型语言模型是一种能够处理大规模数据并进行自然语言处理的计算机程序。它们通常使用深度学习技术，包括卷积神经网络（CNN）、循环神经网络（RNN）和长短时记忆网络（LSTM）等。这些模型通常能够处理大量的文本数据，并能够提取文本中的重要信息和结构。大型语言模型在自然语言处理领域中具有广泛的应用，例如文本分类、情感分析、机器翻译、问答系统等。',
             'images': None,
             'role': 'assistant'},
 'model': 'Qwen/Qwen1.5-0.5B'}
```

# http 网关

启动 main.py 可以把 server.py 和 http_gateway.py 都拉起来

这就有了一个最小的兼容openai chat api的服务，Amazing


```

第一个窗口
$ python main.py 

ollama Chat Completions API
第二个窗口
$ curl http://localhost:11434/api/chat -d '{
  "model": "Qwen/Qwen1.5-0.5B",
  "messages": [
    {
      "role": "user",
      "content": "why is the sky blue?"
    }
  ]
}'
{
  "created_at": "2024-04-11T16:50:07",
  "done": true,
  "message": {
    "content": "The sky is blue because it is made up of a mixture of different gases, such as nitrogen, oxygen, and water vapor. These gases are constantly moving around and interacting with each other, creating a blue color. The blue color is caused by the absorption of light by the different gases, which causes the light to be split into different colors.",
    "images": null,
    "role": "assistant"
  },
  "model": "Qwen/Qwen1.5-0.5B"
}


OpenAI's Chat Completions API
第二个窗口
$ curl http://localhost:11434/v1/chat/completions -d '{
  "model": "Qwen/Qwen1.5-0.5B",
  "messages": [
    {
      "role": "user",
      "content": "why is the sky blue?"
    }
  ]
}'
{
  "created_at": "2024-04-11T17:10:33",
  "done": true,
  "message": {
    "content": "The sky is blue because it is made up of a mixture of different gases, such as nitrogen, oxygen, and water vapor. These gases are constantly moving around and interacting with each other, creating a blue color. The blue color is caused by the absorption of light by the different gases, which causes the light to be split into different colors.",
    "images": null,
    "role": "assistant"
  },
  "model": "Qwen/Qwen1.5-0.5B"
}
```

# 后续
1. huggingface transformers 是事实上的标准，大量的模型在上面发布，但推理的效率明显很低
2. 支持流式推理，生成一个字，输出一个字，明显能提高用户体验

# Reference
https://github.com/QwenLM/Qwen1.5
https://huggingface.co/Qwen
https://hf-mirror.com


