# 模型
测试使用qwen1.5系列模型，包括8种型号尺寸，包括0.5B、1.8B、4B、7B、14B、32B和72B。

很方便体验到随着模型大小的增加，速度变化和模型能力变化

# 推理后端
huggingface transformers，最简单也兼容性最强

```
$ python model.py
{'finish_reason': 'stop',
 'prompt_length': 24,
 'response_length': 69,
 'response_text': '大型语言模型是一种机器学习技术，它可以模拟人类智能，并从中提取有用的信息。它们可以用于识别文本、语音、图像和视频等类型的数据，并可以实现自动翻译、文本摘要、情感分析、问答系统等功能。大型语言模型的开发需要大量的计算资源，因此它们在处理大规模数据时效率非常高。'}
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
{'created_at': '2024-04-14T16:14:24',
 'done': True,
 'message': {'content': '大型语言模型是指能够处理大规模、复杂的信息，包括文本、语音、图像和视频等数据的机器学习模型。它们可以用于各种领域，例如自然语言处理、计算机视觉、对话系统、智能客服、搜索引擎优化、自动驾驶等。这些大型语言模型通常使用深度学习技术，通过大量的训练数据来建立复杂的神经网络架构，并利用大量的训练数据来提高模型的准确性。',
             'images': None,
             'role': 'assistant'},
 'model': 'Qwen/Qwen1.5-0.5B-Chat'}
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
  "model": "Qwen/Qwen1.5-0.5B-Chat",
  "messages": [
    {
      "role": "user",
      "content": "why is the sky blue?"
    }
  ]
}'

{
  "created_at": "2024-04-14T16:16:05",
  "done": true,
  "message": {
    "content": "The color of the sky varies depending on several factors, including the distance from the sun and its position in relation to the Earth. The color of the sky can also be influenced by factors such as humidity, temperature, and wind patterns.\nIn general, the sky appears blue when it is near the horizon because the sun is located at a higher altitude than the earth. This creates a layer of high energy light that passes through the Earth's atmosphere and reaches the surface of the planet.\nHowever, the color of the sky can change over time due to changes in the Earth's orbit around the Sun and other celestial bodies. As the sun moves closer to the Earth, more sunlight is scattered and the color of the sky becomes more yellow and orange instead of blue. This phenomenon is known as \"sky pollution.\"\nDespite these variations, the primary reason why the sky appears blue is because of the presence of sunlight that enters the Earth's atmosphere at sunrise and sunset. The blue\u5149\u7ebf is absorbed by the Earth's atmosphere and diffused out into space, creating the blue appearance we see in the sky.",
    "images": null,
    "role": "assistant"
  },
  "model": "Qwen/Qwen1.5-0.5B-Chat"
}


OpenAI's Chat Completions API
第二个窗口
$ curl http://localhost:11434/v1/chat/completions -d '{
  "model": "Qwen/Qwen1.5-0.5B-Chat",
  "messages": [
    {
      "role": "user",
      "content": "why is the sky blue?"
    }
  ]
}'

{
  "created_at": "2024-04-14T16:17:11",
  "done": true,
  "message": {
    "content": "The sky appears blue because of a combination of two colors: blue light and white light. Blue light travels much faster than other colors, so it reaches our eyes first. When we look at the sky, the blue light from the sun shines through the atmosphere, creating a colorful effect.\nHowever, when we look directly into the sun, the blue light is blocked by the Earth's atmosphere, causing it to scatter or disperse. This causes the blue color to appear in shades of green, red, and orange. In addition, some blue light may be absorbed by the Earth's clouds, which can also affect the appearance of the sky.\nIt's important to note that the way we perceive colors depends on how we use them. By paying attention to where the sun is shining and how it's reflected off the atmosphere, you can better understand why the sky appears blue.",
    "images": null,
    "role": "assistant"
  },
  "model": "Qwen/Qwen1.5-0.5B-Chat"
}
```

# 后续
1. huggingface transformers 是事实上的标准，大量的模型在上面发布，但推理的效率明显很低
2. 支持流式推理，生成一个字，输出一个字，明显能提高用户体验

# Reference
1. https://github.com/QwenLM/Qwen1.5
2. https://huggingface.co/Qwen
3. https://hf-mirror.com


