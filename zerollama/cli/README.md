# 客户端

## pull 下载模型
```
$ python -m zerollama.cli pull Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4
```

## run 运行模型
```
$ python -m zerollama.cli run Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4
正在加载模型...
ZeroNameServer: InMemoryNameServer running!
ZeroInferenceEngine:  Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4 is running!
加载完成!
[对话第1轮]
(用户输入:)
hello
(Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4:)

Hello! How can I assist you today? Is there a specific topic or question you'd like to discuss? I'm here to help with any general knowledge, provide information on various topics, answer your questions, or even engage in friendly conversation. Just let me know and I'll do my best to assist you.

[对话第2轮]
(用户输入:)
你会说中文吗？
(Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4:)

是的，我可以说中文。我具有普通话（中国官方标准汉语）的发音和词汇能力，可以理解和回答与中文相关的问题。我可以提供有关汉字、语法、文化
、历史等方面的信息，并帮助你解决日常生活中的中文疑问。如果你有任何问题或需要进一步解释某个概念，请随时告诉我，我会尽我所能提供帮助。

[对话第3轮]
(用户输入:)
用中文讲个笑话吧
(Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4:)

为什么鸭子不会说英语？
因为它们只会“嘎嘎”和“咕咕”。

[对话第4轮]
(用户输入:)
...
```