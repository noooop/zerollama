# 客户端

## pull 下载模型
```
$ python -m applications.chat_cli pull Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4
```

## run 运行模型
```
$ python -m applications.chat_cli run Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4
ZeroNameServer: InMemoryNameServer running! port: 57764
正在加载模型...
ZeroInferenceEngine:  Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4 is running! port: 58601
加载完成!
!quit 退出, !next 开启新一轮对话。玩的开心！
================================================================================
[对话第1轮]
(用户输入:)
hello
(Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4:)

Hello! How can I help you today? Is there something specific you'd like to know or discuss? I'm here to answer any questions you may have on a wide range of topics, from general knowledge and current events to more technical inquiries. Whether it's about history, science, technology, arts, culture, or anything else in between, I'll do my best to provide relevant information and engage in engaging discussions. So feel free to ask me anything, and I'll be happy to assist you!

[对话第2轮]
(用户输入:)
你会说中文吗？
(Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4:)

是的，我会说中文。我的母语是汉语（普通话），也是世界上使用人数最多的语言之一，全球有超过7亿人会讲中文。除此之外，我还可以说一些常用的汉字和词汇，包括但不限于“你好”、“谢谢”、“对不起”、“再见”、“今天”、“明天”、“昨天”等基本用语。如果你对某些特定的词语或句子有疑问，或者 需要在日常生活中使用中文，请随时告诉我，我将尽我所能为你提供准确的翻译和解释。

[对话第3轮]
(用户输入:)
用中文讲个笑话吧
(Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4:)

有一次，一只鱼在河里游着，突然看到前方有一条大鲨鱼正向它逼近。鱼非常害怕，于是它鼓足勇气游了过去。
可是还没来得及避开，就听到了“扑通”一声。原来那条大鲨鱼已经跃出了水面，并开始追咬起鱼来。鱼吓坏了，但又不甘心就这样被鲨鱼吃掉，于是它 奋力向上一跳，准备再次跳入水中。
结果就在这时，它的脚底不小心踩到了一个硬物，导致它在空中弹了一下，然后重重地摔在地上。鱼一看情况不妙，赶紧爬起来，但它发现自己的脚已 经被扎破了，无法再继续向前。
这时，旁边刚好有人经过，他看见了这个场景，不禁哈哈大笑起来：“这下可惨了！你看，这只鱼竟然被自己的脚给伤了！”
鱼听了这句话，不禁哭了起来：“谁让我这么笨呢，竟会在水面上踩到什么东西啊。”
听到鱼的话，岸上的人都忍不住笑了起来，他们的笑声中充满了善意与安慰。从那以后，鱼每次遇到危险，都会想起那只被困在地面，却仍然坚持不懈 地跳跃，最终摆脱困境的故事。

[对话第4轮]
(用户输入:)
!next
================================================================================
[对话第1轮]
(用户输入:)
你好
(Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4:)

你好！很高兴为您提供帮助！有什么我可以为你解答的问题或提供相关的信息吗？

[对话第2轮]
(用户输入:)
!quit
ZeroInferenceEngine clean_up!
ZeroNameServer clean_up!
quit gracefully
```