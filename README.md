# Microservices framework for large language model application linked by Zeromq

名字起的相当俗气，俗气就对了，让人一眼就知道知道这个库是做啥的

## 使用zmq的大语言应用微服务框架

### 知识点一
[服务器端和客户端 server & client](https://github.com/noooop/zerollama/tree/main/tutorial/basis/0x01)

### 知识点二
[序列化和反序列化 serialize & unserialize](https://github.com/noooop/zerollama/tree/main/tutorial/basis/0x02)

### 合起来
[最小化可行产品 minimum viable product](https://github.com/noooop/zerollama/tree/main/tutorial/basis/0x03)

相当于，扔两块木板到太平洋，看我们能到多远吧

[更多教程](https://github.com/noooop/zerollama/tree/main/tutorial)

[安装&配环境](./setup)

# 应用
1. [与大模型对话 客户端](./applications/chat_cli)
2. [与大模型对话 webui](./applications/chat_webui)
3. [检索增强生成 (RAG)](./applications/rag)

# 支持的模型
## Chat 模型
- [deepseek-llm](https://github.com/noooop/zerollama/tree/main/zerollama/models/deepseek)
- [MiniCPM](https://github.com/noooop/zerollama/tree/main/zerollama/models/openbmb)
- [Qwen1.5](https://github.com/noooop/zerollama/tree/main/zerollama/models/qwen)
- [Yi](https://github.com/noooop/zerollama/tree/main/zerollama/models/yi)
- [llama3](https://github.com/noooop/zerollama/tree/main/zerollama/models/llama) [llama3zh](https://github.com/noooop/zerollama/tree/main/zerollama/models/llama)
- [XVERSE](https://github.com/noooop/zerollama/tree/main/zerollama/models/xverse)

## Retriever(Embeddings) 模型
- [bge-retrieval](https://github.com/noooop/zerollama/tree/main/zerollama/models/baai)
- [bce-embedding](https://github.com/noooop/zerollama/tree/main/zerollama/models/bce)
- [uniem(m3e)](https://github.com/noooop/zerollama/tree/main/zerollama/models/uniem)

## Reranker 模型
- [bge-reranker](https://github.com/noooop/zerollama/tree/main/zerollama/models/baai)
- [bce-reranker](https://github.com/noooop/zerollama/tree/main/zerollama/models/bce)

## Vision-Language Multimodal 模型 
- [CogVLM2](https://github.com/noooop/zerollama/tree/main/zerollama/models/thudm)
- [MiniCPM-V](https://github.com/noooop/zerollama/tree/main/zerollama/models/openbmb)
- [DeepSeek-VL](https://github.com/noooop/zerollama/tree/main/zerollama/models/deepseek)

## Super Resolution 模型 
- [APISR](https://github.com/noooop/zerollama/tree/main/zerollama/models/apisr)

## Contribute
贡献代码请交逻辑清晰思路、简单直接的代码、不要有魔法，甚至可以牺牲一部分性能换取可读性

代码是人读的，也是人维护的，顺便机器可以运行

> Despite its immensity, it is both simple and elegant, fulfilling its designer's dream to create ‘an enormous object drawn as faintly as possible’.
> - New Concept English Developing Skills Lesson 17 The longest suspension bridge in the world