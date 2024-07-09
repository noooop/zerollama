# Microservices framework for large language model application linked by Zeromq

名字起的相当俗气，俗气就对了，让人一眼就知道知道这个库是做啥的

## 使用zmq的大语言应用微服务框架

[文档&教程](https://github.com/noooop/zerollama/tree/main/tutorial)

[安装&配环境](./setup)

# 应用
1. [与大模型对话 客户端](./applications/chat_cli)
2. [与大模型对话 webui](./applications/chat_webui)
3. [检索增强生成 (RAG) (work in progress!)](./applications/rag)
4. [使用 vllm](./applications/chat_with_vllm)
5. [OCR (work in progress!)](./applications/ocr)

# 支持的推理框架
## Chat 模型
- [huggingface transformers](https://github.com/noooop/zerollama/tree/main/zerollama/microservices/inference/transformers_green)
- [llama.cpp](https://github.com/noooop/zerollama/tree/main/zerollama/microservices/inference/llama_cpp_green)
- [vllm (gevent)](https://github.com/noooop/zerollama/tree/main/zerollama/microservices/inference/vllm_green)

## Retriever(Embeddings) 模型
- [sentence transformer](https://github.com/noooop/zerollama/tree/main/zerollama/microservices/inference/sentence_transformer_green)

# 支持的模型
## Chat 模型
- [deepseek-llm](https://github.com/noooop/zerollama/tree/main/zerollama/models/deepseek)
- [MiniCPM](https://github.com/noooop/zerollama/tree/main/zerollama/models/openbmb)
- [Qwen1.5](https://github.com/noooop/zerollama/tree/main/zerollama/models/qwen) [Qwen2](https://github.com/noooop/zerollama/tree/main/zerollama/models/qwen)
- [Yi](https://github.com/noooop/zerollama/tree/main/zerollama/models/yi)
- [llama3](https://github.com/noooop/zerollama/tree/main/zerollama/models/llama) [llama3zh](https://github.com/noooop/zerollama/tree/main/zerollama/models/llama)
- [XVERSE](https://github.com/noooop/zerollama/tree/main/zerollama/models/xverse)
- [Yuan](https://github.com/noooop/zerollama/tree/main/zerollama/models/yuan)
- [Mistral](https://github.com/noooop/zerollama/tree/main/zerollama/models/mistral)
- [GLM-4](https://github.com/noooop/zerollama/tree/main/zerollama/models/thudm)
- [bilibili Index](https://github.com/noooop/zerollama/tree/main/zerollama/models/bilibili)
- [360Zhinao](https://github.com/noooop/zerollama/tree/main/zerollama/models/qihoo360)
- [InternLM](https://github.com/noooop/zerollama/tree/main/zerollama/models/lnternlm)

## Retriever(Embeddings) 模型
- [bge-retrieval](https://github.com/noooop/zerollama/tree/main/zerollama/models/baai)
- [bce-embedding](https://github.com/noooop/zerollama/tree/main/zerollama/models/bce)
- [uniem(m3e)](https://github.com/noooop/zerollama/tree/main/zerollama/models/uniem)
- [acge](https://github.com/noooop/zerollama/tree/main/zerollama/models/aspire)

## Reranker 模型
- [bge-reranker](https://github.com/noooop/zerollama/tree/main/zerollama/models/baai)
- [bce-reranker](https://github.com/noooop/zerollama/tree/main/zerollama/models/bce)
- [jina-reranker-v2](https://github.com/noooop/zerollama/tree/main/zerollama/models/jina)
- [360Zhinao-1.8B-Reranking](https://github.com/noooop/zerollama/tree/main/zerollama/models/qihoo360)

## Vision-Language Multimodal 模型 
- [CogVLM2](https://github.com/noooop/zerollama/tree/main/zerollama/models/thudm)
- [MiniCPM-V](https://github.com/noooop/zerollama/tree/main/zerollama/models/openbmb)
- [DeepSeek-VL](https://github.com/noooop/zerollama/tree/main/zerollama/models/deepseek)
- [Florence-2](https://github.com/noooop/zerollama/tree/main/zerollama/models/florence)

## Super Resolution 模型 
- [APISR](https://github.com/noooop/zerollama/tree/main/zerollama/models/apisr)

## Optical Character Recognition 全家桶
### Text Line Detection
- [surya_tld](https://github.com/noooop/zerollama/tree/main/zerollama/models/surya)
- [ragflow/deepdoc_tld](https://github.com/noooop/zerollama/tree/main/zerollama/models/ragflow)
- [PaddleOCR_tld](https://github.com/noooop/zerollama/tree/main/zerollama/models/paddleocr)

### Text Recognition
- [surya_tr](https://github.com/noooop/zerollama/tree/main/zerollama/models/surya)
- [ragflow/deepdoc_tr](https://github.com/noooop/zerollama/tree/main/zerollama/models/ragflow)
- [PaddleOCR_tr](https://github.com/noooop/zerollama/tree/main/zerollama/models/paddleocr)

### Document Layout Analysis
- [moured/YOLOv10-Document-Layout-Analysis](https://github.com/noooop/zerollama/tree/main/zerollama/models/moured)
- [360LayoutAnalysis](https://github.com/noooop/zerollama/tree/main/zerollama/models/qihoo360)
- [surya_dla](https://github.com/noooop/zerollama/tree/main/zerollama/models/surya)
- [ragflow/deepdoc_dla](https://github.com/noooop/zerollama/tree/main/zerollama/models/ragflow)
- [PaddleOCR_dla](https://github.com/noooop/zerollama/tree/main/zerollama/models/paddleocr)

### Reading Order Detection
- [surya_rod](https://github.com/noooop/zerollama/tree/main/zerollama/models/surya)

### Table Structure Recognition
- [ragflow/deepdoc_tsr](https://github.com/noooop/zerollama/tree/main/zerollama/models/ragflow)
- [PaddleOCR_tsr](https://github.com/noooop/zerollama/tree/main/zerollama/models/paddleocr)

## Contribute
贡献代码请交逻辑清晰思路、简单直接的代码、不要有魔法，甚至可以牺牲一部分性能换取可读性

代码是人读的，也是人维护的，顺便机器可以运行

> Despite its immensity, it is both simple and elegant, fulfilling its designer's dream to create ‘an enormous object drawn as faintly as possible’.
> - New Concept English Developing Skills Lesson 17 The longest suspension bridge in the world