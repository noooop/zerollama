# 检索增强生成 Retrieval-augmented generation (RAG)

## 快速使用指南

### 1. [配置环境](https://github.com/noooop/zerollama/tree/main/setup)

### 2. 准备文档

2.1. rag\documents 文件夹位置 及配置

默认 windows 为 C:\Users\%USERNAME%\.zerollama\rag\documents; linux 为 ~/.zerollama/rag/documents

可以在 [配置文件](https://github.com/noooop/zerollama/tree/main/zerollama/core/config) 指定新路径 比如

```
rag:
  path: Y:\.zerollama\rag\documents
```

2.2. 在 rag\documents 下 新建一个 collection，相当于一个知识库。 比如 test_collection
```
.../rag/documents/
└── test_collection
```

2.3. 添加内容

> 面前只支持txt纯文本格式

```
.../rag/documents/
└── test_collection
    └── text
        ├── a.txt
        └── b.txt
```

### 3. 部署 
服务端
```
$ python -m applications.rag.cli server run
ZeroNameServer: InMemoryNameServer running! port: 9527
ZeroManager for RootZeroManager running! port: 50439
```

部署配置文件 deploy.yml 格式如下
```
chat:
  models: ["Qwen/Qwen1.5-0.5B-Chat-AWQ"]

retriever:
  models: ["BAAI/bge-m3"]

reranker:
  models: ["BAAI/bge-reranker-v2-m3"]

vector_database:
  backend: "BruteForceVectorDatabase"

documents:
  collections: ["test_collection"]
```

客户端，提交部署命令
```
$ python -m applications.rag.cli deploy applications/rag/deploy.yml
================================================================================
pull models
....
================================================================================
start ZeroChatInferenceManager
================================================================================
start ZeroRetrieverInferenceManager
================================================================================
start ZeroRerankerInferenceManager
================================================================================
processing document
....
================================================================================
start ZeroVectorDatabaseManager
================================================================================
start RAGManager
================================================================================
start EntrypointManager
```

### 4. fastapi 接口模式
你可以在 http://locahost:8000/docs 看到 api文档

```
get /
get /api/chat_models
get /api/retriever_models
get /api/reranker_models
get /api/vector_databases/collections
post /api/chat
post /api/embeddings
get /api/default_qa_prompt_tmpl
post /api/rag
```

其中 最重要的 post /api/rag 参数如下
```
{
  "question": "string",
  "chat_model": "string",
  "retriever_model": "string",
  "reranker_model": "string",
  "collection": "string",
  "n_retriever_candidate": 10,
  "n_references": 3,
  "qa_prompt_tmpl_str": "string",
  "stream": false,
  "return_references": false
}
```

## 示例
### Curl 示例
```
curl -X 'POST' \
  'http://localhost:8000/api/rag' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "作者是谁？",
  "chat_model": "Qwen/Qwen1.5-0.5B-Chat-AWQ",
  "retriever_model": "BAAI/bge-m3",
  "reranker_model": "BAAI/bge-reranker-v2-m3",
  "collection": "test_collection",
  "n_retriever_candidate": 10,
  "n_references": 3
}'
```

输出
```
{
  "answer": {
    "model": "Qwen/Qwen1.5-0.5B-Chat-AWQ",
    "finish_reason": "stop",
    "content": "作者是菲利普·鲍尔",
    "completion_tokens": 9,
    "prompt_tokens": 354,
    "total_tokens": 363
  },
  "references": []
}
```


### python requests 流式输出示例

```
$ python -m applications.rag.api_test.main
```

## [支持的模型](https://github.com/noooop/zerollama#%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B)




