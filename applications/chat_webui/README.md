# 与大语言模型对话

# ollama compatible
[支持情况](https://github.com/noooop/zerollama/blob/main/applications/chat_webui/ollama_client_examples/README.md)

# openai compatible
[支持情况](https://github.com/noooop/zerollama/blob/main/applications/chat_webui/openai_client_examples/README.md)

# 浏览器插件
[page-assist](https://github.com/n4ze3m/page-assist) 
[快速指南](./page-assist.md) 

# vscode插件

[Continue](https://marketplace.visualstudio.com/items?itemName=Continue.continue)
[快速指南](./continue.md) 


# webui


# 使用指南

[配置环境](https://github.com/noooop/zerollama/tree/main/setup)


## 帮助
```
$ python -m applications.chat_webui.cli --help
Usage: python -m applications.chat_webui.cli [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  chat
  retriever
  server
```

## 服务端

### 启动
```
$ python -m applications.chat_webui.cli server run
Control-C terminate
ZeroNameServer: InMemoryNameServer running! port: 9527
HttpEntrypoints ollama_compatible running! 0.0.0.0 11434
ZeroManager for ZeroRetrieverInferenceManager running! port: 52187
HttpEntrypoints openai_compatible running! 0.0.0.0 8080
ZeroManager for ZeroChatInferenceManager running! port: 54925
INFO:     Started server process [24800]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:11434 (Press CTRL+C to quit)
INFO:     Started server process [13644]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### 关闭
```
# Control-C 关闭
INFO:     Shutting down
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Application shutdown complete.
INFO:     Finished server process [5764]
INFO:     Finished server process [4988]
HttpEntrypoints clean_up!
HttpEntrypoints clean_up!
ZeroManager clean_up!
ZeroChatInferenceEngine clean_up!
ZeroChatInferenceEngine clean_up!
ZeroManager clean_up!
```

## chat 模型控制

list-families、list-family、pull、start、terminate

### list-families 列出支持的模型家族
```
$ python -m applications.chat_webui.cli chat list-families
Supported chat families:
deepseek-llm
MiniCPM
Qwen1.5
Qwen1.5_gguf
Yi
Yi-1.5
llama3
llama3zh
xverse
xverse_gguf
```

### list-family 列出支持的模型家族成员
```
$ python -m applications.chat_webui.cli chat list-family Qwen1.5
+----------------------------------+------+--------------+-------+---------+----------+
| name                             | size | quantization | bits  | family  | protocol |
+----------------------------------+------+--------------+-------+---------+----------+
| Qwen/Qwen1.5-0.5B-Chat           | 0.5B |              |       | Qwen1.5 | chat     |
...
+----------------------------------+------+--------------+-------+---------+----------+
```

### pull 下载模型 
```
$ python -m applications.chat_webui.cli chat pull Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4
```

### start 加载模型 
```
$ python -m applications.chat_webui.cli chat start Qwen/Qwen1.5-0.5B-Chat
$ python -m applications.chat_webui.cli chat start Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4
$ python -m applications.chat_webui.cli chat start Qwen/Qwen1.5-0.5B-Chat-AWQ
```

支持 llama cpp gguf 模型 
```
$ python -m applications.chat_webui.cli run Qwen/Qwen1.5-0.5B-Chat-GGUF+*q8_0.gguf
```

### terminate 卸载模型 
```
$ python -m applications.chat_webui.cli chat terminate Qwen/Qwen1.5-0.5B-Chat
$ python -m applications.chat_webui.cli chat terminate Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4
$ python -m applications.chat_webui.cli chat terminate Qwen/Qwen1.5-0.5B-Chat-AWQ
```

## retriever(embeddings) 模型控制
与 chat大同小异，list-families、list-family、pull、start、terminate

# 注意
1. 考虑到大语言模型动辄几个G，下载相当缓慢，所以默认设置了local_files_only=True， 模型必须先下载，才能加载。
2. 因为不同模型，不同上下文长度，占用的显存不一样，尤其是加载多个模型，很有可能显存不够，暂时先手动加载卸载模型
