# 如何使用 Continue 

## 1. [配置环境](https://github.com/noooop/zerollama/tree/main/setup)

## 2. 下载模型
```
# 下载 chat 模型
$ python -m applications.chat_webui.cli chat pull Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4
# 下载 retriever(embeddings) 模型
$ python -m applications.chat_webui.cli retriever pull BAAI/bge-m3
```

## 2.5 客户端帮助
```
# 客户端帮助
$ python -m applications.chat_webui.cli --help
Usage: python -m applications.chat_webui.cli [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  chat
  retriever
  server

## 举几个例子
# 列出支持的 chat 模型系列
$ python -m applications.chat_webui.cli chat list-families
Supported chat families:
deepseek-llm
MiniCPM
Qwen1.5
Yi

# 列出支持的 chat 模型家族成员
$ python -m applications.chat_webui.cli chat list-family Qwen1.5
+----------------------------------+------+--------------+-------+---------+----------+
| name                             | size | quantization | bits  | family  | protocol |
+----------------------------------+------+--------------+-------+---------+----------+
| Qwen/Qwen1.5-0.5B-Chat           | 0.5B |              |       | Qwen1.5 | chat     |
...
+----------------------------------+------+--------------+-------+---------+----------+
```

## 3. 启动服务端
```
$ python -m applications.chat_webui.cli server run
Control-C terminate
ZeroManager for ZeroChatInferenceEngine running! port: 53713
ZeroManager for ZeroRetrieverInferenceEngine running! port: 56194
ZeroNameServer: InMemoryNameServer running! port: 9527
HttpEntrypoints running! 0.0.0.0 8080
HttpEntrypoints running! 0.0.0.0 11434
INFO:     Started server process [5764]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
INFO:     Started server process [4988]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:11434 (Press CTRL+C to quit)

这个窗口就不要动了，其他命令在新窗口输入 （要说的这么细吗）
```

## 4. 加载模型
```
# 加载 chat 模型
$ python -m applications.chat_webui.cli chat start Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4
Wait Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4 available.
Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4 available now.

# 加载 retriever(embeddings) 模型
$ python -m applications.chat_webui.cli retriever start BAAI/bge-m3
Wait BAAI/bge-m3 available.
BAAI/bge-m3 available now.
```

## 5. 安装 vscode
[下载vscode](https://code.visualstudio.com/)

## 6. 安装 Continue
<img src="https://github.com/noooop/zerollama/blob/main/static/continue/setup1.png?raw=true" width="400">

## 7. 配置 Continue
<img src="https://github.com/noooop/zerollama/blob/main/static/continue/setup2.png?raw=true" width="400">
<img src="https://github.com/noooop/zerollama/blob/main/static/continue/setup3.png?raw=true" width="400">

```
{
  "title": "Qwen/Qwen1.5-7B-Chat",            这里填想显示的模型名称
  "model": "Qwen/Qwen1.5-7B-Chat",            这里填实际的模型名称
  "completionOptions": {},                    
  "apiBase": "http://192.168.50.123:11434",   这里填机器的IP地址和端口
  "provider": "ollama"
}
```

## 7. 与大模型对话
<img src="https://github.com/noooop/zerollama/blob/main/static/continue/chat.png?raw=true" width="400">

## 8. 与大模型结对编程
<img src="https://github.com/noooop/zerollama/blob/main/static/continue/chat.png?raw=true" width="400">

## 9. 退出
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


# 玩的开心
