# 与大语言模型对话

[配置环境](https://github.com/noooop/zerollama/tree/main/setup)

## 帮助
```
$ python -m applications.chat_webui.cli --help
Usage: python -m applications.chat_webui.cli [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  list-families
  list-family
  pull
  run-server
  start
  terminate
```

## 启动
```
$ python -m applications.chat_webui.cli run-server
Control-C terminate
HttpEntrypoints running! 127.0.0.1 11434
HttpEntrypoints running! 127.0.0.1 8080
ZeroNameServer: InMemoryNameServer running! port: 9527
ZeroManager for ZeroInferenceEngine running! port: 50755
```

## 关闭
```
# Control-C 关闭
HttpEntrypoints clean_up!
HttpEntrypoints clean_up!
ZeroNameServer clean_up!
ZeroManager clean_up!
```

## list-families 列出支持的模型家族
```
python -m applications.chat_webui.cli list-families
Supported chat families:
deepseek-llm
MiniCPM
Qwen1.5
Yi
```

## list-family 列出支持的模型家族成员
```
$ python -m applications.chat_webui.cli list-family Qwen1.5
+----------------------------------+------+--------------+-------+---------+----------+
| name                             | size | quantization | bits  | family  | protocol |
+----------------------------------+------+--------------+-------+---------+----------+
| Qwen/Qwen1.5-0.5B-Chat           | 0.5B |              |       | Qwen1.5 | chat     |
...
+----------------------------------+------+--------------+-------+---------+----------+
```

## 下载模型 pull
```
$ python -m applications.chat_webui.cli pull Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4
```

## 加载模型 start
```
$ python -m applications.chat_webui.cli start Qwen/Qwen1.5-0.5B-Chat
$ python -m applications.chat_webui.cli start Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4
$ python -m applications.chat_webui.cli start Qwen/Qwen1.5-0.5B-Chat-AWQ
```

## 卸载模型 terminate
```
$ python -m applications.chat_webui.cli terminate Qwen/Qwen1.5-0.5B-Chat
$ python -m applications.chat_webui.cli terminate Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4
$ python -m applications.chat_webui.cli terminate Qwen/Qwen1.5-0.5B-Chat-AWQ
```

# 注意
1. 考虑到大语言模型动辄几个G，下载相当缓慢，所以默认设置了local_files_only=True， 模型必须先下载，才能加载。
2. 因为不同模型，不同上下文长度，占用的显存不一样，尤其是加载多个模型，很有可能显存不够，暂时先手动加载卸载模型

# webui

**推荐**

[page-assist](https://github.com/n4ze3m/page-assist) 

You can install the extension from the [Chrome Web Store](https://chromewebstore.google.com/detail/page-assist-%E6%9C%AC%E5%9C%B0-ai-%E6%A8%A1%E5%9E%8B%E7%9A%84-web/jfgfiigpkhlkbnfnbobbkinehhfdhndo)

<img src="https://github.com/noooop/zerollama/blob/main/static/images/chat_server.png?raw=true" width="400">

**兼容**


# api

