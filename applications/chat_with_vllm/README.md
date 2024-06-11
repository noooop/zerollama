# 使用vllm

## 帮助
```
$ python -m applications.chat_with_vllm.cli --help
Usage: python -m applications.chat_with_vllm.cli [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  chat
  deploy
  retriever
  server
```

## 服务端

### 启动
```
$ python -m applications.chat_with_vllm.cli server run
ZeroNameServer: InMemoryNameServer running! port: 9527
ZeroManager for RootZeroManager running! port: 59089
```

### 关闭
```
# Control-C 关闭
```

### 部署模型
```
$ python -m applications.chat_with_vllm.cli deploy applications/chat_with_vllm/deploy.yml
```

### 部署文件 deploy.yml
[使用vllm示例](https://github.com/noooop/zerollama/blob/main/applications/chat_with_vllm/deploy.yml)
