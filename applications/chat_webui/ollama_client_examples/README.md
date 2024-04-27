# ollama client examples

## 支持
- list   输出字段可能不完整
- show   输出字段可能不一样
- chat 
- chat-stream

# 暂不支持
- generate
- embeddings

# 不支持
因为zerollama管理和加载模型的方法跟ollama不一样，所以以下操作不支持

- pull
- push
- create
- delete
- copy

请使用 [applications.chat_webui.cli](https://github.com/noooop/zerollama/tree/v0.2/applications/chat_webui#%E5%B8%AE%E5%8A%A9) 管理和加载模型
