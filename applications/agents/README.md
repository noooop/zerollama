# 使用多智能体 agents

## 客户端支持多种 llm api
这个多智能体项目支持ollama和openai客户端，你可以任意选择熟悉的本地或网络api。

当然也可以试试本项目内部基于zmq的通讯协议。

## llm 服务端
不用费力的找了 llm 服务端了。本项目是个自带电池项目。

[配环境](https://github.com/noooop/zerollama/tree/main/setup)和[安装vllm](https://github.com/noooop/zerollama/tree/v0.5/zerollama/microservices/inference/vllm_green)。 (vllm 只支持linux， windows用户建议使用wsl2

server init 命令一键启动并部署模型
```
$ python -m applications.agents.cli server init applications/agents/deploy.yml
```

默认部署文件 “[deploy.yml](https://github.com/noooop/zerollama/blob/v0.5/applications/agents/deploy.yml)” 使用 vllm 部署 Qwen/Qwen2-7B-Instruct-GPTQ-Int4， 并部署 ollama 和 openai 网关。

## 教程
0. [支持 ollama和openai客户端、支持zerollama内部通讯协议](./tutorial/t0_llm_client.py)
1. [ConversableAgent - 两个大模型相互对话](./tutorial/t1_conversable_agent.py)
2. [waterfall workflow - 多智能体顺序执行的工作流程](./tutorial/t2_waterfall_workflow.py)


