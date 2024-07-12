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
3. [reflect - 反思，通过让大语言模型扮演多个角色，自问自答，不断提高输出结果](./tutorial/t3_reflect.py)
4. [divide and conquer - 分治，将一个大任务分成多个子任务，让大语言模型扮演多个角色分别完成子任务，汇总形成更全面的结果](./tutorial/t4_divide_and_conquer.py)


## 全局 llm_config
通过配置文件，设置全局 llm_config，默认全局配置会覆盖代码里传入的配置，你可以使用 global_priority 来控制是否全局配置优先
```
文件位置
windows
C:\Users\%USERNAME%\.zerollama\config\

linux
~/.zerollama/config/
```

比如使用本地 ollama
```
agents:
  llm_config:
    model: "qwen2:0.5b"
    type: "ollama"
```
