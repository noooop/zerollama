# vllm 

vLLM 是一个快速且易于使用的 LLM 推理和服务库。

[GITHUB](https://github.com/vllm-project/vllm)

[Documentation](https://docs.vllm.ai/en/stable/)

vLLM 使用了 PagedAttention、Continuous batching、Quantization、Optimized CUDA kernels 加速buff叠满

## 安装
vllm 只支持linux， windows用户建议使用wsl2
1. [配环境](https://github.com/noooop/zerollama/tree/main/setup)
2. 安装 vllm
```
$ pip install -r zerollama\microservices\inference\vllm_green\requirements.txt  
```