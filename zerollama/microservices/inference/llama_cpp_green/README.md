# llama-cpp-python 

## 安装
使用 cuda 12.1 与 pytorch 2.2 一致

[下载并安装 CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-1-download-archive)

```
Pre-built Wheel
pip install llama-cpp-python==0.2.69 \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
  
编译安装
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.69 --no-cache-dir
```

其他安装方式请 [参考](https://github.com/abetlen/llama-cpp-python#installation)