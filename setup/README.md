# Windows 11
[详细配置](./windows.md)

# Windows 11 + wsl2
[详细配置](./wsl2.md)

# Ubuntu
[详细配置](./ubuntu.md)

# 试试conda虚拟环境
环境管理允许用户方便安装不同版本的python环境，并在不同环境之间快速地切换。

## 自动配置
```
# ubuntu&wsl2
$ conda env create -f environment_linux.yml

# windows
$ conda env create -f environment_windows.yml
```
区别是 ubuntu&wsl2 使用pip安装pytorch， windows使用conda安装pytorch

网络问题多试几次，找找代理，比如[TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/),自己克服一下吧，叹气

下面的配置都已经一键安装好了

## 手动配置
```
# 创建虚拟环境
$ conda create -n zerollama_v0.4 python=3.11 anaconda

# 查看虚拟环境
$ conda env list 

# 激活虚拟环境
$ conda activate zerollama_v0.4

# 安装依赖, 只会安装在这个虚拟环境里
# pip install .....

# 停用虚拟环境
$ conda deactivate

# 删除虚拟环境
conda remove -n zerollama_v0.4 --all
```

## 安装依赖
### [pytorch](https://pytorch.org/get-started/locally/)  

```
# ubuntu&wsl2
$ pip install torch torchvision torchaudio

# windows
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**注意**
[PyTorch 2.2](https://github.com/pytorch/pytorch/releases/tag/v2.2.0) SDPA 已经集成 FlashAttention-2，不需要单独安装，毕竟这货编译起来非常慢。

以下版本经过测试

torch-2.2.2+cu121

### [transformers](https://github.com/huggingface/transformers)

pip install transformers==4.38.2

以下版本经过测试

transformers-4.38.2

### [accelerate](https://github.com/huggingface/accelerate)

pip install accelerate

以下版本经过测试

accelerate-0.29.2

### 量化模型支持
[optimum](https://github.com/huggingface/optimum)

pip install optimum

以下版本经过测试

optimum-1.18.1

[GPTQ](https://github.com/AutoGPTQ/AutoGPTQ)

以下版本经过测试

auto-gptq-0.7.1

[AWQ](https://github.com/casper-hansen/AutoAWQ)

以下版本经过测试

autoawq-0.2.4 (Pin transformers>=4.35.0,<=4.38.2) [详情](https://github.com/casper-hansen/AutoAWQ/releases/tag/v0.2.4)

# clone工程代码
```
$ git clone https://github.com/noooop/zerollama.git
```

# 其他需要手工安装的库
很遗憾，并不是所有库都可以自动安装好，有些库有复杂的依赖，或者跟其他库之间有冲突，需要人工安装

## 安装 llama-cpp-python
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

以下版本经过测试

llama-cpp-python==0.2.69 + cuda 12.1

## 安装 BCEmbedding

[BCEmbedding](https://github.com/noooop/zerollama/tree/main/zerollama/models/bce) 依赖 transformers>=4.35.0,<4.37.0

[qwen1.5](https://github.com/noooop/zerollama/tree/main/zerollama/models/qwen) 依赖 transformers>=4.37.0

本项目在 transformers==4.38.2 下测试，并考虑下个版本使用最新的 transformers==v4.41.0。

实测先安装BCEmbedding，然后再升级到transformers==4.38.2也是可用的。

```
pip install BCEmbedding==0.1.5
```

## 向量数据库

### faiss
```
conda install -c pytorch faiss-cpu
```

### hnswlib
[Building AVX Extensions ](https://github.com/chroma-core/hnswlib?tab=readme-ov-file#building-avx-extensions)

For maximum compatibility, the distributed wheels are not compiled to make use of Advanced Vector Extensions (AVX). If your hardware supports AVX, you may get better performance by recompiling this library on the machine on which it is intended to run.

```
pip install chroma-hnswlib==0.7.3
```

### chromadb

chromadb 依赖 hnswlib， 所以使用Building AVX Extensions 会提高性能。
```
pip install chromadb==0.5.0
```

# 切换到工作目录
项目默认的工作目录是项目根目录
```
$ cd zerollama
```

 
# 配置文件
[跳转链接](https://github.com/noooop/zerollama/tree/main/zerollama/core/config)
