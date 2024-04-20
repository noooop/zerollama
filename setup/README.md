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
$ conda create -n zerollama_v0.1 python=3.11 anaconda

# 查看虚拟环境
$ conda env list 

# 激活虚拟环境
$ conda activate zerollama_v0.1

# 安装依赖, 只会安装在这个虚拟环境里
# pip install .....

# 停用虚拟环境
$ conda deactivate

# 删除虚拟环境
conda remove -n zerollama_v0.1 --all
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

# 切换到工作目录
项目默认的工作目录是项目根目录
```
$ cd zerollama
```

 

