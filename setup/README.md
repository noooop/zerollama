# Windows 11
[详细配置](./windows.md)

# Windows 11 + wsl2
[详细配置](./wsl2.md)

# Ubuntu
[详细配置](./ubuntu.md)


# 安装依赖
## [pytorch](https://pytorch.org/get-started/locally/)  

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

以下版本经过测试

torch-2.2.2+cu121

## [transformers](https://github.com/huggingface/transformers)

pip install transformers==4.38.2

以下版本经过测试

transformers-4.38.2

## [accelerate](https://github.com/huggingface/accelerate)

pip install accelerate

以下版本经过测试

accelerate-0.29.2

## 量化模型支持
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