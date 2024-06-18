# PaddleOCR

PaddleOCR 旨在打造一套丰富、领先、且实用的 OCR 工具库，助力开发者训练出更好的模型，并应用落地。

## setup

> paddlepaddle-gpu==2.6.1 官方说支持 cuda12.0
> 
> 实测 cuda12.1 也可以运行，跟 torch==2.3.1 不冲突
> 
> lucky

1. 安装 依赖 
```
$ pip install -r zerollama\zerollama\models\paddleocr\requirements.txt  
```

2. 安装 cudnn8 

下载 cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip

解压到 某个地方 比如 E:\cudnn-windows-x86_64-8.9.7.29_cuda12-archive\bin

~./.zerollama/config/global.yml 加入 cudnn_path

```
cuda:
  cudnn_path: E:\cudnn-windows-x86_64-8.9.7.29_cuda12-archive\bin
```


## License
本项目的发布受<a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>许可认证。

## Reference
[GITHUB](https://github.com/PaddlePaddle/PaddleOCR)