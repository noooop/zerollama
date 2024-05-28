# APISR: Anime Production Inspired Real-World Anime Super-Resolution

### setup

1. 安装 zerollama [配置环境](https://github.com/noooop/zerollama/tree/main/setup)

现在的工作目录是 .../zerollama

2. 安装 APISR

```
$ cd ../    # APISR 工程目录和 zerollama 平级
$ git clone https://github.com/Kiteretsu77/APISR.git  
```

3. 安装 APISR 依赖 
```
$ pip install -r zerollama\zerollama\models\apisr\requirements.txt  
```

4. 下载 APISR 模型
想办法把下面四个模型下载下来，并放入 APISR/pretrained
下载地址 https://github.com/Kiteretsu77/APISR/releases/
```
4x_APISR_RRDB_GAN_generator.pth
4x_APISR_GRL_GAN_generator.pth
2x_APISR_RRDB_GAN_generator.pth
4x_APISR_DAT_GAN_generator.pth
```

### License
This project is released under the GPL 3.0 license.

### Reference
[GITHUB](https://github.com/Kiteretsu77/APISR)

[Paper Link](https://arxiv.org/abs/2403.01598)