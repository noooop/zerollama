# 配置文件

```
文件位置
windows
C:\Users\%USERNAME%\.zerollama\config\

linux
~/.zerollama/config/

```

## global
windows 示例
```
C:\Users\%USERNAME%\.zerollama\config\global.yml
```

```
huggingface:
  HF_ENDPOINT: https://hf-mirror.com        # huggingface 代理
  HF_HOME: D:/.cache/huggingface/           # huggingface 模型目录

modelscope:
  USE_MODELSCOPE: True                      # 是否使用 modelscope 默认为 True
  MODELSCOPE_CACHE: D:/.cache/modelscope/   # modelscope 模型目录
```

wsl2 示例
```
$ mkdir -p ~/.zerollama/config/
$ vim ~/.zerollama/config/global.yml
```

```
huggingface:
  HF_ENDPOINT: https://hf-mirror.com
  HF_HOME: /mnt/d/.cache/huggingface/
modelscope:
  USE_MODELSCOPE: True
  MODELSCOPE_CACHE: D:/.cache/modelscope/
```

