# qihoo360 360LayoutAnalysis

### setup

1. 安装 依赖 
```
$ pip install -r zerollama\models\qihoo360\requirements.txt  
```

4. 下载 模型
想办法把下面2个模型下载下来
下载地址 https://huggingface.co/qihoo360/360LayoutAnalysis/tree/main
```
paper-8n.pt
report-8n.pt

```
并放入与zerollama根目录 平级的 models/360LayoutAnalysis
```
- zerollama
    - ......
- models
    - 360LayoutAnalysis
        - paper-8n.pt
        - ....
```

### License
This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses.The content of this project itself is licensed under the [Apache license 2.0](./LICENSE.txt).

本仓库源码遵循开源许可证Apache 2.0。360LayoutAnalysis模型开源模型支持商用，若需将本模型及衍生模型用于商业用途，请通过邮箱([360ailab-nlp@360.cn](mailto:360ailab-nlp@360.cn))联系进行申请， 具体许可协议请见[《360LayoutAnalysis模型开源模型许可证》](./360LayoutAnalysis开源模型许可证.txt)。

### Reference
[GITHUB](https://github.com/360AILAB-NLP/360LayoutAnalysis/)

[huggingface](https://huggingface.co/qihoo360/360LayoutAnalysis)