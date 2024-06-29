# qihoo360 

## 360Zhinao

###  TL;DR
360智脑大模型特点如下：
- **基础模型**：采用 3.4 万亿 Tokens 的高质量语料库训练，以中文、英文、代码为主，在相关基准评测中，同尺寸有竞争力。
- **对话模型**：具有强大的对话能力，开放4K、32K、360K三种不同文本长度。据了解，360K（约50万字）是当前国产开源模型文本长度最长的。

### News
- [2024.04.12] 我们发布了360Zhinao-7B 1.0版本，同时开放Base模型和4K、32K、360K三种文本长度的Chat模型。

### License Agreement

本仓库源码遵循开源许可证Apache 2.0。

360智脑开源模型支持商用，若需将本模型及衍生模型用于商业用途，请通过邮箱(g-zhinao-opensource@360.cn)联系进行申请， 具体许可协议请见[《360智脑开源模型许可证》](https://github.com/Qihoo360/360zhinao/blob/main/360%E6%99%BA%E8%84%91%E5%BC%80%E6%BA%90%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E8%AF%81.txt)。

The source code of this repository follows the open-source license Apache 2.0.

360​Zhinao open-source models support commercial use. If you wish to use these models or continue training them for commercial purposes, please contact us via email (g-zhinao-opensource@360.cn) to apply. For the specific license agreement, please see [<<360 Zhinao Open-Source Model License>>](https://github.com/Qihoo360/360zhinao/blob/main/360%E6%99%BA%E8%84%91%E5%BC%80%E6%BA%90%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E8%AF%81.txt).


## 360LayoutAnalysis

###  TL;DR
1)涵盖中文论文、英文论文、中文研报三个垂直领域及1个通用场景模型；

2)轻量化推理快速【基于yolov8训练，单模型6.23MB】；

3)中文论文场景包含段落信息【CLDA不具备段落信息，我们开源独有】；

4)中文研报场景/通用场景【基于数万级别高质量数据训练，我们开源独有】

### News
- 2024-06-15，我们优先开源了面向论文和研报两个场景的版面分析轻量化模型权重及相应的标签体系，旨在能够识别文档中的段落边界等信息，并准确区分文本、图像、表格、公式等其他元素，最终推动产业发展。

- 2024-06-28，新增英文论文场景、通用场景两个新版式分析模型，开源版式分析模型达到4个。

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