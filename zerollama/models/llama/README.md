# LLaMa
LLaMa是Meta公司发布的大型语言模型。

Meta做了Openai应该做的事情，Meta是真开源，Meta牛逼。

# News
* 2024.04.18: LLaMa 3 首发 8B/70B
* 2023.07.18: LLaMa 2 发布 7B/13B/70B
* 2023.02.27: LLaMa 发布  7B/13B/33B/65B


## LLama3
（当地时间）2024年4月18日，Meta在官网上宣布公布了旗下最新大模型Llama 3，也是目前市场上最好的开源大模型。上下文窗口为8k，使用了超过15T的公开数据，其中5%为非英文数据，涵盖30多种语言，训练数据量是前代Llama 2的七倍，包含的代码数量是Llama 2的四倍。

首发8B和70B，而Llama 3的400B+模型虽然仍在训练中...

根据Meta的测试结果，Llama 3 8B模型在MMLU、GPQA、HumanEval等多项性能基准上均超过了Gemma 7B和Mistral 7B Instruct，70B模型则超越了名声在外的闭源模型Claude 3的中间版本Sonnet，和谷歌的Gemini Pro 1.5相比三胜两负。


### 注意 
To download the weights from Hugging Face, You need to agree to share your contact information.

llama3 模型下载需要注册huggingface账户、填写申请、配置HF_TOKEN  

### License
Our model and weights are licensed for both researchers and commercial entities, upholding the principles of openness. Our mission is to empower individuals, and industry through this opportunity, while fostering an environment of discovery and ethical AI advancements.

See the LICENSE file, as well as our accompanying Acceptable Use Policy

### Reference
[GITHUB](https://github.com/meta-llama/llama3)

[Hugging Face](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6)

[Homepage](https://llama.meta.com/llama3)

## LLaMa3-zh

llama3训练数据里中文占比很少，所以会有中文提问英文回答的情况，以下是一些社区微调中文版LLaMa3.

- [Unichat-llama3-Chinese-8B](https://huggingface.co/UnicomLLM/Unichat-llama3-Chinese-8B)


### Reference
[xianbao's llama3-zh collections](https://huggingface.co/collections/xianbao/llama3-zh-662ba8503bdfe51948a28403)