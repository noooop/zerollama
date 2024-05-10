# 大语言模型推理理论极限和实际测试一， 单用户独占实时交互

# TL;DR
1. 限定本文讨论单用户独占实时交互的大语言模型推理场景，区分于训练阶段，多用户实时交互推理和离线批处理。这种场景下，最重要的指标是延迟（Latency），包括首字延迟（First Token Latency）和生成延迟（Latency）。
2. 通过分析大语言模型的推理机制的时间和空间复杂度，计算出大语言模型的计算量 FLOPs 和访存量 MACs。
3. 分别计算 Prefill 阶段和 Decoding 阶段推理理论极限
4. 讨论GQA、量化、投机采样、对推理理论极限的影响
5. 分别测试 HuggingFace Transformers 库和 llama.cpp 实际推理速度测试

# 1. 讨论范围限定
大模型使用分为训练和推理两个阶段

- 训练阶段，因为要做反向传播，要保存中间状态，要更新优化器状态，性能优化技术比如 flashattention、混合精度训练（Mixed-precision Training）、
ZeRO Redundancy Optimizer、PEFT等，都是训练专用的，推理时用不着。本文主要讨论推理阶段，训练阶段专用的技术不在本文讨论范围内。
- 推理阶段，实际包含很多种使用场景：

单用户独占实时交互，机器只为一个用户服务。这是个人PC或者手机端的典型使用场景。
这种场景下，最重要的指标是延迟（Latency），包括首字延迟（First Token Latency）也就是系统生成第一个字符所需的响应时间，
和生成延迟（Latency）之后的字符响应时间。
吞吐量（Throughput）就是延迟的倒数，乏趣没有讨论价值。**本文关注的是这种场景**。

多用户实时交互。服务器部署，云服务提供商的典型场景。
需要平衡吞吐量（Throughput）和延迟（Latency）。
一般是设定服务级别协议 (Service-level agreement)， 比如首字延迟（First Token Latency）小于 2~3 秒， 
生成延迟（Latency）小于 50 毫秒，也就是一秒钟生成 20 个字符，就能达到用户感觉流畅。在sla规定的延迟下，尽可能多的服务用户。
**[下一篇文章]()(还没有写)关注这个这个场景。**

离线批处理。因为不需要实时交互，所以不用考虑延迟（Latency），只需要优化吞吐量（Throughput）以及用最经济的方式（使用性价比最高的卡）完成任务。

以qwen1.5家族为例，有八种尺寸 0.5B, 1.8B, 4B, 7B, 14B, 32B, 72B, 110B。方便纵向对比。

# 2. 推理机制
## 2.1 推理整体流程
LLM 是一个自回归模型，用之前的词，预测下一个词的概率分布。
然后在分布中采样一个新词，重复之前步骤，直到新采样的词为生成结束标识（或者到达指定长度等）。

LLM 使用了 Masked Self-Attention，上下文，llm只能“看到上文”，“看不到下文“。
相比于“上下文都能看到”的[BERT](https://arxiv.org/abs/1810.04805), 对于语义的理解肯定要弱一点，但也给了缓存中间经过的可能，性能会高一些。

更具体的，LLM 每个词只依赖之前的词，不依赖之后的词，也就是中间状态也只依赖之前的词，不依赖之后的词，所以中间经过可以缓存起来。
相比之下 BERT 每个词的状态依赖前后的词，如果后面增加一个词，前面词的中间状态就不一样了，需要重算。

带有中间状态缓存的伪代码。

```
def chat(messages, tokenizer, model):
    prompt = tokenizer.apply_chat_template(messages)                     # 1. 应用对话模版，chat_template 是 Jinja template
    prompt_tokens = tokenizer(prompt)                                    # 2. 将 prompt 使用 tokenizer 分词

    completion_tokens = []                                               # 输出的 tokens
    tokens = prompt_tokens                                               # 待处理的 tokens
    hidden_states_cache = []                                             # 模型中间状态缓存
    
    while True:
        logits, hidden_states = model.eval(tokens, hidden_states_cache)  # 3. 模型推理
        token = model.sample(logits)                                     # 4. 采用下一个 token
        completion_tokens.append(token)                                  # 输出采样的 token
        
        if model.token_is_eog(token):                                    # 5. 如果采样的 token 是结束标识 
            break                                                        #    生成结束

        else:                                                            # 6. 如果没有结束，将新 token 加入待处理的 tokens
            tokens.clear()                                               #    清空待处理的 tokens
            tokens.append(token)                                         #    将新采样的 token 加入待处理的 tokens
            hidden_states_cache.extend(hidden_states)                    #    将新的中间状态加入缓存
    
    text = tokenizer.detokenize(completion_tokens)                       # 7. 将生成的 tokens 解码为字符
    return text
```

## 2.2 整体流程
分词器(步骤1, 2, 7)和采样(步骤4)，有复杂的分支，速度也很快，一般在CPU上完成，具体原理请参考：

[tokenization_qwen2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/tokenization_qwen2.py)

[分词器tokenizer](https://zhuanlan.zhihu.com/p/649030161)  

[Subword算法](https://zhuanlan.zhihu.com/p/86965595)

[transformers sample实现](https://github.com/huggingface/transformers/blob/092f1fdaa4224fdd88c616dc9678e6fcb37bfffd/src/transformers/generation/utils.py#L2501)

[采样 sample](https://zhuanlan.zhihu.com/p/664293575)

模型推理(步骤3)过程请参考:

[modeling_qwen2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)


伪代码
```
def eval(tokens):
    inputs_embeds = embed_tokens(tokens)                # 1. Embedding
    hidden_states = inputs_embeds
    for decoder_layer in layers:
        hidden_states = decoder_layer(hidden_states)    # 2. Decoder_layer
    hidden_states = norm(hidden_states)
    logits = lm_head(hidden_states)                     # 3. lm_head
    next_token_logits = logits[:, -1, :]                # 4. 采样下一个词只需要最后一个词的logits
    return next_token_logits


def decoder_layer(hidden_states):
    residual = hidden_states
    hidden_states = input_layernorm(hidden_states)
    hidden_states = self_attn(hidden_states)
    hidden_states = residual + hidden_states
    
    residual = hidden_states
    hidden_states = post_attention_layernorm(hidden_states)
    hidden_states = mlp(hidden_states)
    hidden_states = residual + hidden_states
    
    return hidden_states
```


# 3. 参数量计算
> 1. 单位浮点数占用空间下面计为 F:
> - float32: F = 4bytes = 32bits
> - float16: F = 2bytes = 16bits
> - bfloat16: F = 2bytes = 16bits
> 2. M = 1024 * 1024 
> 
>    B = 1024 * 1024 * 1024
>
> 
> 3. Embedding 每个token只取其中一条，一般不计入"激活参数量"， 下称参数量


## 3.1 模型参数量计算
以 0.5B 为例:
```
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 1024)
    (layers): ModuleList(
      (0-23): 24 x Qwen2DecoderLayer(
        (self_attn): Qwen2SdpaAttention(
          (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
          (o_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=1024, out_features=2816, bias=False)
          (up_proj): Linear(in_features=1024, out_features=2816, bias=False)
          (down_proj): Linear(in_features=2816, out_features=1024, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm()
        (post_attention_layernorm): Qwen2RMSNorm()
      )
    )
    (norm): Qwen2RMSNorm()
  )
  (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
)


Embedding 参数量 = vocab_size * hidden_size = 151936 * 1024 = 148.375 M  (不计入激活参数量

q_proj, k_proj, v_proj带bias， o_proj不带 gate_proj, up_proj, down_proj都不带bias

一个 DecoderLayer Attention 参数量 = 3 * hidden_size * hidden_size + 1 * (hidden_size + 1) * hidden_size  
                                 = 3 * 1024 * 1024 + 1 * (1024+1) * 1024 = 4M
一个 DecoderLayer MLP 参数量 = 3 * hidden_size * intermediate_size = 3 * 1024 * 2816 = 8.25M

DecoderLayer 一共参数量 = num_hidden_layers X (Attention 参数量 + MLP 参数量) = 24 * (4M + 8.25M) = 294 M
lm_head 参数量 = hidden_size * vocab_size = 1024 * 151936 = 148.375 M

Qwen 1.5 总参数量 = 294 M + 148.375 M = 442.375 M = 0.432 B
```
qwen1.5 32B 和 110B 使用了 GQA
```
32B n_groups = num_attention_heads // num_key_value_heads = 40 // 8 = 5
110B n_groups = num_attention_heads // num_key_value_heads = 64 // 8 = 8

32B head_dim = hidden_size // num_attention_heads = 5120 // 40 = 128
110B head_dim = hidden_size // num_attention_heads = 8192 // 64 = 128

q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=True)
k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True)
v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True)
o_proj = nn.Linear(num_heads * num_attention_heads, hidden_size, bias=False)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# repeat k/v heads if n_kv_heads < n_heads
key_states = repeat_kv(key_states, n_groups)
value_states = repeat_kv(value_states, n_groups)

可以看到 k_proj 和 v_proj 输出小了 n_groups 倍, 计算attention时，repeat_kv，一个当 n_groups 个使。

同样架构，有没有使用GQA，模型参数量对比：
qwen1.5 32B:  29.6B vs 32B
qwen1.5 110B:  102B vs 111B 
```

模型参数量计算方法

```
def model_parameters(num_hidden_layers, hidden_size, intermediate_size, vocab_size, n_groups):
    layer_parameters = (  (hidden_size + 1) * hidden_size              # q_proj
                        + (hidden_size + 1) * hidden_size // n_groups  # k_proj
                        + (hidden_size + 1) * hidden_size // n_groups  # v_proj
                        + hidden_size       * hidden_size              # o_proj

                        + hidden_size * intermediate_size              # gate_proj
                        + hidden_size * intermediate_size              # up_proj
                        + intermediate_size * hidden_size              # down_proj
                       )
    lm_head_parameters = hidden_size * vocab_size
    return (num_hidden_layers * layer_parameters + lm_head_parameters)
```

Qwen1.5 家族参数量

| 模型大小 | vocab_size | hidden_size | num_hidden_layers | intermediate_size | n_groups | 模型（激活）参数量 |
|------|------------|-------------|-------------------|-------------------|----------|-----------|
| 0.5B | 151936     | 1024        | 24                | 2816              | 1        | 0.43B     |
| 1.8B | 151936     | 2048        | 24                | 5504              | 1        | 1.42B     |
| 4B   | 151936     | 2560        | 40                | 6912              | 1        | 3.32B     |
| 7B   | 151936     | 4096        | 32                | 11008             | 1        | 6.61B     |
| 14B  | 152064     | 5120        | 40                | 13696             | 1        | 12.6B     |
| 32B  | 152064     | 5120        | 64                | 27392             | 5        | 29.6B     |
| 72B  | 152064     | 8192        | 80                | 24576             | 1        | 66B       |
| 110B | 152064     | 8192        | 80                | 49152             | 8        | 102B      |


## 3.2. KV缓存参数量计算

每DecoderLayer层都会输出key_states和value_states, 
每个key_states, value_states 大小为 num_key_value_heads * head_dim = hidden_size // n_groups

KV缓存计算方法
```
def kv_cache_parameters(num_hidden_layers, hidden_size, n_groups):
    layer_parameters = (
        hidden_size // n_groups +   # key_states
        hidden_size // n_groups     # value_states
    )

    return num_hidden_layers * layer_parameters
```

每个词KV缓存大小

| 模型大小 | hidden_size | num_hidden_layers | n_groups | KV缓存参数量（每词） |
|------|-------------|-------------------|----------|-------------|
| 0.5B | 1024        | 24                | 1        | 0.046875M   |
| 1.8B | 2048        | 24                | 1        | 0.09375M    |
| 4B   | 2560        | 40                | 1        | 0.1953125M  |
| 7B   | 4096        | 32                | 1        | 0.25M       |
| 14B  | 5120        | 40                | 1        | 0.390625M   |
| 32B  | 5120        | 64                | 5        | 0.125M      |
| 72B  | 8192        | 80                | 1        | 1.25M       |
| 110B | 8192        | 80                | 8        | 0.15625M    |


注意到

32B、110B KV缓存占用比 4B 模型还小，GQA确实可以有效减少KV缓存大小


## 3.3. 4090 显存占用分析

如果浮点数用bfloat16格式占用16bits显存，24G显存的显卡加载了模型后，理论上还能有多少token的kv cache,
也就是还能生成多少个token呢？

更进一步，如果使用量化技术，int8、int4存储呢？

更进一步，模型参数、KV缓存使用不同的量化技术呢？

```
def kv_cache_len(parameters, kv_cache_parameters, memory_size, w_size, a_size):
    return (memory_size - parameters * w_size / 8) / (kv_cache_parameters * a_size / 8)
```

| 模型大小 | 模型参数量 | kv cache 参数量（每词） | w16a16 | w8a16  | w4a16  | w6a16  | w8a8   | w4a8   | 
|------|-------|------------------|--------|--------|--------|--------|--------|--------|
| 0.5B | 0.43B | 0.046875M        | 252705 | 257424 | 259784 | 258604 | 514849 | 519568 |
| 1.8B | 1.42B | 0.09375M         | 115553 | 123312 | 127192 | 125252 | 246625 | 254384 |
| 4B   | 3.32B | 0.1953125M       | 45525  | 54220  | 58567  | 56393  | 108440 | 117134 |
| 7B   | 6.61B | 0.25M            | 22072  | 35612  | 42382  | 38997  | 71224  | 84764  |
| 14B  | 12.6B | 0.390625M        | oom    | 15114  | 23285  | 19199  | 30228  | 46571  |
| 32B  | 29.6B | 0.125M           | oom    | oom    | 37778  | 7515   | oom    | 75556  |
| 72B  | 66B   | 1.25M            | oom    | oom    | oom    | oom    | oom    | oom    |
| 110B | 102B  | 0.15625M         | oom    | oom    | oom    | oom    | oom    | oom    |

表里面w表示模型参数占用bits，a表示 kv cache 一个 token 占用bits。

1. 感谢 w8a16 让 4090 用上 14B 的模型，感谢 w4a16 让 4090 用上 32B 的模型。
2. GQA确实可以有效减少KV缓存大小，能生成更多的token。
3. 其他量化技术比如很有性价比的 w6a16 方案，很有前途的 w4a8 方案。以及更多考量在 [量化]() 一节讨论。

# 4. 推理理论极限

# 4.1. 模型速度指标

计算量 FLOPs (Floating Point OPeration) 指的是输入一个token，模型完成一次前向传播所发生的浮点运算次数，即模型的时间复杂度，单位是 FLOPs。
下称作每词算力。

访存量MAC (Memory Access Cost) ：指的是输入一个token，模型完成一次前向传播所发生的内存交换总量，即模型的空间复杂度，单位是 Byte。
下称作每词带宽。

# 4.2 瓶颈：访存带宽

大语言模型计算主要由线性层 和 Scaled dot product attention 两种操作组成。

参考 [3.1 计算量与参数量的关联](https://zhuanlan.zhihu.com/p/624740065) 

> 我们可以近似认为：在一次前向传递中，对于每个token，每个模型参数，需要进行2次浮点数运算，即一次乘法法运算和一次加法运算。

参考 [LLM inference speed of light](http://arthurchiao.art/blog/llm-inference-speed-zh/)

> 以上两步计算有一个重要的共同特征：从矩阵或 KV-cache 读取的每个元素，只需要进行非常少量的浮点运算。
> - 矩阵-向量乘法对每个矩阵元素执行一次乘加运算（2 FLOPs）；
> - attention 对每个 key 执行一次乘加，对每个 value 执行一次乘加。

模型使用 FP16 作为矩阵元素的类型, 每词算力 FLOPs：每词带宽 MACs： 为 2 FLOPs : 2 Bit 

参考 [LLM inference speed of light](http://arthurchiao.art/blog/llm-inference-speed-zh/)

> 现代 CPU/GPU 的 ALU 操作（乘法、加法）内存 IO 速度要快得多。例如：
> - AMD Ryzen 7950X：67 GB/s 内存带宽和 2735 GFLOPS，Flop:byte = 40:1
> - NVIDIA GeForce RTX 4090：1008 GB/s 显存带宽和 83 TFLOPS，Flop:byte = 82:1
> - NVIDIA H100 SXM：3350 GB/s 内存带宽和 67 TFLOPS， 对于矩阵乘法，tensor core 提供 ~494 TFLOPS 稠密算力，Flop:byte = 147:1。
> 
> 对于 FP16/FP8 等精度较低的浮点数，比率更夸张：
> - H100 TensorCore 对于 dense FP8 矩阵的理论吞吐量为 1979 TFLOPS，FLOP:byte = 590:1。
> 
> 在这些场景中，无论是否使用 TensorCore 或使用什么浮点格式，ALU 都非常充足。
> 
> 因此，transformer 这种只需要对每个元素执行两次操作的场景，必定受到访存带宽的限制。


# Reference
[LLM inference speed of light](https://zeux.io/2024/03/15/llm-inference-sol/), [中文翻译](http://arthurchiao.art/blog/llm-inference-speed-zh/)

[分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065) 

[高性能 LLM 推理框架的设计与实现](https://mp.weixin.qq.com/s/ADsaNBbGoA7zx6YdmOMsyA)


