# 大语言模型推理理论极限和实际测试一， 单用户独占实时交互

# TL;DR
1. 限定本文讨论单用户独占实时交互的大语言模型推理场景，区分于训练阶段，多用户实时交互推理和离线批处理。这种场景下，最重要的指标是延迟（Latency），包括首字延迟（First Token Latency）和生成延迟（Latency）。
2. 通过分析大语言模型的推理机制的时间和空间复杂度，计算出大语言模型的计算量 FLOPs 和访存量 MACs。
3. 分别计算 Prefill 阶段和 Decoding 阶段推理理论极限
4. 讨论 GQA、量化、投机采样、对推理理论极限的影响
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

下面以 Qwen1.5 家族为例，有八种尺寸 0.5B, 1.8B, 4B, 7B, 14B, 32B, 72B, 110B，方便纵向对比。


# 2. 推理机制
> 其中 32B 和 110B 使用了 GQA，计算稍有改变，会在 [GQA](#51-%E5%88%86%E7%BB%84%E6%9F%A5%E8%AF%A2%E6%B3%A8%E6%84%8F%E5%8A%9B-gqa) 一节详细讨论。
> 这章先忽略这两个模型。

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
> 3. Embedding 每个token只取其中一条，一般不计入"激活参数量"， 总参数量包含 Embedding


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

self_attn 模块 q_proj, k_proj, v_proj带bias， o_proj不带 
mlp 模块 gate_proj, up_proj, down_proj都不带bias

一层 DecoderLayer Attention 参数量 = 1 * hidden_size * hidden_size + 3 * (hidden_size + 1) * hidden_size  
                                 = 1 * 1024 * 1024 + 3 * (1024+1) * 1024 = 4M
一层 DecoderLayer MLP 参数量 = 3 * hidden_size * intermediate_size = 3 * 1024 * 2816 = 8.25M

DecoderLayers 一共参数量 = num_hidden_layers X (Attention 参数量 + MLP 参数量) = 24 * (4M + 8.25M) = 294 M
lm_head 参数量 = hidden_size * vocab_size = 1024 * 151936 = 148.375 M

0.5B 总参数量 = 294 M + 148.375 M = 442.375 M = 0.432 B
```


模型参数量计算方法
```
def model_parameters(num_hidden_layers, hidden_size, intermediate_size, vocab_size):
    layer_parameters = (  (hidden_size + 1) * hidden_size              # q_proj
                        + (hidden_size + 1) * hidden_size              # k_proj
                        + (hidden_size + 1) * hidden_size              # v_proj
                        + hidden_size       * hidden_size              # o_proj

                        + hidden_size * intermediate_size              # gate_proj
                        + hidden_size * intermediate_size              # up_proj
                        + intermediate_size * hidden_size              # down_proj
                       )

    Lp = num_hidden_layers * layer_parameters                          # DecoderLayers 一共参数量
    Hp = hidden_size * vocab_size                                      # lm_head 参数量
    Ep = hidden_size * vocab_size                                      # Embedding 参数量
    Ap = Lp + Hp                                                       # 激活参数量 = DecoderLayers 一共参数量 + lm_head 参数量
    Tp = Ap + Ep                                                       # 总参数量 = 激活参数量 + Embedding 参数量
    return Lp, Hp, Ap, Tp
```

Qwen1.5 家族参数量

| 模型大小 | vocab_size | hidden_size | num_hidden_layers | intermediate_size | DecoderLayers 参数量 | lm_head 参数量 | 激活参数量  | 总参数量   |
|------|------------|-------------|-------------------|-------------------|-------------------|-------------|--------|--------|
| 0.5B | 151936     | 1024        | 24                | 2816              | 0.29B             | 0.14B       | 0.43B  | 0.58B  |
| 1.8B | 151936     | 2048        | 24                | 5504              | 1.13B             | 0.29B       | 1.42B  | 1.71B  |
| 4B   | 151936     | 2560        | 40                | 6912              | 2.95B             | 0.36B       | 3.32B  | 3.68B  |
| 7B   | 151936     | 4096        | 32                | 11008             | 6.03B             | 0.58B       | 6.61B  | 7.19B  |
| 14B  | 152064     | 5120        | 40                | 13696             | 11.74B            | 0.73B       | 12.47B | 13.19B |
| 72B  | 152064     | 8192        | 80                | 24576             | 65.00B            | 1.16B       | 66.16B | 67.32B |

可以看到
- 小模型 “lm_head 参数量” 占比比较大，0.5B占总参数的30%以上，而 72B 占比不到2%。

## 3.2. KV缓存参数量计算

每 DecoderLayer 层都会输出 key_states 和 value_states

每个 key_states, value_states 大小为 hidden_size

所以，KV缓存计算方法
```
def kv_cache_parameters(num_hidden_layers, hidden_size):
    layer_parameters = (
        hidden_size +              # key_states
        hidden_size                # value_states
    )

    return num_hidden_layers * layer_parameters
```

每个词KV缓存大小

| 模型大小 | hidden_size | num_hidden_layers | KV缓存参数量（每词） |
|------|-------------|-------------------|-------------|
| 0.5B | 1024        | 24                | 0.046875M   |
| 1.8B | 2048        | 24                | 0.09375M    |
| 4B   | 2560        | 40                | 0.1953125M  |
| 7B   | 4096        | 32                | 0.25M       |
| 14B  | 5120        | 40                | 0.390625M   |
| 72B  | 8192        | 80                | 1.25M       |


## 3.3. 4090 显存占用分析

如果浮点数用bfloat16格式占用16bits显存，24G显存的显卡加载了模型后，
理论上还能有多少token的kv cache, 也就是还能生成多少个token呢？

```
def kv_cache_len(parameters, kv_cache_parameters, memory_size, w_size, a_size):
    return (memory_size - parameters * w_size / 8) / (kv_cache_parameters * a_size / 8)
```

| 模型大小 | 激活参数量 | kv cache 参数量（每词） | w16kv16 | w8kv16 | w4kv16 | 
|------|-------|------------------|---------|--------|--------|
| 0.5B | 0.43B | 0.046875M        | 252705  | 257424 | 259784 |
| 1.8B | 1.42B | 0.09375M         | 115553  | 123312 | 127192 |
| 4B   | 3.32B | 0.1953125M       | 45525   | 54220  | 58567  |
| 7B   | 6.61B | 0.25M            | 22072   | 35612  | 42382  |
| 14B  | 12.6B | 0.390625M        | oom     | 15114  | 23285  |
| 72B  | 66B   | 1.25M            | oom     | oom    | oom    |

表里面w表示模型参数占用bits，kv 表示 kv cache 一个 token 占用bits。

可以看到
- 模型参数使用16bits表示，4090 显卡甚至没法加载14B模型。降低浮点数表示占用，比如使用 8bits 甚至 4bits 才行。
- 更多信息将在 [量化](#52-%E9%87%8F%E5%8C%96-quantize) 一节讨论。

# 4. 推理理论极限

## 4.1. 模型速度指标

计算量 FLOPs (Floating Point OPeration) 指的是输入一个token，模型完成一次前向传播所发生的浮点运算次数，即模型的时间复杂度，单位是 FLOPs。
下称作每词算力。

访存量 MACs (Memory Access Cost) ：指的是输入一个token，模型完成一次前向传播所发生的内存交换总量，即模型的空间复杂度，单位是 Byte。
下称作每词带宽。

## 4.2 瓶颈：访存带宽

大语言模型计算主要由线性层 和 Scaled dot product attention 两种操作组成。

参考 [3.1 计算量与参数量的关联](https://zhuanlan.zhihu.com/p/624740065) 

> 我们可以近似认为：在一次前向传递中，对于每个token，每个模型参数，需要进行2次浮点数运算，即一次乘法法运算和一次加法运算。

参考 [LLM inference speed of light](http://arthurchiao.art/blog/llm-inference-speed-zh/)

> 以上两步计算有一个重要的共同特征：从矩阵或 KV-cache 读取的每个元素，只需要进行非常少量的浮点运算。
> - 矩阵-向量乘法对每个矩阵元素执行一次乘加运算（2 FLOPs）；
> - attention 对每个 key 执行一次乘加，对每个 value 执行一次乘加。

模型使用 FP16 作为矩阵元素的类型, 每词算力 FLOPs：每词带宽 MACs： 为 2 FLOPs : 2 Bytes

参考 [LLM inference speed of light](http://arthurchiao.art/blog/llm-inference-speed-zh/)

> 现代 CPU/GPU 的 ALU 操作（乘法、加法）内存 IO 速度要快得多。例如：
> - AMD Ryzen 7950X：67 GB/s 内存带宽和 2735 GFLOPS，Flop:byte = 40:1
> - NVIDIA GeForce RTX 4090：1008 GB/s 显存带宽和 83 TFLOPS，Flop:byte = 82:1
> - NVIDIA H100 SXM：3350 GB/s 内存带宽和 67 TFLOPS， 对于矩阵乘法，tensor core 提供 ~494 TFLOPS 稠密算力（FP32），Flop:byte = 147:1。
> 
> 对于 FP16/FP8 等精度较低的浮点数，比率更夸张：
> - H100 TensorCore 对于 dense FP8 矩阵的理论吞吐量为 1979 TFLOPS，FLOP:byte = 590:1。
> 
> 在这些场景中，无论是否使用 TensorCore 或使用什么浮点格式，ALU 都非常充足。
> 
> 因此，transformer 这种只需要对每个元素执行两次操作的场景，必定受到访存带宽的限制。

## 4.3. 预填充 (Prefill) 阶段 推理理论极限

预填充 (Prefill) 阶段对应于输出第一个词的的，需要将 prompt_tokens 中间结果都写入 kv cache 里，才能推理得到第一个词。
这个阶段是对kv cache进行预填充。

有两种填充方式
1. 外层循环按词方向，内层循环按层方向计算，一个词一个词填充，每个词都需要将模型加载一遍。
2. 外层循环按层方向，内层循环按词方向计算，一个层一个层填充，每层参数只需要加载一次，也就是预填充阶段只需要加载一遍模型。

很明显第二种填充方式速度快一些，下面只分析第二种填充方式。
```
脑子里快速过一遍第二种填充方式：
外层循环执行Embedding，内层循环按词方向计算，每个词得到一个 embedding，没有任何问题
外层循环执行第一个 DecoderLayer
    k_proj 读取参数，内层循环按词方向计算， 得到 key_states 写入 kv cache，没有任何问题
    v_proj 读取参数，内层循环按词方向计算， 得到 value_states 写入 kv cache，没有任何问题
    
    q_proj 读取参数，内层循环按词方向计算， 得到 query_states，没有任何问题
        (这里多说一句，这样层并行 query_states 就得写在某个临时的地方，以后还要用一次，但不像kv一样需要写在永久的地方。
    
    做 sdpa (scaled_dot_product_attention)，内层循环按词方向计算，
        每个词 从 kv cache 里面读取之前词的 key_states，value_states， 然后用刚才得到的 query_states
  
        等一下 
            1. 执行的时候一定要确认 key_states，value_states 都已经写入完成，避免脏读（Dirty Read）
              （一个事务读取了另一个未提交事务中的数据）
            2. q_proj 放在 做 scaled_dot_product_attention 里面，query_states 就不需要找地方存，但 q_proj 每个词都得加载一遍。
               这就要权衡 hidden_size 大 q_proj 要大一些; prompt_tokens 长度长，query_states要大一些 
               所以超长的 prompt_tokens 需要进一步分析，现在还是按 hidden_size >>  prompt_tokens 长度
        
        只要写入到kv cache正确，现在就可以从kv cache读取出来计算，没有任何问题
        
    o_proj, 后面的 mlp， 都是 线性层 参数之间没有依赖 也不会有什么问题。
外层循环执行完所有的 DecoderLayer，没有任何问题
最后只需要对最后一个token 的 hidden_states 执行 lm_head， 得到 next_token_logits， 采样下一个词。

所以第二种填充方式的重点是写入 kv cache，避免 sdpa 脏读（Dirty Read）。这也是预填充 (Prefill) 的名字由来。
```

预填充 (Prefill) 阶段 推理理论极限：
- 读取延迟 = （模型参数 * 1 * 模型每浮点占用大小 +（Σ kv cache参数）* kv cache每浮点占用大小) / 显卡带宽
- 计算延迟 粗略 = （模型参数 * prompt_tokens 长度 + Σ kv cache参数）* 每参数两次浮点运算 / 显卡算力
- 计算延迟 精细 = （DecoderLayers 参数量 *  + Σ kv cache参数 + lm_head 参数量 * 1）* 每参数两次浮点运算 / 显卡算力
- 首字延迟 First Token Latency = max(读取延迟，计算延迟)

> 计算延迟 粗略 和 计算延迟 精细 的区别是，lm_head算一次还是算多次。有的llm库没有相关优化，会算多次，下面可以看到对于小模型，差距还是比较大，

```
def prefill_first_token_latency_roughly(Mp, KVp, w_size, a_size, n, n_groups, bandwidth, computing_power):
    read_latency = ((Mp * 1) * w_size + 0.5 * n * (n+1) * KVp * a_size) / bandwidth * 1000
    computing_latency = (Mp * n + 0.5 * n * (n+1) * KVp * n_groups) * 2 / computing_power * 1000
    return read_latency, computing_latency


def prefill_first_token_latency_exactly(Lp, Hp, KVp, w_size, a_size, n, n_groups, bandwidth, computing_power):
    read_latency = (((Lp + Hp) * 1) * w_size + 0.5 * n * (n+1) * KVp * a_size) / bandwidth * 1000
    computing_latency = (Lp * n + 0.5 * n * (n+1) * KVp * n_groups + Hp) * 2 / computing_power * 1000
    return read_latency, computing_latency
```

因为 kv cache 读取次数是自然数列求和，公式是0.5 * n * (n+1) 的 二次函数，所以 read_latency 和 computing_latency 都是 prompt_tokens 长度 的二次函数。

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/static/prefill-w16kv16.png?raw=true" width="600">
<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/static/prefill-w8kv16.png?raw=true" width="600">
<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/static/prefill-w4kv16.png?raw=true" width="600">

可以看到 读取延迟 和 计算延迟 曲线可能有交点，标志这从 带宽瓶颈到算力瓶颈的拐点，这个拐点可以算出来：

| 模型大小 | 拐点 w16kv16 粗略 | 拐点 w16kv16 精细 | 拐点 w8kv16 粗略 | 拐点 w8kv16 精细 | 拐点 w4kv16 精细 | 拐点 w4kv16 精细 |
|------|---------------|---------------|--------------|--------------|--------------|--------------|
| 0.5B | 无 \ 无         | 无 \ 无         | 55 \ 170     | 无 \ 无        | 23 \ 203     | 44 \ 106     |
| 1.8B | 127 \ 245     | 无 \ 无         | 48 \ 325     | 68 \ 228     | 22 \ 351     | 29 \ 268     |
| 4B   | 116 \ 302     | 无 \ 无         | 47 \ 371     | 55 \ 317     | 22 \ 396     | 25 \ 347     |
| 7B   | 99 \ 553      | 113 \ 481     | 45 \ 607     | 50 \ 544     | 21 \ 630     | 23 \ 571     |
| 14B  | 95 \ 692      | 103 \ 638     | 44 \ 743     | 47 \ 694     | 21 \ 766     | 22 \ 718     |
| 72B  | 90 \ 1216     | 92 \ 1191     | 43 \ 1263    | 44 \ 1239    | 21 \ 1285    | 21 \ 1262    |

可以看到有两个拐点
- 第一个拐点，加载模型瓶颈 切换到 算力瓶颈
- 第二个拐点，算力瓶颈 切换到 加载 kv cache瓶颈

> 在第一个拐点 加载模型瓶颈 切换到 算力瓶颈 之前，增加提示词长度，延迟基本不变，瓶颈在加载模型上；将一次 Prefill 完成的工作分成两次，相当于模型加载了两次，延迟也变成两倍。所以这个序列长度不要切分，一次性完成。
> 
> 在第一个拐点之后，只要切分的小工作大于拐点，没有增加计算和kv cache读取，Prefill 延迟不变。所以可以将 Prefill 工作切分成第一个拐点大小的多份，总Prefill延迟不变。**这个结论非常重要**


## 4.4. 解码 （Decoding） 阶段 推理理论极限
解码 （Decoding） 阶段对应于输出之后的 token 的输出。

kv cache已经填入之前 token。 推理过程就是从下往上过一遍模型，最后从 lm_head 输出 logits 采样。

当语言模型生成文本时，它是逐个 token 进行的，后一个词依赖前一个词，没有并行性的可能性，也玩不出什么花样。

> [Speculative Decoding](https://arxiv.org/abs/2211.17192) [后面会讨论](#53-%E6%8A%95%E6%9C%BA%E9%87%87%E6%A0%B7-speculative-decoding)，暂时先忽略

Decoding 阶段 推理理论极限：
- 读取延迟 = （模型参数 * 模型每浮点占用大小 + kv cache参数 * kv cache每浮点占用大小) / 显卡带宽
- 计算延迟 = （模型参数 + kv cache 参数）* 每参数两次浮点运算 / 显卡算力
- Latency = max(读取延迟，计算延迟) = 读取延迟

变成更有物理意义的的参数形式

Latency = (n-1)/每过多少词输出延迟增加1ms + 第一个词的输出延迟
        = (n-1)/W + B

以 4090 为例 带宽 1008 GB/s，算力 82.58 TFLOPS

| 模型大小 | w16kv16 W | w16kv16 B | w8kv16 W | w8kv16 B | w4kv16 W | w4kv16 B | 
|------|-----------|-----------|----------|----------|----------|----------|
| 0.5B | 11010     | 0.86ms    | 11010    | 0.43ms   | 11010    | 0.21ms   |
| 1.8B | 5505      | 2.82ms    | 5505     | 1.41ms   | 5505     | 0.70ms   |
| 4B   | 2642      | 6.58ms    | 2642     | 3.29ms   | 2642     | 1.65ms   |
| 7B   | 2064      | 13.12ms   | 2064     | 6.56ms   | 2064     | 3.28ms   |
| 14B  | 1321      | 24.74ms   | 1321     | 12.37ms  | 1321     | 6.19ms   |
| 72B  | 413       | 131.28ms  | 413      | 65.64ms  | 413      | 32.82ms  |

可以看到
- W （每过多少词输出延迟增加1ms） 只跟 kv cache参数 * kv cache每浮点占用大小有关
- B （第一个词的输出延迟）（忽略第一个kv cache） 只跟 模型参数 * 模型每浮点占用大小 
- B 跟 预填充 (Prefill) 阶段 prompt_tokens 长度 = 1 相同
- 估算一下，比如7B w8kv16， 1w token时输出一个token的延迟
```
  Latency = (1/W)(n-1) + B 
          ≈ (1w / 2064)ms + 6.56ms  
          ≈ 6.56ms + 5ms 
          ≈ 11.56ms
```

# 5. 一些速度优化方法

## 5.1. 分组查询注意力 (GQA) 
[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/static/gqa.png?raw=true" width="600">

qwen1.5 32B 和 110B 使用了 GQA

```
n_groups = num_attention_heads // num_key_value_heads

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
```

- 访存量 MACs 减少
    - 可以看到 k_proj 和 v_proj 输出小了 n_groups 倍
    - 也意味着 kv cache 减少 n_groups 倍，相同的显存容量，可以做更长的输出
    - 也意味着 "每过多少词输出延迟增加1ms" 变大，或者推理延迟增加变缓
- 计算量 FLOPs 稍微增加
    - 计算 attention 时，repeat_kv，一个当 n_groups 个使，执行 n_groups 次乘加运算（2 FLOPs） 。
    - 稍微改善 FLOPs：MACs， 毕竟 ALU 非常充足。
  
### 5.1.1. 模型参数量计算方法, 考虑 GQA

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

    Lp = num_hidden_layers * layer_parameters                          # DecoderLayers 一共参数量
    Hp = hidden_size * vocab_size                                      # lm_head 参数量
    Ep = hidden_size * vocab_size                                      # Embedding 参数量
    Ap = Lp + Hp                                                       # 激活参数量 = DecoderLayers 一共参数量 + lm_head 参数量
    Tp = Ap + Ep                                                       # 总参数量 = 激活参数量 + Embedding 参数量
    return Lp, Hp, Ap, Tp
```
Qwen1.5 家族参数量

| 模型大小       | vocab_size | hidden_size | num_hidden_layers | intermediate_size | n_groups | DecoderLayers 参数量 | lm_head 参数量 | 激活参数量   | 总参数量    |
|------------|------------|-------------|-------------------|-------------------|----------|-------------------|-------------|---------|---------|
| 0.5B       | 151936     | 1024        | 24                | 2816              | 1        | 0.29B             | 0.14B       | 0.43B   | 0.58B   |
| 1.8B       | 151936     | 2048        | 24                | 5504              | 1        | 1.13B             | 0.29B       | 1.42B   | 1.71B   |
| 4B         | 151936     | 2560        | 40                | 6912              | 1        | 2.95B             | 0.36B       | 3.32B   | 3.68B   |
| 7B         | 151936     | 4096        | 32                | 11008             | 1        | 6.03B             | 0.58B       | 6.61B   | 7.19B   |
| 14B        | 152064     | 5120        | 40                | 13696             | 1        | 11.74B            | 0.73B       | 12.47B  | 13.19B  |
| 32B(无GQA)  | 152064     | 5120        | 64                | 27392             | 1        | 31.33B            | 0.73B       | 32.05B  | 32.78B  |
| 32B        | 152064     | 5120        | 64                | 27392             | 5        | 28.83B            | 0.73B       | 29.55B  | 30.28B  |
| 72B        | 152064     | 8192        | 80                | 24576             | 1        | 65.00B            | 1.16B       | 66.16B  | 67.32B  |
| 110B(无GQA) | 152064     | 8192        | 80                | 24576             | 1        | 110.00B           | 1.16B       | 111.16B | 112.32B |
| 110B       | 152064     | 8192        | 80                | 49152             | 8        | 101.25B           | 1.16B       | 102.41B | 103.57B |


可以看到
- 使用 GQA 模型总参数小了一些

### 5.1.2. 每个词KV缓存大小, 考虑 GQA

```
def kv_cache_parameters(num_hidden_layers, hidden_size, n_groups):
    layer_parameters = (
        hidden_size // n_groups +   # key_states
        hidden_size // n_groups     # value_states
    )
    return num_hidden_layers * layer_parameters
```

| 模型大小       | hidden_size | num_hidden_layers | n_groups | KV缓存参数量（每词） |
|------------|-------------|-------------------|----------|-------------|
| 0.5B       | 1024        | 24                | 1        | 0.046875M   |
| 1.8B       | 2048        | 24                | 1        | 0.09375M    |
| 4B         | 2560        | 40                | 1        | 0.1953125M  |
| 7B         | 4096        | 32                | 1        | 0.25M       |
| 14B        | 5120        | 40                | 1        | 0.390625M   |
| 32B(无GQA)  | 5120        | 64                | 1        | 0.625 M     |
| 32B        | 5120        | 64                | 5        | 0.125M      |
| 72B        | 8192        | 80                | 1        | 1.25M       |
| 110B(无GQA) | 8192        | 80                | 1        | 1.25 M      |
| 110B       | 8192        | 80                | 8        | 0.125M      |

可以看到 
- GQA确实可以有效减少KV缓存大小
- 32B、110B KV缓存占用比 4B 模型还小

### 5.1.3. 4090 显存占用分析

| 模型大小       | 激活参数量 | kv cache 参数量（每词） | w16kv16 | w8kv16 | w4kv16 | w6kv16 | w4kv8  | 
|------------|-------|------------------|---------|--------|--------|--------|--------|
| 0.5B       | 0.43B | 0.046875M        | 252705  | 257424 | 259784 | 258604 | 519568 |
| 1.8B       | 1.42B | 0.09375M         | 115553  | 123312 | 127192 | 125252 | 254384 |
| 4B         | 3.32B | 0.1953125M       | 45525   | 54220  | 58567  | 56393  | 117134 |
| 7B         | 6.61B | 0.25M            | 22072   | 35612  | 42382  | 38997  | 84764  |
| 14B        | 12.6B | 0.390625M        | oom     | 15114  | 23285  | 19199  | 46571  |
| 32B(无GQA)  | 29.6B | 0.625M           | oom     | oom    | 6531   | oom    | 13062  |
| 32B        | 29.6B | 0.125M           | oom     | oom    | 37778  | 7515   | 75556  |
| 72B        | 66B   | 1.25M            | oom     | oom    | oom    | oom    | oom    |
| 110B(无GQA) | 102B  | 1.25 M           | oom     | oom    | oom    | oom    | oom    |
| 110B       | 102B  | 0.125M           | oom     | oom    | oom    | oom    | oom    |

可以看到
- GQA确实可以有效减少KV缓存大小，能生成更多的token。
- 感谢 w4kv16 让 4090 用上 32B 的模型。
- 其他量化技术比如很有性价比的 w6kv16 方案，很有前途的 w4kv8 方案。以及更多考量在 [量化](#52-%E9%87%8F%E5%8C%96-quantize) 一节讨论。

### 5.1.4. 预填充 (Prefill) 阶段 带宽瓶颈到算力瓶颈拐点

| 模型大小       | 拐点 w16kv16 粗略 | 拐点 w16kv16 精细 | 拐点 w8kv16 粗略 | 拐点 w8kv16 精细 | 拐点 w4kv16 精细 | 拐点 w4kv16 精细 |
|------------|---------------|---------------|--------------|--------------|--------------|--------------|
| 0.5B       | 无 \ 无         | 无 \ 无         | 55 \ 170     | 无 \ 无        | 23 \ 203     | 44 \ 106     |
| 1.8B       | 127 \ 245     | 无 \ 无         | 48 \ 325     | 68 \ 228     | 22 \ 351     | 29 \ 268     |
| 4B         | 116 \ 302     | 无 \ 无         | 47 \ 371     | 55 \ 317     | 22 \ 396     | 25 \ 347     |
| 7B         | 99 \ 553      | 113 \ 481     | 45 \ 607     | 50 \ 544     | 21 \ 630     | 23 \ 571     |
| 14B        | 95 \ 692      | 103 \ 638     | 44 \ 743     | 47 \ 694     | 21 \ 766     | 22 \ 718     |
| 32B(无GQA)  | 90 \ 1175     | 92 \ 1144     | 43 \ 1222    | 44 \ 1192    | 21 \ 1244    | 21 \ 1215    |
| 32B        | 85 \ 6051     | 87 \ 5898     | 42 \ 6094    | 43 \ 5942    | 21 \ 6115    | 21 \ 5964    |
| 72B        | 90 \ 1216     | 92 \ 1191     | 43 \ 1263    | 44 \ 1239    | 21 \ 1285    | 21 \ 1262    |
| 110B(无GQA) | 87 \ 2108     | 88 \ 2084     | 42 \ 2153    | 43 \ 2130    | 21 \ 2175    | 21 \ 2151    |
| 110B       | 84 \ 17602    | 85 \ 17400    | 42 \ 17644   | 42 \ 17443   | 20 \ 17665   | 21 \ 17464   |

可以看到
- 第一个拐点，加载模型瓶颈 切换到 算力瓶颈 差别不大，比较GQA参数占总参数占比很小
- 第二个拐点，算力瓶颈 切换到 加载 kv cache 差别很大，GQA确实可以有效减少KV缓存大小

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/static/prefill-w16kv16-gqa.png?raw=true" width="600">
<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/static/prefill-w8kv16-gqa.png?raw=true" width="600">
<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/static/prefill-w4kv16-gqa.png?raw=true" width="600">

### 5.1.5. 解码 （Decoding） 阶段 推理理论极限

| 模型大小       | w16kv16 W | w16kv16 B | w8kv16 W | w8kv16 B | w4kv16 W | w4kv16 B | 
|------------|-----------|-----------|----------|----------|----------|----------|
| 0.5B       | 11010     | 0.86ms    | 11010    | 0.43ms   | 11010    | 0.21ms   |
| 1.8B       | 5505      | 2.82ms    | 5505     | 1.41ms   | 5505     | 0.70ms   |
| 4B         | 2642      | 6.58ms    | 2642     | 3.29ms   | 2642     | 1.65ms   |
| 7B         | 2064      | 13.12ms   | 2064     | 6.56ms   | 2064     | 3.28ms   |
| 14B        | 1321      | 24.74ms   | 1321     | 12.37ms  | 1321     | 6.19ms   |
| 32B(无GQA)  | 826       | 63.60ms   | 826      | 31.80ms  | 826      | 15.90ms  |
| 32B        | 4129      | 58.64ms   | 4129     | 29.32ms  | 4129     | 14.66ms  |
| 72B        | 413       | 131.28ms  | 413      | 65.64ms  | 413      | 32.82ms  |
| 110B(无GQA) | 413       | 220.56ms  | 413      | 110.28ms | 413      | 55.14ms  |
| 110B       | 3303      | 203.20ms  | 3303     | 101.60ms | 3303     | 50.80ms  |

可以看到
- 第一个词的输出延迟 变小
- "每过多少词输出延迟增加1ms" 大幅变大，或者推理延迟增加大幅变缓

### 5.1.6. 总结 
- 虽然引入 GQA 可能会导致模型质量有下降，但对于推理延迟和长上下文收益太大了。
- GQA 让 4090 用上 32B 的模型，使得 Qwen1.5 32B 是家族里面性价比最高的模型。

## 5.2 量化 (Quantize)
模型量化 Quantize 是指将神经网络的浮点转换为定点，减少模型空间占用。

收益也非常明显，降低浮点占用意味着成倍的提高响应时间，增加上下文长度。
对于访存带宽限制的llm，只要精巧的写 cuda kernel 将计算阶段量化和反量化融合进去，总可以被访存时间 cover 住。
剩下就是精度和速度的权衡。 

主流的量化方法，比如 qwen 1.5 官方提供了 [AWQ](https://arxiv.org/abs/2306.00978) 和 [GPTQ](https://arxiv.org/abs/2210.17323) (int8、int4) 量化模型。以及 GGUF 格式的8种量化模型。


参考 [Qwen1.5 官方](https://huggingface.co/Qwen/Qwen1.5-4B-Chat-GGUF) 给出的量化模型效果:

To demonstrate their model quality, we follow [`llama.cpp`](https://github.com/ggerganov/llama.cpp) to evaluate their perplexity on wiki test set. Results are shown below:

| Size | fp16  | q8_0  | q6_k  | q5_k_m | q5_0  | q4_k_m | q4_0  | q3_k_m | q2_k  |
|------|-------|-------|-------|--------|-------|--------|-------|--------|-------|
| 0.5B | 34.20 | 34.22 | 34.31 | 33.80  | 34.02 | 34.27  | 36.74 | 38.25  | 62.14 |
| 1.8B | 15.99 | 15.99 | 15.99 | 16.09  | 16.01 | 16.22  | 16.54 | 17.03  | 19.99 |
| 4B   | 13.20 | 13.21 | 13.28 | 13.24  | 13.27 | 13.61  | 13.44 | 13.67  | 15.65 |
| 7B   | 14.21 | 14.24 | 14.35 | 14.32  | 14.12 | 14.35  | 14.47 | 15.11  | 16.57 |
| 14B  | 10.91 | 10.91 | 10.93 | 10.98  | 10.88 | 10.92  | 10.92 | 11.24  | 12.27 |
| 32B  | 8.87  | 8.89  | 8.91  | 8.94   | 8.93  | 8.96   | 9.17  | 9.14   | 10.51 |
| 72B  | 7.97  | 7.99  | 7.99  | 7.99   | 8.01  | 8.00   | 8.01  | 8.06   | 8.63  |

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/static/quantize.png?raw=true" width="600">

简单建议，6bit量化几乎无损，4bit量化是性价比较高的方案。更小尺寸的量化模型部署时要进行充分测试。

参考 [llama.cpp k-quants](https://github.com/ggerganov/llama.cpp/pull/1684)

> Perhaps worth noting is that the 6-bit quantized perplexity is within 0.1% or better from the original fp16 model.

参考 [DeepSpeed-FP6：大型语言模型中以FP6为核心的强大推理服务](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-fp6/03-05-2024/README-Chinese.md)
> FP6的突破...，该格式在各种任务的性能和灵活性方面均表现出色。...到了与FP16模型相当的结果。.....这一创新使FP6成为提高LLMs效率的有效途径。

参考 [How Good Are Low-bit Quantized LLaMA3 Models? An Empirical Study](https://arxiv.org/abs/2404.14047)

> Both of them (AWQ, QuIP) can keep LLAMA3’s capability at 3-bit and even push the 2-bit quantization to promising.

前沿的量化论文已经讨论[2bits](https://arxiv.org/abs/2404.12759)甚至[1-bit](https://arxiv.org/abs/2402.17764) [OneBit](https://arxiv.org/abs/2402.11295)。
除了模型可以量化，kv cache 也可以量化, 比如 [SmoothQuant](https://arxiv.org/abs/2211.10438)、[KVQuant](https://arxiv.org/abs/2401.18079)、[W4A8KV4](https://arxiv.org/abs/2405.04532)。

## 5.3 投机采样 (Speculative Decoding)

# 6. 实际推理速度测试

# 6.1. HuggingFace Transformers
### 6.1.1. bfloat16 模型

模型占用显存

| 模型                     | 理论总参数量  | 模型格式     | 理论模型占用显存 | 实际模型文件大小  | 显存实际占用  | 
|------------------------|---------|----------|----------|-----------|---------|
| Qwen/Qwen1.5-0.5B-Chat | 0.58B   | bfloat16 | 1.15GB   | 1.24 GB   | 1.07GB  |
| Qwen/Qwen1.5-1.8B-Chat | 1.71B   | bfloat16 | 3.42GB   | 3.67 GB   | 3.83GB  |
| Qwen/Qwen1.5-4B-Chat   | 3.68B   | bfloat16 | 7.36GB   | 7.9 GB    | 8.01GB  |
| Qwen/Qwen1.5-7B-Chat   | 7.19B   | bfloat16 | 14.38GB  | 15.45 GB  | 14.88GB |
| Qwen/Qwen1.5-14B-Chat  | 13.19B  | bfloat16 | 26.39GB  | 28.34 GB  | oom     |
| Qwen/Qwen1.5-32B-Chat  | 30.28B  | bfloat16 | 60.56GB  | 65.04 GB  | oom     |
| Qwen/Qwen1.5-72B-Chat  | 67.32B  | bfloat16 | 134.64GB | 144.47 GB | oom     |
| Qwen/Qwen1.5-110B-Chat | 103.57B | bfloat16 | 207.14GB | 222.39 GB | oom     |

### 6.1.2. 预填充 (Prefill) 阶段 推理速度 单位 ms

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/static/prefill1.png?raw=true" width="600">

| 模型                     | 实测25  | 实测100 | 实测200 | 实测300 | 实测400 | 实测500 |
|------------------------|-------|-------|-------|-------|-------|-------|
| Qwen/Qwen1.5-0.5B-Chat | 6.12  | 6.17  | 6.83  | 7.31  | 8.03  | 8.74  |
| Qwen/Qwen1.5-1.8B-Chat | 7.15  | 7.79  | 10    | 12.68 | 15.61 | 16.31 |
| Qwen/Qwen1.5-4B-Chat   | 13.6  | 14.56 | 20.29 | 28.76 | 35.82 | 37.52 |
| Qwen/Qwen1.5-7B-Chat   | 21.01 | 23.18 | 29.98 | 46.93 | 52.34 | 61.02 |

|                        | 实测25 | 实测100 | 实测200 | 实测300 | 实测400 | 实测500 |
|------------------------|------|-------|-------|-------|-------|-------|
| Qwen/Qwen1.5-0.5B-Chat | 0.26 | 1.03  | 2.07  | 3.11  | 4.17  | 5.25  |
| Qwen/Qwen1.5-1.8B-Chat | 0.84 | 3.37  | 6.76  | 10.18 | 13.62 | 17.07 |
| Qwen/Qwen1.5-4B-Chat   | 1.96 | 7.87  | 15.78 | 23.74 | 31.74 | 39.79 |
| Qwen/Qwen1.5-7B-Chat   | 3.91 | 15.67 | 31.39 | 47.17 | 63.01 | 78.91 |

|                        | 实测25 | 实测100 | 实测200 | 实测300 | 实测400 | 实测500 |
|------------------------|------|-------|-------|-------|-------|-------|
| Qwen/Qwen1.5-0.5B-Chat | 4%   | 17%   | 30%   | 43%   | 52%   | 60%   |
| Qwen/Qwen1.5-1.8B-Chat | 12%  | 43%   | 68%   | 80%   | 87%   | 105%  |
| Qwen/Qwen1.5-4B-Chat   | 14%  | 54%   | 78%   | 83%   | 89%   | 106%  |
| Qwen/Qwen1.5-7B-Chat   | 19%  | 68%   | 105%  | 101%  | 120%  | 129%  |

### 6.1.3. 解码 （Decoding） 阶段 推理速度 单位 ms

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/static/decoding1.png?raw=true" width="600">

理论速度

| 模型                     | 2048  | 4096  | 6144   | 8192   | 10240  | 12288  | 14336  | 16384  |
|------------------------|-------|-------|--------|--------|--------|--------|--------|--------|
| Qwen/Qwen1.5-0.5B-Chat | 2698  | 6185  | 10435  | 15446  | 21220  | 27755  | 35052  | 43111  |
| Qwen/Qwen1.5-1.8B-Chat | 7631  | 16868 | 27629  | 39914  | 53722  | 69054  | 85911  | 104290 |
| Qwen/Qwen1.5-4B-Chat   | 16361 | 36073 | 58959  | 85020  | 114256 | 146666 | 182251 | 221010 |
| Qwen/Qwen1.5-7B-Chat   | 30909 | 66225 | 105604 | 149047 | 196554 | 248124 | 303757 | 363454 |

实际速度

| 模型                     | 2048  | 4096  | 6144  | 8192   | 10240  | 12288  | 14336  | 16384  |
|------------------------|-------|-------|-------|--------|--------|--------|--------|--------|
| Qwen/Qwen1.5-0.5B-Chat | 12100 | 24265 | 36481 | 48889  | 62012  | 76313  | 91938  | 109417 |
| Qwen/Qwen1.5-1.8B-Chat | 13460 | 29493 | 47919 | 68936  | 93109  | 120300 | 150521 | 183629 |
| Qwen/Qwen1.5-4B-Chat   | 26769 | 58778 | 95816 | 139449 | 189314 | 245423 | 307444 | 375503 |
| Qwen/Qwen1.5-7B-Chat   | 40957 | 88720 | oom   | oom    | oom    | oom    | oom    | oom    |

速度比

| 模型                     | 2048 | 4096 | 6144 | 8192 | 10240 | 12288 | 14336 | 16384 |
|------------------------|------|------|------|------|-------|-------|-------|-------|
| Qwen/Qwen1.5-0.5B-Chat | 22%  | 25%  | 29%  | 32%  | 34%   | 36%   | 38%   | 39%   |
| Qwen/Qwen1.5-1.8B-Chat | 57%  | 57%  | 58%  | 58%  | 58%   | 57%   | 57%   | 57%   |
| Qwen/Qwen1.5-4B-Chat   | 61%  | 61%  | 62%  | 61%  | 60%   | 60%   | 59%   | 59%   |
| Qwen/Qwen1.5-7B-Chat   | 75%  | 75%  | oom  | oom  | oom   | oom   | oom   | oom   |

- 小模型跟理论差距大一些，大模型差距小一些


# 6.2. llama.cpp 库实际推理速度测试

# 7. 总结

# Reference
[LLM inference speed of light](https://zeux.io/2024/03/15/llm-inference-sol/), [中文翻译](http://arthurchiao.art/blog/llm-inference-speed-zh/)

[分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065) 

[高性能 LLM 推理框架的设计与实现](https://mp.weixin.qq.com/s/ADsaNBbGoA7zx6YdmOMsyA)


