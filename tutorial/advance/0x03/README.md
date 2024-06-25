# 大语言模型推理理论极限和实际测试二， 多用户实时交互

# TL;DR

# 1. 回顾
[上篇文章](https://github.com/noooop/zerollama/tree/main/tutorial/advance/0x02) 分析了单用户独占实时交互场景下的大语言模型推理理论极限和实际测试。

- 推理过程可以分成，预填充 (Prefill) 和解码 （Decoding） 两个阶段
- 预填充 (Prefill) 阶段对应于输出第一个词的的，需要将提示词每个词的中间结果都写入 kv cache 里，才能推理得到第一个词。
这个阶段是对 kv cache 进行预填充。
- 解码 （Decoding） 阶段对应于输出之后的 token 的输出。 kv cache 已经填入之前 token。 推理过程就是从下往上过一遍模型，最后从 lm_head 输出 logits 采样。
- 现代 CPU/GPU 的 ALU 操作（乘法、加法）内存 IO 速度要快得多，transformer 这种只需要对每个元素执行两次操作的场景，必定受到访存带宽的限制。
- 以 NVIDIA GeForce RTX 4090 为例，1008 GB/s 显存带宽和 83 TFLOPS，Flop:byte = 82:1，算力非常充足。
- 对于解码 （Decoding） 阶段，每输出一个词需要读一遍模型，读取模型的极限就是速度的极限
- 对于预填充 (Prefill) 阶段，读一遍模型计算提示词中的多个词，无论从 SIMD （Single Instruction Multiple Data） 的角度还是从带宽瓶颈 vs 算力瓶颈的角度都很划算
- 投机采样 (Speculative Decoding) 利用了 llm 批次执行速度比单次执行效率高，加速模型输出
- 文章最后，解码 (Decoding) 阶段是一种 长度为 1 的预填充 (Prefill) 阶段，将解码和预填充统一起来，在分析 CUDA kernels 的最优运行区间的时候非常有效。
- 这篇文章，讨论多用户实时交互，也就是：m 个用户解码 (Decoding) 阶段就相当于长度为 m 的预填充 (Prefill) 阶段。就要求 CUDA kernels 在典型用户数区间里面最优。（这句油门踩的是不是太快了，狗头

# 2. 24G 的 4090 能同时服务多少用户
llm模型推理的时候，每个 token 的中间结果，都要占用一小块显存空间，称为 kv cache。上篇文章 从理论分析到实际测试，24G 显存服务单用户都不太富裕。 

现在要服务多用户，本不富裕的显存又雪上加霜。所以先要搞清楚显存能服务用户的边界在哪里。

且听我从盘古开天地开始慢慢道来：

1. 随着大模型越来越卷，训练时上下文（原生上下文长度）也越来越长，例如llama 2k、llama2 4k、llama3 8k、qwen 2k、qwen2 32K.
2. 随着扩展上下文研究的不断发展，支持的长上下文模型越来越长，例如qwen1.5 32K, qwen2 128K、gpt-4/gpt-4o 128k、Claude 3 200k、[Llama-3 8B Gradient Instruct 1048k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k)。
3. 长上下文意味着：
   - 能处理更长的文本内容，比如摘要一整本书、一个长播客
   - 能输出更长的文本内容，比如输出一整个文档
   - Prompt Engineering，为了让大语言模型理解和解决复杂问题，需要用更多篇幅细致描述具体需求，在网上一千多字的Prompt并不少见
   - In Context Learning (ICL) 建议提示中包括多个相关示例，可以提高模型能力
   - Chain-of-Thought(CoT) 建议提示中包括一系列中间推理步骤，帮助大语言模型进行复杂推理
   - .....
4. 所以当下（2024年6月），自己用1K上下文略显寒酸，8K主流；商用起步32K，要有竞争力至少要跟gpt-4/gpt-4o 128k对齐。

> 更多长上下文相关内容请参考 [The What, Why, and How of Context Length Extension Techniques in Large Language Models -- A Detailed Survey](https://arxiv.org/abs/2401.07872)

5. 24G 的 4090 实际支持最大上下文，以静态分配 kv cache 的 llama.cpp 为例

| 模型   | w8kv16理论 | w4kv16理论 | q8_0   | q6_k   | q5_k_m | q5_0   | q4_k_m | q4_0   | q3_k_m | q2_k   |
|------|----------|----------|--------|--------|--------|--------|--------|--------|--------|--------|
| 0.5B | 257424   | 259784   | 180468 | 181236 | 181501 | 181501 | 181744 | 181744 | 182009 | 182253 |
| 1.8B | 123312   | 127192   | 98808  | 100082 | 100863 | 101109 | 101355 | 101627 | 102123 | 102646 |
| 4B   | 54220    | 58567    | 45565  | 47609  | 48376  | 48376  | 49131  | 49395  | 49901  | 50680  |
| 7B   | 35612    | 42382    | 28398  | 31469  | 32757  | 32757  | 33780  | 34290  | 35060  | 36344  |
| 14B  | 15114    | 23285    | 11250  | 14063  | 15846  | 16617  | 17395  | 18431  | 19186  | 20479  |
| 32B  | oom      | 37778    | oom    | oom    | 5370   | 7158   | 15086  | 18419  | 25852  | 36076  |

- 7B模型，加载4bit模型后，还有差不多3万 kv cache，1k上下文可以同时服务30个用户，8k上下文同时服务4个用户
- 14B模型，加载4bit模型后，还有差不多1.5万 kv cache，1k上下文可以同时服务15个用户，8k上下文同时服务2个用户
- 32B模型，加载4bit模型后，还有差不多1.5万 kv cache，1k上下文可以同时服务15个用户，8k上下文同时服务2个用户 (GQA 使得 32B 成为 qwen1.5 系列性价比最高的模型)

6. 能不能超卖，代价是什么
- 用户不可能每时每刻都打满最长的上下文，统计请求上下文长度，适当的超卖，可以提高资源利用率
- 比如标称支持8k上下文，90%请求上下文长度4K，两倍超卖也可以让90%的请求满意。（如何制定 service-level agreement (SLA) 不在本文讨论范围内
- 如果不够幸运，推理写满了 kv cache。 肯定要有倒霉蛋请求释放 kv cache，让其他请求继续执行。这个过程称之为 抢占 (Preemption)。只要超卖就会出现抢占。
- 只要显存分配的总 kv cache 大于最长请求上下文长度，通过不断地抢占和华容道，一定能把所有请求都顺利回复，就是可能延迟有点高，让用户不太满意。
- 以 vllm 为例，有两种抢占方式，recompute 重算倒霉蛋请求、swap 将倒霉蛋请求 暂时交换到cpu内存。
- ~~暂时交换到cpu内存肯定比重算恢复快，cpu内存也比较便宜，有条件尽量用swap~~
- 后续测试发现，交换到cpu内存是pcie带宽瓶颈。相比之下算力非常充足，recompute 可能更快。默认 recompute 还是有道理的。
- （实际上还有一种方式，只要抢占就立马回复用户，finish_reason写length肯定会被认为有bug，那就finish_reason写preemption，侮辱性极强，用户看到应该再也不会用你们的产品了
- 如果抢占比例高，会产生系统颠簸。颠簸（Thrashing）：在分页系统中，如果进程频繁地在内存和外存之间移动，导致大量的缺页中断，这称为颠簸，降低了系统性能。 
- 所以，超卖能提高提高资源利用率，代价是抢占导致部分用户延迟变高，甚至引起系统颠簸。

7. 预测请求输出长度
- 如果能预测请求输出长度，系统就能玩出更多花样
- [Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction](https://arxiv.org/abs/2404.08509) 介绍了一种通过预测请求输出长度，speculative shortest-job-first (SSJF) 投机短作业有限调度的方法
对比 先到先服务 first-come-first-serve (FCFS)，提升2.2–3.6×吞吐量

8. 所以 24G 的 4090 能服务多少用户？
- 7B模型，4bit模型，1k上下文就不超卖了，大概可以同时服务32个用户；8k上下文，超卖两倍同时服务8个用户.
- 14B/32B模型,，4bit模型，1k上下文就不超卖了，大概可以同时服务16个用户；8k上下文，超卖两倍同时服务4个用户.
- 后续测试也按这个量级压力测试，测1024个用户、一百万个用户也没啥意义

9. 虽然A100、H100显存80G，下一代的 B100 听说显存192GB，还可以多卡部署、集群部署，但超大模型排着队发布，显存依然不富裕，上述理论分析和实际计算方式还有效。

10. 最后，感叹一下 llm 跟现在主流的硬件真不太匹配，卡带宽、卡显存、相反算力却相当丰富，最先进的 HBM3E 才是这场大语言模型硬件核心。

# 3. 显存管理
LLM 输出总长度事先未知，kv cache 需要按需不断增加，为 llm 推理框架增加了不少麻烦。
- （最主要动机，通过一次性批次执行多个请求，可以大幅提高吞吐、提高GPU利用率。如果批次执行和顺序执行吞吐没有区别，就不用折腾了
- naive 的实现方式，为 kv cache 预先申请最大上下文长度的连续内存。
  - 早期的 orca 和 FasterTransformer 使用这种方式
  - 因为大多数的请求输出比最大上下文小的多，这种方法非常低效
  - 比如 24G 的 4090 加载4bit 7B模型模型后，还有差不多3万 kv cache，分配 4个8K 连续内存，只能同时服务四个用户显然非常低效
- 简单的显存管理会产生显存碎片（fragmentation）
  - 比如 24G 的 4090 加载4bit 7B模型模型后，还有差不多3万 kv cache，分配 32个1k 的连续内存，当服务8K请求时，只需要将8个连续内存拼起来。
  - 但多数时候时候空闲的内存不是连续的，拼不起来
- 受到操作系统的虚拟内存分页启发，vLLM 引入了 PagedAttention 减轻 kv cache 的显存碎片。
  - 如今，PagedAttention 已成为事实上的标准，用于 LLM 服务系统中的动态分配显存。
  - vLLM 使用 Block-Table(s) 实现显存分页，block_size 默认 16， 可选8、16、32。更多细节请参考原始论文 [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180)
  - LightLLM 使用极限的 block_size 等于 1 分配显存。更多细节请参考 [github](https://github.com/ModelTC/lightllm)
  - FlashInfer 使用 Compressed Block-Table(s) 实现显存分页。 更多细节请参考[官方文档](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)
- 显存管理的未来，以上几种分页方式都在用户态管理显存，根据 [vAttention论文](https://arxiv.org/abs/2405.04437) 有以下缺点：
  - 不同的分页方式不通用，不能轻易移植。需要重新写 attention kernel cuda代码，适应PagedAttention。比如适配 FlashInfer decode kernels，vllm需要修改15个文件，600行代码。FlashAttention 要修改 280行代码
  - 增加推理框架的复杂性，显存管理本来是操作系统的工作，现在要在推理框架里完成
  - 引入性能开销，经过测试，会导致attention计算在很多场景下超过10%的速度下降
  - vAttention 通过修改 NVIDIA drivers unified memory management 开源部分代码，添加适合的 CUDA virtual memory APIs。实现系统态显存管理，
  - 围观 [issues](https://github.com/vllm-project/vllm/issues/4675) 期待 vllm 加入 vAttention。

- 总之，显存管理是单用户系统和多用户系统的最大区别。之前测试的 llama.cpp 和 HuggingFace Transformers 至少目前没有加入显存管理功能，不能高效支持多用户推理场景。 

# 4. 连续批处理(Continuous batching)
LLM 输出总长度不统一，输出短的请求会提前退出，资源空闲，影响整体吞吐率。

连续批处理(Continuous batching)，当有请求退出，资源空闲时，不等所有请求都完成，中途及时将等待队列中的请求加入到批次中，减少流水线气泡，提高资源利用率，提高整体吞吐率。

连续批处理(Continuous batching)的关键是一次模型推理，同时进行解码 (Decoding) 阶段和预填充 (Prefill) 阶段。
（铺垫已久的 m 个用户解码阶段就相当于长度为 m 的预填充阶段。试图将解码阶段和预填充阶段统一起来的线索终于可以回收了

大语言模型 LLM (decoder-only transformer)，基本上只有两种操作，线性层（qkvo，mlp）、scaled dot product attention (SDPA)。

对于线性层
- 每个 token 的输入输出相互独立，互相不影响，读一次模型算完所有 token，绝佳的 SIMD 优化
- 线性层支持同时进行进行解码阶段和预填充阶段

对于scaled dot product attention (SDPA)
- 每个 token 的 kv cache 内容、长度、显存位置都不同，自己抱着自己的 kv cache 算，不要互相影响
- SDPA 反正都是自己算自己的，同时进行进行解码阶段和预填充阶段也不是问题

下图来自 [continuous-batching-llm-inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)，
可以看到使用连续批处理(Continuous batching) 技术，流水线被填充的满满当当，非常好。

<img src="https://images.ctfassets.net/xjan103pcp94/744TAv4dJIQqeHcEaz5lko/b823cc2d92bbb0d82eb252901e1dce6d/cb_03_diagram-continuous-batching.png" width="800">

使用连续批处理(Continuous batching)立竿见影，参考 [continuous-batching-llm-inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)
> - Up to 23x throughput improvement using continuous batching and continuous batching-specific memory optimizations (using vLLM).
> - 8x throughput over naive batching by using continuous batching (both on Ray Serve and Hugging Face’s text-generation-inference).
> - 4x throughput over naive batching by using an optimized model implementation (NVIDIA’s FasterTransformer).

更多细节请参考 [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf)

# 5. 

# 6. 实际推理速度测试

## 6.1. VLLM 库
vLLM 是一个快速且易于使用的 LLM 推理和服务库。

[GITHUB](https://github.com/vllm-project/vllm)

[Documentation](https://docs.vllm.ai/en/stable/)

vLLM 使用了 PagedAttention、Continuous batching、Quantization、Optimized CUDA kernels 加速buff叠满

## 6.2 预填充 (Prefill) 阶段
### 6.2.1 参赛队员
[上篇文章](https://github.com/noooop/zerollama/tree/main/tutorial/advance/0x02) 5 名种子选手
- llama.cpp GGUF q8_0 和 q4_k_m；使用预分配 kv cache 速度稳定丝滑
- HuggingFace Transformers bfloat16；“无损” 模型
- AWQ use_exllama_v2；卫冕冠军，使用exllama_v2 kernel
- HuggingFace Transformers AWQ gemm；GEMM 实现也加入对比吧

以及没有上场的选手及原因
- llama.cpp GGUF 有8种模型，速度曲线特征差不多，取两个有代表性的
- bnb int8 只有两个尺寸的 cuda kernel，最优区间非常窄
- bnb int4 先解量化，然后用 bf16 计算，看 bf16 的曲线就行
- GPTQ use_exllama_v2 跟 AWQ use_exllama_v2 cuda 实现一样
- AWQ exllama 只有 长度为 1 的 kernel，不适合多用户场景
- fuse_layers 最优区间在序列长度等于1，长度超过50后开始拖后腿

vllm 4名选手
- vllm bfloat16
- vllm FP8
- vllm GPTQ
- vllm AWQ

> [vllm v0.5.0](https://github.com/vllm-project/vllm/releases/tag/v0.5.0) 终于支持了我心心念念的 [FP8](https://docs.vllm.ai/en/stable/quantization/fp8.html) 赶快测一下

> 遗憾
> - vllm 还支持不少 runtime quantization, 比如 deepspeedfp 和 bitsandbytes
> - 但是使用时需要传一个json文件作为quantization_config，非常麻烦
> - 为什么不能像 HuggingFace Transformers 那样给一个 quantization_config 的参数
> - [参考 RFC Support specifying quant_config details in the LLM or Server entrypoints](https://github.com/vllm-project/vllm/issues/4743)

### 6.2.2 7B 模型 nsight systems profiling 分析

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/prefill-7B.png?raw=true" width="800">

- 速度曲线
- 对比 4090 推理理论极限 和 预填充 (Prefill) 长度等于 1 实际速度

| 名称                 | 速度 (ms) |
|--------------------|---------|
| hf bf16            | 18.77   |
| vllm bf16          | 16.59   |
| w16kv16 理论         | 13.12   |
| vllm fp8           | 13.29   |
| q8_0               | 9.87    |
| w8kv16 理论          | 6.56    |
| q4_k_m             | 12.18   |
| AWQ gemm           | 12.00   |
| AWQ use_exllama_v2 | 8.2     |
| vllm AWQ           | 7.88    |
| vllm GPTQ          | 6.99    |
| w4kv16  理论         | 3.28    |

- bf16 对比量化模型天然就慢一截。BPW 的影响不能忽略
- 值得注意 q8_0 在1-8区间，比 vllm fp8 快，llama.cpp 在这个区间使用 gemv 确实有速度优势
- 建议 vllm 在1-8区间 加入 gemv 优化

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/vllm-bf16-7b.png?raw=true" width="800">

- vllm bfloat16
  - 曲线与 HF bfloat16 非常接近
  - 使用的 bf16 gemm kernel 和 HF bfloat16 几乎一模一样
  - kernel 一样的情况下，vllm 调度比 HF 快一点点，但快的不多

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/vllm-fp8-7b.png?raw=true" width="800">

- vllm FP8
  - 对比 vllm bfloat16 加速明显
  - 使用 gemm_e4m3 kernel，确实比 bf16 gemm 快一些
  - 1-20 使用 gemm_e4m3 128X64， 32+ 使用 gemm_e4m3 256X64 也算有接力

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/vllm-gptq-7b.png?raw=true" width="800">

- vllm GPTQ
  - 使用了 Marlin kernel 真的是扬眉吐气
  - 1-100 区间全场最快
  - 100 以上开始比 vllm FP8 慢，500 直逼 vllm bfloat16
  - 以下有两种解读
    - 使用一种 kernel 最优速度区间就是会顾头不顾腚
    - Marlin kernel 本质就是一种 bf16 的 kernel，序列长度长，就接近 vllm bfloat16
    - 期待一种 FP8 e4m3 Marlin kernel 加入对比

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/vllm-awq-7b.png?raw=true">

- vllm AWQ
  - (v0.5.0.post1) WARNING awq quantization is not fully optimized yet. The speed can be slower than non-quantized models.
  - 上届卫冕冠军沦落至此
  - 只有一种 awq gemm 4bit kernel，没有了 exllama_v2 kernel 加持，速度就很憋屈
  - 1-256 区间 vllm AWQ 速度曲线跟 HF AWQ gemm 趋势一样，对比之前文章，cuda kernel 也一样。
  - 256 使用了下面提到的 FP16_MATMUL_HEURISTIC_CONDITION
  - 1-32 区间虽然慢但接近最快的 vllm GPTQ，所以这一个 kernel 是轻负载调教
  - 128 以上甚至比 vllm bfloat16 都慢

- 256 长度速度突变
        
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/awq.py#L165

```
# num_tokens >= threshold
FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256
if FP16_MATMUL_HEURISTIC_CONDITION:
    out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
    out = torch.matmul(reshaped_x, out)
else:
    out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros,
                       pack_factor)
```
可能其他实现也有类似的代码，反正 vllm 长度 256 都会有速度突变

### 6.2.4 其他尺寸模型预填充 (Prefill) 阶段速度曲线

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/prefill-0.5B.png?raw=true" width="800">

- 0.5B，vllm GPTQ 有上篇文章测试 HF GPTQ Marlin kernel的锯齿速度曲线
- vllm 启动开销比 HF 小，运行小模型比较有优势，相反 llama.cpp 启动开销就比较大
- vllm bfloat16 速度最快，比 vllm FP8 都快，小尺寸模型真是一切皆有可能

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/prefill-1.8B.png?raw=true" width="800">

- 1.8B，Marlin kernel 的锯齿速度曲线稍微不明显，所以 Marlin kernel 实际上不太适合小尺寸（0.5B 1.8B）模型

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/prefill-4B.png?raw=true" width="800">

- 4B，没有特别需要说明的

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/prefill-7B.png?raw=true" width="800">

- 7B，没有特别需要说明的

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/prefill-14B.png?raw=true" width="800">

- 14B，vllm FP8 加载 14B 模型 oom，不应该啊

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/prefill-32B.png?raw=true" width="800">

- 32B，vllm AWQ 同 HF AWQ 一样，加载 32B 模型 oom，真遗憾

### 6.2.5 小结
- vllm 启动开销比较低，相同模型都比 HF 快一些
- 轻负载 (1-8) 建议使用量化模型
  - 4bit 7B模型理论速度 3ms，天然比 bf16 快；bf16 理论速度 13ms 对比量化模型天然就慢一截
  - vllm GPTQ 使用 Marlin kernel 速度是最快的，Marlin kernel 调教也偏轻负载
  - vllm 不支持 exllama_v2，没有看到 exllama_v2 和 Marlin 正面对决比较遗憾 
  - vllm AWQ 使用 gemm 效果就不太行，官方也有 WARNING 暂时不建议使用
  - AWQ 量化方法对模型影响比 GPTQ 小，AWQ 实际上性价比很高
- 重负载建议使用vllm bfloat16
  - 重负载，量化模型有反量化开销，反而速度不如 bfloat16
  - vllm bfloat16 1-8 使用 gemv 优化一下， 现在的 18ms 优化到理论的 13ms 提高也不是很明显
- vllm FP8 刚发布，对模型能力影响程度还需要观望

## 6.3. 解码 （Decoding） 阶段
### 6.3.1. 单用户场景

- 解码 （Decoding） 阶段 延迟有两部分组成
  - 固定的读取模型时间，在延迟曲线里就表现为 截距 B
  - 随着序列长度增加，读取 kv cache 线性增加，在延迟曲线里就表现为斜率 W

- 用户数为 1 的 解码 （Decoding） 阶段 相当于 长度为 1 预填充 (Prefill) 阶段 
  - 把 “对比 4090 推理理论极限 和 预填充 (Prefill) 长度等于 1 实际速度”图 再拿过来，把卫冕冠军 AWQ use_exllama_v2 fuse_layers 也加上
  - （下表就相当于 固定的读取模型时间， 参考着看一下

| 名称                             | 速度 (ms) |
|--------------------------------|---------|
| hf bf16                        | 18.77   |
| vllm bf16                      | 16.59   |
| w16kv16 理论                     | 13.12   |
| vllm fp8                       | 13.29   |
| q8_0                           | 9.87    |
| w8kv16 理论                      | 6.56    |
| q4_k_m                         | 12.18   |
| AWQ gemm                       | 12.00   |
| AWQ use_exllama_v2             | 8.2     |
| vllm AWQ                       | 7.88    |
| vllm GPTQ                      | 6.99    |
| AWQ use_exllama_v2 fuse_layers | 6.37    |
| w4kv16  理论                     | 3.28    |


<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/decoding-0.5B-1.png?raw=true" width="800">

- 0.5B 

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/decoding-1.8B-1.png?raw=true" width="800">

- 1.8B

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/decoding-4B-1.png?raw=true" width="800">

- 4B 

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/decoding-7B-1.png?raw=true" width="800">

- 7B 

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/decoding-14B-1.png?raw=true" width="800">

- 14B 

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/decoding-32B-1.png?raw=true" width="800">

- 32B 

总结：
- vllm 稳定丝滑，llama.cpp 也甘拜下风
- bf16，vllm bf16 模型比 hf bf16 快
- fp8，vllm fp8 比 gguf q8_0 快 （这两个模型姑且有可比性吧
- ~int4，vllm GPTQ 比 gguf q4_k_m，AWQ gemm，AWQ use_exllama_v2 都快
- 卫冕冠军 AWQ use_exllama_v2 fuse_layers， 因为叠了 fuse_layers buff，开始速度比 vllm GPTQ 快一点点
- 可能是 vllm self attention 速度比较快，可能是 vllm kv cache 比较快，vllm GPTQ 速度曲线增速缓，AWQ use_exllama_v2 fuse_layers 增速快，长度500后开始反超
- 所以 vllm 哪怕在用户为 1 的场景下，比 llama.cpp 和 HuggingFace Transformers 都快
- 用 vllm 就对了

### 6.3.2. 多用户场景

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/latency-throughput/latency-throughput-32.png?raw=true" width="800">

1. 上图采样了并发用户量为1、2、4、8、16、32、64、128，vllm GPTQ 模型解码 （Decoding） 输出长度为 32 时，延迟和吞吐关系。
- 并发用户数越多，延迟越大，吞吐越大。 延迟和吞吐存在制约关系
- 模型越小，延迟和吞吐曲线越靠左上，越好。当然模型越小能力也越差，也存在制约关系
- 研究曲线走势
  - 0.5B、1.8B、4B 并发 128 往上提高应该还挺大，毕竟小模型比较小
  - 14B，32B 从 64 并发到 128 并发 提升已经不大。肘点，或者叫平衡延迟和吞吐的最优并发用户数应该在 64 左右
  - 问题就来了，24G 显存撑不起 64 用户并发，kv cache 很快会被写满
- 实际上，输出长度为 32 算出来的峰值吞吐不可持续
  - 随着长度增加，kv cache 读取量增加
  - 高并发进一步放大了读取量，延迟增长非常快
  - 延迟曲线上表现为截距随并发变大，斜率也随并发变大

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/latency-throughput/decoding-7B-latency.png?raw=true" width="800">

2. 上图是并发用户量为1、2、4、8、16、32、64、128，vllm GPTQ 7B模型解码 （Decoding） 输出长度到 1000 的延迟
- 验证上面提到的“延迟曲线上表现为截距随并发变大，斜率也随并发变大”
- 验证高并发导致“kv cache 很快会被写满”，出现抢占
  - 并发1-16可以平稳顺利的结束
  - 并发32 总体还比较顺利, 速度没有模型衰减
  - 并发数继续增加到64、128，会发生多次抢占，基本不可用
- 曲线开始的尖峰是全力prefill导致的，所有请求都同时进来，同时做 prefill。

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/latency-throughput/decoding-7B-throughput.png?raw=true" width="800">

3. 上图是并发用户量为1、2、4、8、16、32、64、128，vllm GPTQ 7B模型解码 （Decoding） 输出长度到 1000 的吞吐
- 验证上面提到的“随着长度增加，kv cache 读取量增加，输出长度为 32 算出来的峰值吞吐不可持续”

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/latency-throughput/latency-throughput-512.png?raw=true" width="800">

4. 上图采样了并发用户量为1、2、4、8、16、32、64、128，vllm GPTQ 模型解码 （Decoding） 输出长度为 512 时，延迟和吞吐关系。
- 研究曲线走势
  - 14B和32B模型，并发数继续增加到64、128，kv cache 很快会被写满，抢占，实际并发为32
  - 从曲线前段趋势推断，平衡延迟和吞吐的最优并发用户数从64往32移动
- 忘掉不切实际的峰值吞吐吧，上下文长度1K时，24G的4090控制并发到32以下，这时延迟非常低，速度也非常稳定，用户会非常满意。
- 对于一张一万五的游戏卡，不要要求太多

5. Preemption 抢占
使用默认的 recompute 抢占模式，抢占恢复会有一个很高的延迟，带来不好的用户体验，使用swap模型能否改善呢？

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/latency-throughput/preemption_mode.png?raw=true" width="800">

没啥区别，甚至更糟糕
- llm 模型推理带宽瓶颈，swap 走pcie总线也带宽瓶颈，显存内部带宽比pcie总线快多了。
- 相比之下算力非常充足，recompute 可能更快

### 6.4. Chunked Fill

根据之前的理论推导的实验验证，24G 的 4090 服务 32 用户并发时，能很好的平衡延迟、吞吐、kv cache容量。

那就开启 vllm chunked_prefill， max_num_seqs 设为 32，max_num_batched_tokens 设为 64 给 预填充 (Prefill) 阶段 的请求留一些空间

> WARNING vllm 开启 chunked_prefill 会随机出现 RuntimeError: CUDA error: an illegal memory access was encountered。[复现代码](https://github.com/noooop/zerollama/blob/v0.4/test/debug/vllm_chunked_fill.py)

```
engine_args = EngineArgs(model=model_name,
                         ....
                         enable_chunked_prefill=True,
                         max_num_batched_tokens=64,
                         max_num_seqs=32)
```

### 6.4.1. 预填充 (Prefill) 阶段

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/chunked_fill/prefill-0.5B-chunked_fill.png?raw=true" width="800">

- 0.5B 模型太小，不需要 chunked fill 优化

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/chunked_fill/prefill-1.8B-chunked_fill.png?raw=true" width="800">

- 1.8B 模型太小，不需要 chunked fill 优化

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/chunked_fill/prefill-4B-chunked_fill.png?raw=true" width="800">

- 4B 模型太小，不需要 chunked fill 优化

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/chunked_fill/prefill-7B-chunked_fill.png?raw=true" width="800">

- 7B 就有趣了
- 黑线的 vllm GPTQ chunked_fill 和 墨绿色的 vllm GPTQ 比较接近
- 使用 chunked fill 优化，对较长提示词预填充速度损失比较小
- Marlin kernel 本来就是为 1-32 区间轻负载调教，将长提示词拆分更适合 Marlin kernel 发挥

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/chunked_fill/prefill-14B-chunked_fill.png?raw=true" width="800">

- 14B，同样 vllm GPTQ chunked_fill 和 vllm GPTQ 比较接近
- vllm AWQ chunked_fill 和 vllm AWQ 也比较接近， 甚至 256-450 区间 比使用 FP16_MATMUL_HEURISTIC_CONDITION 还要快一点

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/chunked_fill/prefill-32B-chunked_fill.png?raw=true" width="800">

- 32B，我的天，使用 chunked_fill， 居然能将 32B 的 AWQ 模型跑起来
- chunked_fill 居然有节省运行时显存的功能

总结：
- 0.5B、1.8B、4B 模型太小，不需要 chunked fill 优化
- 7B、14B，32B，vllm GPTQ chunked_fill 和 vllm GPTQ 比较接近， 可以开启 chunked fill 优化
- 开启 chunked fill 优化有节省运行时显存的功能

### 6.4.2. 解码 （Decoding） 阶段

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/chunked_fill/decoding-7B-latency.png?raw=true" width="800">

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/chunked_fill/decoding-7B-throughput.png?raw=true" width="800">

- 7B 

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/chunked_fill/decoding-14B-latency.png?raw=true" width="800">

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/chunked_fill/decoding-14B-throughput.png?raw=true" width="800">

- 14B 

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/chunked_fill/decoding-32B-latency.png?raw=true" width="800">

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/chunked_fill/decoding-32B-throughput.png?raw=true" width="800">

- 32B 

这种强行限制并发数至32，并发64、128的曲线被强行摁到32，减少系统颠簸，个人觉得非常好。


