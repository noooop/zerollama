# 大语言模型推理理论极限和实际测试二， 多用户实时交互

# TL;DR

# 1. 回顾
[上篇文章](https://github.com/noooop/zerollama/tree/main/tutorial/advance/0x02) 分析了单用户独占实时交互场景下的大语言模型推理理论极限和实际测试。

- 推理过程可以分成，预填充 (Prefill) 和解码 （Decoding） 两个阶段
- 预填充 (Prefill) 阶段对应于输出第一个词的的，需要将提示词每个词的中间结果都写入 kv cache 里，才能推理得到第一个词。
这个阶段是对 kv cache 进行预填充。
- 解码 （Decoding） 阶段对应于输出之后的 token 的输出。 kv cache 已经填入之前 token。 推理过程就是从下往上过一遍模型，最后从 lm_head 输出 logits 采样。
- 现代 CPU/GPU 的 ALU 操作（乘法、加法）内存 IO 速度要快得多，transformer 这种只需要对每个元素执行两次操作的场景，必定受到访存带宽的限制。
- NVIDIA GeForce RTX 4090：1008 GB/s 显存带宽和 83 TFLOPS，Flop:byte = 82:1
- 对于解码 （Decoding） 阶段，每输出一个词需要读一遍模型，读取模型的极限就是速度的极限
- 对于预填充 (Prefill) 阶段，读一遍模型计算提示词中的多个词，无论从 SIMD （Single Instruction Multiple Data） 的角度还是从带宽瓶颈vs算力瓶颈的角度都很划算
- 投机采样 (Speculative Decoding) 利用了 llm 批次执行速度比单次执行效率高，加速模型输出
- 文章最后，解码 (Decoding) 阶段是一种 长度为 1 的预填充 (Prefill) 阶段，将解码和预填充统一起来，在分析 CUDA kernels 的最优运行区间的时候非常有效。
- 这篇文章，讨论多用户实时交互，也就是：m个用户解码 (Decoding) 阶段就相当于长度为m的预填充 (Prefill) 阶段。就要求 CUDA kernels 在典型用户数区间里面最优。（这句油门踩的是不是太快了，狗头

# 2. 24G 的 4090 能服务多少用户
llm模型推理的时候，每个 token 的中间结果，都要占用一小块显存空间，称为 kv cache。上篇文章 从理论分析到实际测试，24G 显存服务单用户都不太富裕。 

现在要服务多用户，本不富裕的显存又雪上加霜。所以先要搞清楚卡显存的边界在哪里。（llm 跟现在主流的硬件真不太匹配，卡带宽、卡显存、算力却相当丰富。

且听我从盘古开天地开始慢慢道来：

1. 随着大模型越来越卷，训练时上下文（原生上下文长度）也越来越长，例如llama 2k、llama2 4k、llama3 8k、qwen 2k、qwen2 32K.
2. 随着扩展上下文研究的不断发展，支持的长上下文模型越来越长，例如qwen1.5 32K, qwen2 128K、gpt-4/gpt-4o 128k、Claude 3 200k、[Llama-3 8B Gradient Instruct 1048k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k)。
3. 长上下文意味着：
   - 能处理更长的文本内容，比如摘要一整本书、一个长播客
   - 能输出更长的文本内容，比如输出一整个markdown文档
   - 做更细致的提示词工程，描述更细节的要求、放多个示例
   - 给大模型更多Chain-of-Thought空间
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
- 只要显存分配的 kv cache 大于最长请求上下文长度，通过不断地抢占和华容道，一定能把所有请求都顺利回复，就是可能延迟有点高，让用户不太满意。
- 以 vllm 为例，有两种抢占方式，recompute 重算倒霉蛋请求、swap 将倒霉蛋请求 暂时交换到cpu内存。暂时交换到cpu内存肯定比重算恢复快，cpu内存也比较便宜，有条件尽量用swap
- （实际上还有一种方式，只要抢占就立马回复用户，finish_reason写length肯定会被认为有bug，那就finish_reason写preemption，侮辱性极强，用户看到应该再也不会用你们的产品了
- 所以，超卖能提高提高资源利用率，代价是抢占导致部分用户延迟变高。

7. 预测请求输出长度
- 如果能预测请求输出长度，系统就能玩出更多花样
- [Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction](https://arxiv.org/abs/2404.08509) 介绍了一种通过预测请求输出长度，speculative shortest-job-first (SSJF) 投机短作业有限调度的方法
对比 先到先服务 first-come-first-serve (FCFS)，提升2.2–3.6×吞吐量

8. 所以 24G 的 4090 能服务多少用户？
- 7B模型，4bit模型，1k上下文就不超卖了，大概可以同时服务32个用户；8k上下文，超卖两倍同时服务8个用户.
- 14B/32B模型,，4bit模型，1k上下文就不超卖了，大概可以同时服务16个用户；8k上下文，超卖两倍同时服务4个用户.
- 后续测试也按这个量级压力测试，测1024个用户、一百万个用户也没啥意义

9. 虽然A100、H100显存80G，下一代的 B100 听说显存192GB，还可以多卡部署、集群部署，但超大模型排着队发布，显存依然不富裕，上述理论分析和实际计算方式还有效。


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

### 6.2.2 7B 模型 nsight systems profiling 分析

<img src="https://github.com/noooop/noooop.github.io/blob/main/benchmarking/vllm/prefill-7B.png?raw=true" width="800">

- 速度曲线
- 对比 4090 推理理论极限 和 预填充 (Prefill) 长度等于 1 实际速度

| 名称                 | 速度 (ms) |
|--------------------|---------|
| hf bf16            | 18.77   |
| vllm bf16          | 16.59   |
| w16kv16 理论         | 13.12   |
| q8_0               | 9.87    |
| vllm fp8           | 13.29   |
| w8kv16 理论          | 6.56    |
| q4_k_m             | 12.18   |
| AWQ gemm           | 12.00   |
| AWQ use_exllama_v2 | 8.2     |
| vllm GPTQ          | 6.99    |
| vllm AWQ           | 7.88    |
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
  - 4bit 7B模型理论速度 3ms，天然比 bf16 快；bf16 理论速度 13ms 对比量化模型天然就慢一截，
  - vllm GPTQ 使用 Marlin kernel 速度是最快的，Marlin kernel 调教也偏轻负载
  - vllm 不支持 exllama_v2，没有看到 exllama_v2 和 Marlin 正面对决比较遗憾 
  - vllm AWQ 使用 gemm 效果就不太行，官方也有 WARNING 暂时不建议使用
  - AWQ 量化方法对模型影响比 GPTQ 小，AWQ 实际上性价比很高
- 重负载建议使用vllm bfloat16
  - 重负载，量化模型有反量化开销，反而速度不如 bfloat16
  - vllm bfloat16 1-8 使用 gemv 优化一下， 现在的 18ms 优化到理论的 13ms 提高也不是很明显
- vllm FP8 刚发布，对模型能力影响程度还需要观望

