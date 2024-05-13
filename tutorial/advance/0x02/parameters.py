
B = 1024 * 1024 * 1024
M = 1024 * 1024


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


def kv_cache_parameters(num_hidden_layers, hidden_size, n_groups):
    layer_parameters = (
        hidden_size // n_groups +   # key_states
        hidden_size // n_groups     # value_states
    )
    return num_hidden_layers * layer_parameters


def kv_cache_len(parameters, kv_cache_parameters, memory_size, w_size, a_size):
    return (memory_size - parameters * w_size / 8) / (kv_cache_parameters * a_size / 8)


def first_token_latency_roughly(Mp, KVp, w_size, a_size, n, n_groups, bandwidth, computing_power):
    read_latency = ((Mp * 1) * w_size + (2*n-1) * KVp * a_size) / bandwidth * 1000
    computing_latency = (Mp * n + (2*n-1) * KVp * n_groups) * 2 / computing_power * 1000
    return read_latency, computing_latency


def first_token_latency_exactly(Lp, Hp, KVp, w_size, a_size, n, n_groups, bandwidth, computing_power):
    read_latency = (((Lp + Hp) * 1) * w_size + (2*n-1) * KVp * a_size) / bandwidth * 1000
    computing_latency = (Lp * n + (2*n-1) * KVp * n_groups + Hp) * 2 / computing_power * 1000
    return read_latency, computing_latency


def latency(Mp, KVp, w_size, a_size, n, bandwidth):
    B = (Mp * w_size + 1 * KVp * a_size) / bandwidth * 1000
    W = 1 / (KVp * a_size / bandwidth * 1000)

    read_latency1 = (1/W) * (n-1) + B
    read_latency2 = (Mp * w_size + n * KVp * a_size) / bandwidth * 1000

    return read_latency1, read_latency2, W, B


def batch_latency(Mp, KVp, w_size, a_size, m, n, bandwidth):
    return (Mp * (n-m) * w_size + (2*n-1) * KVp * a_size) / bandwidth * 1000


if __name__ == '__main__':
    info = [
        ["0.5B", 151936, 1024, 24, 2816, 1],
        ["1.8B", 151936, 2048, 24, 5504, 1],
        ["4B", 151936, 2560, 40, 6912, 1],
        ["7B", 151936, 4096, 32, 11008, 1],
        ["14B", 152064, 5120, 40, 13696, 1],
        ["32B", 152064, 5120, 64, 27392, 1],
        ["32B", 152064, 5120, 64, 27392, 5],
        ["72B", 152064, 8192, 80, 24576, 1],
        ["110B", 152064, 8192, 80, 49152, 1],
        ["110B", 152064, 8192, 80, 49152, 8]
    ]

    import sympy as sm

    bandwidth = 1008 * 1024 * 1024 * 1024 * 8
    computing_power = 82.58 * 1024 * 1024 * 1024 * 1024

    """
    print("model_parameters: B")
    for name, vocab_size, hidden_size, num_hidden_layers, intermediate_size, n_groups in info:
        Lp, Hp, Ap, Tp = model_parameters(num_hidden_layers, hidden_size, intermediate_size, vocab_size, n_groups)
        print(name, f"{Lp/B:0.2f}B {Hp/B:0.2f}B {Ap/B:0.2f}B {Tp/B:0.2f}B")

    print("kv_cache_parameters: M")
    for name, vocab_size, hidden_size, num_hidden_layers, intermediate_size, n_groups in info:
        print(name, kv_cache_parameters(num_hidden_layers, hidden_size, n_groups)/M, "M")

    memory_size = 24 * 1024 * 1024 * 1024
    for s in ["w16kv16", "w8kv16", "w4kv16", "w6kv16", "w8kv8", "w4kv8"]:
        w_size, a_size = s.split("kv")
        w_size = int(w_size[1:])
        a_size = int(a_size)

        print(s)
        for name, vocab_size, hidden_size, num_hidden_layers, intermediate_size, n_groups in info:
            Lp, Hp, Ap, Tp = model_parameters(num_hidden_layers, hidden_size, intermediate_size, vocab_size, n_groups)
            kv_size = kv_cache_parameters(num_hidden_layers, hidden_size, n_groups)

            print(name, int(kv_cache_len(Ap, kv_size, memory_size, w_size, a_size)))

    print("First Token Latency")
    for n in [1, 20, 27, 200, 1000]:
        for s in ["w16kv16", "w8kv16", "w4kv16"]:
            w_size, a_size = s.split("kv")
            w_size = int(w_size[1:])
            a_size = int(a_size)

            print(n, s)
            for name, vocab_size, hidden_size, num_hidden_layers, intermediate_size, n_groups in info:
                Lp, Hp, Ap, Tp = model_parameters(num_hidden_layers, hidden_size, intermediate_size, vocab_size, n_groups)
                kv_size = kv_cache_parameters(num_hidden_layers, hidden_size, n_groups)

                #print(name, f"{Lp / B:0.2f}B {Hp / B:0.2f}B {Tp / B:0.2f}B")

                r, c = first_token_latency_roughly(Ap, kv_size, w_size, a_size, n, n_groups, bandwidth, computing_power)
                print(name, f"{r:0.2f}ms {c:0.2f}ms {max(r, c):0.2f}ms")

                r, c = first_token_latency_exactly(Lp, Hp, kv_size, w_size, a_size, n, n_groups, bandwidth, computing_power)

                print(name, f"{r:0.2f}ms {c:0.2f}ms {max(r, c):0.2f}ms")

    print("Prefill turning point")

    for s in ["w16kv16", "w8kv16", "w4kv16"]:
        w_size, a_size = s.split("kv")
        w_size = int(w_size[1:])
        a_size = int(a_size)

        print(s)
        n = sm.symbols('n')
        for name, vocab_size, hidden_size, num_hidden_layers, intermediate_size, n_groups in info:
            Lp, Hp, Ap, Tp = model_parameters(num_hidden_layers, hidden_size, intermediate_size, vocab_size, n_groups)
            kv_size = kv_cache_parameters(num_hidden_layers, hidden_size, n_groups)

            r, c = first_token_latency_roughly(Ap, kv_size, w_size, a_size, n, n_groups, bandwidth, computing_power)

            o = sm.solve(r-c, n)

            print(name, f"{o[0]:0.2f}")

            r, c = first_token_latency_exactly(Lp, Hp, kv_size, w_size, a_size, n, n_groups, bandwidth, computing_power)
            o = sm.solve(r - c, n)
            print(name, f"{o[0]:0.2f}")

    print("Decoding Latency")
    for n in [1, 20, 1000]:
        for s in ["w16kv16", "w8kv16", "w4kv16"]:
            w_size, a_size = s.split("kv")
            w_size = int(w_size[1:])
            a_size = int(a_size)

            print(n, s)
            for name, vocab_size, hidden_size, num_hidden_layers, intermediate_size, groups in info:
                Lp, Hp, Ap, Tp = model_parameters(num_hidden_layers, hidden_size, intermediate_size, vocab_size, groups)
                kv_size = kv_cache_parameters(num_hidden_layers, hidden_size, groups)

                latency1, latency2, W, B = latency(Ap, kv_size, w_size, a_size, n, bandwidth)
                #print(latency1, latency2)
                print(name, f"{W:0.0f} {B:0.2f}ms")

    print("model_parameters: B")
    for name, vocab_size, hidden_size, num_hidden_layers, intermediate_size, n_groups in info:
        Lp, Hp, Ap, Tp = model_parameters(num_hidden_layers, hidden_size, intermediate_size, vocab_size, n_groups)
        print(name, f"{Tp*2/B:0.2f}GB")

    print("kv_cache_parameters: B")
    for name, vocab_size, hidden_size, num_hidden_layers, intermediate_size, n_groups in info:
        for max_tokens in [2 ** 14]:
            Lp, Hp, Ap, Tp = model_parameters(num_hidden_layers, hidden_size, intermediate_size, vocab_size, n_groups)
            kv_size = kv_cache_parameters(num_hidden_layers, hidden_size, n_groups)
            print(name, max_tokens, f"{Tp*2/B:0.2f}GB", f"{kv_size*max_tokens*2/B:0.2f}GB", f"{(Tp+kv_size*max_tokens)*2/B:0.2f}GB")
    """

    m = 25
    for s in ["w16kv16", "w8kv16", "w4kv16"]:
        print(s)

        for n in range(1, 1281):
            w_size, a_size = s.split("kv")
            w_size = int(w_size[1:])
            a_size = int(a_size)


            for name, vocab_size, hidden_size, num_hidden_layers, intermediate_size, groups in info:
                Lp, Hp, Ap, Tp = model_parameters(num_hidden_layers, hidden_size, intermediate_size, vocab_size,
                                                  groups)
                KVp = kv_cache_parameters(num_hidden_layers, hidden_size, groups)

                latency = batch_latency(Tp, KVp, w_size, a_size, m, n, bandwidth)

                if n in [128 * i for i in range(1, 11)]:
                    print(name, latency)


