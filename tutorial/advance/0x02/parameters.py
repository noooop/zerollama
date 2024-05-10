
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
    lm_head_parameters = hidden_size * vocab_size
    return (num_hidden_layers * layer_parameters + lm_head_parameters)


def kv_cache_parameters(num_hidden_layers, hidden_size, n_groups):
    layer_parameters = (
        hidden_size // n_groups +   # key_states
        hidden_size // n_groups     # value_states
    )
    return num_hidden_layers * layer_parameters


def kv_cache_len(parameters, kv_cache_parameters, memory_size, w_size, a_size):
    return (memory_size - parameters * w_size / 8) / (kv_cache_parameters * a_size / 8)

if __name__ == '__main__':

    info = [
        ["0.5B", 151936, 1024, 24, 2816, 1],
        ["1.8B", 151936, 2048, 24, 5504, 1],
        ["4B", 151936, 2560, 40, 6912, 1],
        ["7B", 151936, 4096, 32, 11008, 1],
        ["14B", 152064, 5120, 40, 13696, 1],
        ["32B", 152064, 5120, 64, 27392, 5],
        ["72B", 152064, 8192, 80, 24576, 1],
        ["110B", 152064, 8192, 80, 49152, 8]
    ]

    print("model_parameters: B")
    for name, vocab_size, hidden_size, num_hidden_layers, intermediate_size, groups in info:
        print(name, model_parameters(num_hidden_layers, hidden_size, intermediate_size, vocab_size, groups)/B, "B")

    print("kv_cache_parameters: M")
    for name, vocab_size, hidden_size, num_hidden_layers, intermediate_size, groups in info:
        print(name, kv_cache_parameters(num_hidden_layers, hidden_size, groups)/M, "M")

    memory_size = 24 * 1024 * 1024 * 1024
    for s in ["w16a16", "w8a16", "w4a16", "w6a16", "w8a8", "w4a8"]:
        w_size, a_size = s.split("a")
        w_size = int(w_size[1:])
        a_size = int(a_size)

        print(s)
        for name, vocab_size, hidden_size, num_hidden_layers, intermediate_size, groups in info:
            p = model_parameters(num_hidden_layers, hidden_size, intermediate_size, vocab_size, groups)
            kv_size = kv_cache_parameters(num_hidden_layers, hidden_size, groups)

            print(name, int(kv_cache_len(p, kv_size, memory_size, w_size, a_size)))
