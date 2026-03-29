from __future__ import annotations

import torch

from dian_spring_test.models import GroupedQueryAttention, StandardMHA
from dian_spring_test.utils import set_seed


def run_standard_mha_demo() -> None:
    print("=== Standard MHA shape check ===")
    batch_size, seq_len, hidden_dim, num_heads = 2, 7, 32, 4
    #sub_dim=hidden_dim//num_heads
    x = torch.randn(batch_size, seq_len, hidden_dim)
    model = StandardMHA(hidden_dim=hidden_dim, num_heads=num_heads)
    y, _, debug = model(x)
    print(f"input_shape={tuple(x.shape)} output_shape={tuple(y.shape)}")
    print(f"head_debug={debug}")

    #return::output, cache, debug = model(x, use_cache=True)


def run_kv_cache_demo() -> None:
    print("\n=== KV Cache autoregressive decoding demo ===")
    batch_size, hidden_dim, num_heads = 2, 32, 4
    model = StandardMHA(hidden_dim=hidden_dim, num_heads=num_heads, causal=True)

    init_x = torch.randn(batch_size, 10, hidden_dim)
    _, cache, debug = model(init_x, use_cache=True)
    print(f"step=init q={debug['q_shape']} k_cache={debug['k_shape']} v_cache={debug['v_shape']}")

    for step in range(1, 6):
        new_token = torch.randn(batch_size, 1, hidden_dim)
        _, cache, debug = model(new_token, past_key_values=cache, use_cache=True)
        print(f"step={step} q={debug['q_shape']} k_cache={debug['k_shape']} v_cache={debug['v_shape']}")


def run_gqa_demo() -> None:
    print("\n=== MHA / GQA / MQA unified demo ===")
    x = torch.randn(2, 5, 32)
    settings = [(4, "MHA"), (2, "GQA"), (1, "MQA")]
    for num_kv_heads, name in settings:
        module = GroupedQueryAttention(hidden_dim=32, num_heads=4, num_kv_heads=num_kv_heads)
        y = module(x)
        print(f"variant={name} num_kv_heads={num_kv_heads} output_shape={tuple(y.shape)}")


def main() -> None:
    set_seed(1234)
    run_standard_mha_demo()
    run_kv_cache_demo()
    run_gqa_demo()


if __name__ == "__main__":
    main()
