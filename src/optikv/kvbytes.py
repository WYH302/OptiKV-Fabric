"""Per-token KV byte formulas. These are the only definitions the simulator uses."""

from __future__ import annotations


def gqa_bytes_per_token(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> int:
    """GQA / MHA KV bytes per token.

    bytes = L * 2 * H_kv * head_dim * dtype_bytes
    The factor 2 is the key tensor plus the value tensor.
    """
    return int(num_layers) * 2 * int(num_kv_heads) * int(head_dim) * int(dtype_bytes)


def mla_bytes_per_token(
    num_hidden_layers: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dtype_bytes: int,
) -> int:
    """DeepSeek MLA KV bytes per token.

    bytes = num_hidden_layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
    """
    return int(num_hidden_layers) * (int(kv_lora_rank) + int(qk_rope_head_dim)) * int(dtype_bytes)
