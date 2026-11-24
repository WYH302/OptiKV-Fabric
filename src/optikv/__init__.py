"""OptiKV-Fabric request-level KV cache simulator."""

from optikv.kvbytes import gqa_bytes_per_token, mla_bytes_per_token

__version__ = "0.1.0"
__all__ = ["gqa_bytes_per_token", "mla_bytes_per_token", "__version__"]
