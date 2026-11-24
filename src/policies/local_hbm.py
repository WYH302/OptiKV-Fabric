"""Admit KV only into the worker's HBM."""

from policies.base import PolicySpec

SPEC = PolicySpec(name="local_hbm", use_hbm=True, store=True, eviction="lru")
