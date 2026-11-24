"""Recompute every input token. Nothing is stored."""

from policies.base import PolicySpec

SPEC = PolicySpec(name="no_cache", store=False, eviction="lru")
