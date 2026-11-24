"""HBM plus a local DRAM tier. No remote fetch and no optical circuit."""

from policies.base import PolicySpec

SPEC = PolicySpec(name="local_dram", use_hbm=True, use_dram=True, store=True, eviction="lru")
