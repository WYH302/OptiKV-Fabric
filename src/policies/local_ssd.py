"""HBM, local DRAM, and local SSD. No remote fetch and no optical circuit."""

from policies.base import PolicySpec

SPEC = PolicySpec(
    name="local_ssd",
    use_hbm=True,
    use_dram=True,
    use_ssd=True,
    store=True,
    eviction="lru",
)
