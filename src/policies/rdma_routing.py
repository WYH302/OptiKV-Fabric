"""Electrical RDMA pool plus prefix routing. No optical reconfiguration."""

from policies.base import PolicySpec

SPEC = PolicySpec(
    name="rdma_routing",
    use_hbm=True,
    use_dram=True,
    use_ssd=True,
    use_rdma=True,
    use_remote_hbm=True,
    optical="none",
    routing=True,
    store=True,
    eviction="lru",
)
