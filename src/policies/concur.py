"""CONCUR (ICML '26): Agent-level AIMD concurrency admission control over electrical RDMA.

Regulates the active session concurrency window based on real-time KV cache
eviction/capacity pressure to prevent middle-phase HBM thrashing while running
over standard electrical RDMA + local memory tiers (tau = 0).
"""

from policies.base import PolicySpec

SPEC = PolicySpec(
    name="concur_rdma",
    use_hbm=True,
    use_dram=True,
    use_ssd=True,
    use_rdma=True,
    use_remote_hbm=True,
    optical="none",
    routing=True,
    prefetch=False,
    store=True,
    eviction="lru",
    admission_control=True,
)
