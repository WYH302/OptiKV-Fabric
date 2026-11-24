"""Upgraded 2025-2026 Electrical Agentic SOTA baseline (KVFlow / SYMPHONY paradigm).

Equips the full electrical memory hierarchy (HBM + DRAM + SSD + Remote HBM + 25 GB/s
contended RDMA pool, tau = 0) with the exact same software mechanisms as OptiKV-Fabric:
prefix-and-load routing, congestion-gated inter-turn prefetching over PCIe/RDMA,
Agent Anchors, and Prefix-Dependency-Aware GDSF eviction.
"""

from policies.base import PolicySpec

SPEC = PolicySpec(
    name="rdma_prefetch",
    use_hbm=True,
    use_dram=True,
    use_ssd=True,
    use_rdma=True,
    use_remote_hbm=True,
    optical="none",
    routing=True,
    prefetch=True,
    store=True,
    eviction="reuse",
    prefetch_utilization_cap=0.75,
    anchor_prefix_blocks=4,
    recency_tau_s=600.0,
)
