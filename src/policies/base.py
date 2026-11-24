"""Shared policy description. The simulator is request-level and reads these flags."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicySpec:
    name: str
    use_hbm: bool = False
    use_dram: bool = False
    use_ssd: bool = False
    use_rdma: bool = False
    use_remote_hbm: bool = False
    use_optical_pool: bool = False
    optical: str = "none"
    routing: bool = False
    recompute_fallback: bool = False
    placement: bool = False
    prefetch: bool = False
    eviction: str = "lru"
    store: bool = True
    oracle: bool = False
    book_at_arrival: bool = False
    prefetch_utilization_cap: float = 0.75
    anchor_prefix_blocks: int = 2
    recency_tau_s: float = 600.0
    tier_hbm_min_p: float = 0.5
    tier_dram_min_p: float = 0.25
    tier_ssd_min_p: float = 0.08

    @property
    def enable_eviction(self) -> bool:
        return self.eviction == "reuse"
