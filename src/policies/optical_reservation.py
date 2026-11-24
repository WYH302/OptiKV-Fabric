"""Yoon & Yu (arXiv:2610.18444) style optical reservation baseline adapted to historical prefixes.

Maintains a standing circuit per OCS port (same peer pays tau = 0; switching to a
different peer physically pays tau) and books the optical circuit upon request
arrival so reconfiguration can overlap with worker compute queueing. Single-peer
remote HBM fetch only: no multi-tier DRAM/SSD/pool spill, no inter-turn burst
prefetch, no optical-aware routing, and no lossless recompute abort.
"""

from policies.base import PolicySpec

SPEC = PolicySpec(
    name="optical_reservation",
    use_hbm=True,
    use_remote_hbm=True,
    optical="reservation",
    book_at_arrival=True,
    recompute_fallback=False,
    placement=False,
    prefetch=False,
    routing=False,
    store=True,
    eviction="lru",
)
