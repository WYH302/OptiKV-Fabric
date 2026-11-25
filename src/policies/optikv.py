"""OptiKV-Fabric: optical-aware routing, lossless recompute abort, value-based
multi-tier placement, windowed inter-turn burst prefetch, and reuse-aware eviction.

Causal online P(reuse) uses ONLY history observed at or before arrival:
- Agent Anchor detection (early system/tool prefix blocks or blocks seen >= 2 times)
- Second-observation retention (first-seen tail blocks enter probation; second hit promotes)
- Exponential recency decay in inter-turn session gap
- Optional Gaussian noise injection for prediction-error robustness testing
"""

from __future__ import annotations

import math

from policies.base import PolicySpec

DEFAULTS = {
    "enable_routing": True,
    "enable_recompute_fallback": True,
    "enable_placement": True,
    "enable_prefetch": True,
    "enable_eviction": True,
    "prefetch_utilization_cap": 0.75,
    "anchor_prefix_blocks": 4,
    "recency_tau_s": 300.0,
    "tier_hbm_min_p": 0.5,
    "tier_dram_min_p": 0.25,
    "tier_ssd_min_p": 0.08,
}


def p_reuse(
    obs_before: int,
    is_anchor: bool,
    session_gap_s: float,
    tau_s: float,
    noise: float = 0.0,
) -> float:
    """Strictly causal history-only reuse probability estimate in [0, 1]."""
    if is_anchor:
        base = 1.0
    elif obs_before >= 2:
        base = min(1.0, 0.70 + 0.10 * math.log1p(obs_before))
    elif obs_before == 1:
        base = 0.55
    else:
        # First-seen non-anchor tail block: probationary score
        base = 0.18
    if tau_s <= 0.0:
        decay = 1.0
    else:
        decay = math.exp(-max(float(session_gap_s), 0.0) / float(tau_s))
        if is_anchor:
            decay = max(0.35, decay)
    score = base * decay + float(noise)
    return max(0.0, min(1.0, score))


class ReuseEvictor:
    """Evict lowest EvictScore = P(reuse) * T_recompute_saved / KV_bytes."""

    def choose(self, entries, request_index: int):
        eligible = [entry for entry in entries if entry.pin_count == 0]
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda entry: (entry.evict_score, entry.reuse_score, entry.last_access_s, entry.block_id),
        )


def build(flags: dict | None = None) -> PolicySpec:
    merged = dict(DEFAULTS)
    if flags:
        merged.update(flags)
    placement = bool(merged["enable_placement"])
    return PolicySpec(
        name="optikv",
        use_hbm=True,
        use_dram=placement,
        use_ssd=placement,
        use_rdma=False,
        use_remote_hbm=True,
        use_optical_pool=placement,
        optical="keep",
        routing=bool(merged["enable_routing"]),
        recompute_fallback=bool(merged["enable_recompute_fallback"]),
        placement=placement,
        prefetch=bool(merged["enable_prefetch"]),
        eviction="reuse" if merged["enable_eviction"] else "lru",
        store=True,
        book_at_arrival=True,
        prefetch_utilization_cap=float(merged["prefetch_utilization_cap"]),
        anchor_prefix_blocks=int(merged["anchor_prefix_blocks"]),
        recency_tau_s=float(merged["recency_tau_s"]),
        tier_hbm_min_p=float(merged["tier_hbm_min_p"]),
        tier_dram_min_p=float(merged["tier_dram_min_p"]),
        tier_ssd_min_p=float(merged["tier_ssd_min_p"]),
    )
