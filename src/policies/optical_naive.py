"""Optical fetch that pays tau on every request.

The circuit is dropped after the move, so the next request pays tau even if
the peer is unchanged. This is the strawman. It does not place across tiers,
does not prefetch, and does not abort into recompute. A caller that keeps the
circuit, rather than this policy, still sees tau 0 for a repeated target.
"""

from policies.base import PolicySpec

SPEC = PolicySpec(
    name="optical_naive",
    use_hbm=True,
    use_remote_hbm=True,
    optical="naive",
    book_at_arrival=False,
    recompute_fallback=False,
    placement=False,
    prefetch=False,
    store=True,
    eviction="lru",
)
