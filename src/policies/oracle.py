"""Lower-bound oracles.

Both variants treat every block seen in an earlier request as available at
zero transfer time, so neither can lose to a policy that still has to move
or recompute those bytes.

oracle_infinite_bandwidth fetches any historical block in zero time.
oracle_infinite_hbm retains every historical block and also pays zero time
to have it on the worker that runs the request. Their TTFT expressions match;
the names record which resource the bound removes.
"""

from policies.base import PolicySpec


def _spec(name: str) -> PolicySpec:
    return PolicySpec(
        name=name,
        use_hbm=True,
        oracle=True,
        store=True,
        routing=False,
        eviction="lru",
        recompute_fallback=False,
        placement=False,
        prefetch=False,
    )


INFINITE_BANDWIDTH = _spec("oracle_infinite_bandwidth")
INFINITE_HBM = _spec("oracle_infinite_hbm")
