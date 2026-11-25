"""Policy registry. `oracle` expands to the two infinite-resource variants."""

from __future__ import annotations

from policies.base import PolicySpec
from policies.concur import SPEC as CONCUR_RDMA
from policies.eviction_belady import BeladyEvictor
from policies.eviction_rrevict import RREvictEvictor
from policies.eviction_s3fifo import S3FIFOEvictor
from policies.eviction_sieve import SIEVEEvictor
from policies.local_dram import SPEC as LOCAL_DRAM
from policies.local_hbm import SPEC as LOCAL_HBM
from policies.local_ssd import SPEC as LOCAL_SSD
from policies.no_cache import SPEC as NO_CACHE
from policies.optical_naive import SPEC as OPTICAL_NAIVE
from policies.optical_reservation import SPEC as OPTICAL_RESERVATION
from policies.optikv import build as build_optikv
from policies.oracle import INFINITE_BANDWIDTH, INFINITE_HBM
from policies.rdma_prefetch import SPEC as RDMA_PREFETCH
from policies.rdma_routing import SPEC as RDMA_ROUTING

_FIXED = {
    NO_CACHE.name: NO_CACHE,
    LOCAL_HBM.name: LOCAL_HBM,
    LOCAL_DRAM.name: LOCAL_DRAM,
    LOCAL_SSD.name: LOCAL_SSD,
    RDMA_ROUTING.name: RDMA_ROUTING,
    RDMA_PREFETCH.name: RDMA_PREFETCH,
    CONCUR_RDMA.name: CONCUR_RDMA,
    OPTICAL_NAIVE.name: OPTICAL_NAIVE,
    OPTICAL_RESERVATION.name: OPTICAL_RESERVATION,
    INFINITE_BANDWIDTH.name: INFINITE_BANDWIDTH,
    INFINITE_HBM.name: INFINITE_HBM,
}

LOCAL_S3 = PolicySpec(
    name="eviction_s3fifo",
    use_hbm=True,
    store=True,
    eviction="s3fifo",
)
LOCAL_SIEVE = PolicySpec(
    name="eviction_sieve",
    use_hbm=True,
    store=True,
    eviction="sieve",
)
LOCAL_RREVICT = PolicySpec(
    name="eviction_rrevict",
    use_hbm=True,
    store=True,
    eviction="rrevict",
)
LOCAL_GDSF = PolicySpec(
    name="eviction_gdsf",
    use_hbm=True,
    store=True,
    eviction="reuse",
)
LOCAL_BELADY = PolicySpec(
    name="eviction_belady",
    use_hbm=True,
    store=True,
    eviction="belady",
)
_FIXED[LOCAL_S3.name] = LOCAL_S3
_FIXED[LOCAL_SIEVE.name] = LOCAL_SIEVE
_FIXED[LOCAL_RREVICT.name] = LOCAL_RREVICT
_FIXED[LOCAL_GDSF.name] = LOCAL_GDSF
_FIXED[LOCAL_BELADY.name] = LOCAL_BELADY

POLICY_NAMES = [
    "no_cache",
    "local_hbm",
    "local_dram",
    "local_ssd",
    "rdma_routing",
    "rdma_prefetch",
    "concur_rdma",
    "optical_naive",
    "optical_reservation",
    "optikv",
    "oracle_infinite_bandwidth",
    "oracle_infinite_hbm",
    "eviction_s3fifo",
    "eviction_sieve",
    "eviction_rrevict",
    "eviction_gdsf",
    "eviction_belady",
]


def expand_policy_names(names: list[str]) -> list[str]:
    expanded: list[str] = []
    for name in names:
        if name == "oracle":
            expanded.extend(["oracle_infinite_bandwidth", "oracle_infinite_hbm"])
        else:
            expanded.append(name)
    return expanded


def build_policy(name: str, optikv_flags: dict | None = None) -> PolicySpec:
    if name == "optikv":
        return build_optikv(optikv_flags)
    try:
        return _FIXED[name]
    except KeyError as exc:
        known = ", ".join(POLICY_NAMES)
        raise KeyError(f"unknown policy {name}. Known policies: {known}") from exc


__all__ = [
    "POLICY_NAMES",
    "BeladyEvictor",
    "PolicySpec",
    "RREvictEvictor",
    "S3FIFOEvictor",
    "SIEVEEvictor",
    "build_policy",
    "expand_policy_names",
]
