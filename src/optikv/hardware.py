"""Cluster capacities, optical-port parameters, and the cited cost ledger."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from optikv.modelspec import ROOT

# Cited from arXiv:2610.18444. These are not measurements from this project.
COST_SOURCE = "arXiv:2610.18444"
OCS_PORT_POWER_W = 0.39
OCS_PORT_COST_USD = 520.0
ELECTRICAL_400G_PORT_POWER_W = 39.4
ELECTRICAL_400G_PORT_COST_USD_LOW = 583.0
ELECTRICAL_400G_PORT_COST_USD_HIGH = 1090.0
LEGAL_CONCURRENT_LINKS = (1, 2, 4, 8)


@dataclass(frozen=True)
class CostLedger:
    ocs_port_power_w: float = OCS_PORT_POWER_W
    ocs_port_cost_usd: float = OCS_PORT_COST_USD
    electrical_400g_port_power_w: float = ELECTRICAL_400G_PORT_POWER_W
    electrical_400g_port_cost_usd_low: float = ELECTRICAL_400G_PORT_COST_USD_LOW
    electrical_400g_port_cost_usd_high: float = ELECTRICAL_400G_PORT_COST_USD_HIGH
    source: str = COST_SOURCE

    def as_columns(self) -> dict[str, float | str]:
        return {
            "ocs_port_power_w": self.ocs_port_power_w,
            "ocs_port_cost_usd": self.ocs_port_cost_usd,
            "electrical_400g_port_power_w": self.electrical_400g_port_power_w,
            "electrical_400g_port_cost_usd_low": self.electrical_400g_port_cost_usd_low,
            "electrical_400g_port_cost_usd_high": self.electrical_400g_port_cost_usd_high,
            "cost_source": self.source,
        }


def load_cost_ledger(path: Path | str | None = None) -> CostLedger:
    if path is None:
        path = ROOT / "configs" / "hardware" / "cost_ledger.yaml"
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return CostLedger()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return CostLedger(
        ocs_port_power_w=float(raw.get("ocs_port_power_w", OCS_PORT_POWER_W)),
        ocs_port_cost_usd=float(raw.get("ocs_port_cost_usd", OCS_PORT_COST_USD)),
        electrical_400g_port_power_w=float(raw.get("electrical_400g_port_power_w", ELECTRICAL_400G_PORT_POWER_W)),
        electrical_400g_port_cost_usd_low=float(
            raw.get("electrical_400g_port_cost_usd_low", ELECTRICAL_400G_PORT_COST_USD_LOW)
        ),
        electrical_400g_port_cost_usd_high=float(
            raw.get("electrical_400g_port_cost_usd_high", ELECTRICAL_400G_PORT_COST_USD_HIGH)
        ),
        source=str(raw.get("source", COST_SOURCE)),
    )


@dataclass(frozen=True)
class Cluster:
    num_workers: int
    hbm_bytes_per_worker: int
    dram_bytes_per_worker: int
    ssd_bytes_per_worker: int
    rdma_pool_bytes: int
    dram_bandwidth_gb_s: float
    ssd_bandwidth_gb_s: float
    rdma_bandwidth_gb_s: float
    rdma_latency_us: float
    alpha_overlap: float
    ocs_concurrent_links: int
    ocs_bandwidth_gb_s: float
    ocs_latency_us: float
    ocs_reconfig_us: float
    bipartite_port_contention: bool = False
    wdm_wavelengths: int = 1
    couple_decode_queue: bool = False
    decode_batch_slots: int = 16

    def __post_init__(self) -> None:
        if self.ocs_concurrent_links not in LEGAL_CONCURRENT_LINKS:
            raise ValueError(
                f"ocs concurrent links must be one of {LEGAL_CONCURRENT_LINKS}, "
                f"got {self.ocs_concurrent_links}"
            )
        if not 0.0 <= self.alpha_overlap <= 1.0:
            raise ValueError("alpha_overlap must be in [0, 1]")

    @property
    def reconfig_s(self) -> float:
        return self.ocs_reconfig_us * 1e-6

    @property
    def ocs_latency_s(self) -> float:
        return self.ocs_latency_us * 1e-6

    @property
    def rdma_latency_s(self) -> float:
        return self.rdma_latency_us * 1e-6

    @property
    def ocs_bandwidth_B_s(self) -> float:
        return self.ocs_bandwidth_gb_s * 1e9


def cluster_from_dict(raw: dict) -> Cluster:
    ocs = raw.get("ocs", {})
    return Cluster(
        num_workers=int(raw["num_workers"]),
        hbm_bytes_per_worker=int(raw["hbm_bytes_per_worker"]),
        dram_bytes_per_worker=int(raw["dram_bytes_per_worker"]),
        ssd_bytes_per_worker=int(raw["ssd_bytes_per_worker"]),
        rdma_pool_bytes=int(raw["rdma_pool_bytes"]),
        dram_bandwidth_gb_s=float(raw.get("dram_bandwidth_gb_s", 50.0)),
        ssd_bandwidth_gb_s=float(raw.get("ssd_bandwidth_gb_s", 3.0)),
        rdma_bandwidth_gb_s=float(raw.get("rdma_bandwidth_gb_s", 25.0)),
        rdma_latency_us=float(raw.get("rdma_latency_us", 10.0)),
        alpha_overlap=float(raw.get("alpha_overlap", 0.7)),
        ocs_concurrent_links=int(ocs.get("concurrent_links", 1)),
        ocs_bandwidth_gb_s=float(ocs.get("bandwidth_gb_s", 45.0)),
        ocs_latency_us=float(ocs.get("latency_us", 5.0)),
        ocs_reconfig_us=float(ocs.get("reconfig_us", 25000.0)),
        bipartite_port_contention=bool(ocs.get("bipartite_port_contention", raw.get("bipartite_port_contention", False))),
        wdm_wavelengths=int(ocs.get("wdm_wavelengths", raw.get("wdm_wavelengths", 1))),
        couple_decode_queue=bool(raw.get("couple_decode_queue", False)),
        decode_batch_slots=int(raw.get("decode_batch_slots", 16)),
    )


def byte_transfer_s(nbytes: int, bandwidth_gb_s: float, latency_us: float = 0.0) -> float:
    """Decimal GB/s: 1 GB/s = 1e9 B/s. A zero-byte move costs nothing."""
    if nbytes <= 0:
        return 0.0
    return latency_us * 1e-6 + nbytes / (bandwidth_gb_s * 1e9)
