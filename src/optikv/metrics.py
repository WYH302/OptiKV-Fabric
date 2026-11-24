"""Result tables and SLO/goodput/cost metrics. Figures read these CSVs verbatim."""

from __future__ import annotations

import csv
from pathlib import Path

SUMMARY_COLUMNS = [
    "run_id",
    "policy",
    "model",
    "dtype_bytes",
    "kv_bytes_per_token",
    "trace_name",
    "trace_kind",
    "seed",
    "num_requests",
    "num_workers",
    "hbm_bytes_per_worker",
    "dram_bytes_per_worker",
    "ssd_bytes_per_worker",
    "rdma_pool_bytes",
    "ocs_concurrent_links",
    "bandwidth_gb_s",
    "latency_us",
    "reconfig_us",
    "rdma_bandwidth_gb_s",
    "rdma_latency_us",
    "dram_bandwidth_gb_s",
    "ssd_bandwidth_gb_s",
    "alpha_overlap",
    "hardware_profile",
    "scaling_label",
    "calibration_status",
    "quadratic_extrapolation",
    "reuse_noise_std",
    "enable_routing",
    "enable_recompute_fallback",
    "enable_placement",
    "enable_prefetch",
    "enable_eviction",
    "prefetch_utilization_cap",
    "input_tokens",
    "recomputed_tokens",
    "hit_rate",
    "hbm_hit_rate",
    "dram_hit_rate",
    "ssd_hit_rate",
    "rdma_hit_rate",
    "remote_hbm_hit_rate",
    "optical_pool_hit_rate",
    "mean_ttft_s",
    "p50_ttft_s",
    "p95_ttft_s",
    "p99_ttft_s",
    "mean_queue_s",
    "mean_transfer_s",
    "mean_reconfig_s",
    "mean_prefill_s",
    "mean_decode_s",
    "mean_tpot_s",
    "abort_count",
    "prefetch_bursts",
    "prefetch_hit_tokens",
    "spurious_prefetch_bytes",
    "ocs_active_s",
    "ocs_reconfig_s",
    "ocs_duty_cycle",
    "standing_circuit_hits",
    "slo_ttft_s",
    "slo_attainment",
    "goodput_rps",
    "goodput_per_watt",
    "goodput_per_dollar",
    "evictions",
    "pinned_evictions",
    "transfer_bytes",
    "ocs_port_power_w",
    "ocs_port_cost_usd",
    "electrical_400g_port_power_w",
    "electrical_400g_port_cost_usd_low",
    "electrical_400g_port_cost_usd_high",
    "cost_source",
    "git_commit",
    "timestamp",
]

REQUEST_COLUMNS = [
    "run_id",
    "policy",
    "request_id",
    "session_id",
    "turn_index",
    "arrival_s",
    "worker",
    "input_tokens",
    "output_tokens",
    "hbm_hit_tokens",
    "dram_hit_tokens",
    "ssd_hit_tokens",
    "rdma_hit_tokens",
    "remote_hbm_hit_tokens",
    "optical_pool_hit_tokens",
    "transferred_tokens",
    "recomputed_tokens",
    "hit_rate",
    "ttft_s",
    "queue_s",
    "transfer_s",
    "reconfig_s",
    "prefill_s",
    "decode_s",
    "aborted",
    "prefetch_applied",
    "calibration_status",
]


def layer_ttft(queue_s: float, transfer_s: float, prefill_s: float, decode_s: float, alpha: float) -> float:
    """TTFT = T_queue + max(T_transfer, alpha * T_prefill_delta) + (1 - alpha) * T_prefill_delta + T_first_decode.

    alpha = 0 is fully blocking. Default study value is 0.7 (layer-wise streaming overlap).
    """
    overlapped = max(transfer_s, alpha * prefill_s) + (1.0 - alpha) * prefill_s
    return queue_s + overlapped + decode_s


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def write_csv(path: Path | str, rows: list[dict], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
