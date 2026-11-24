#!/usr/bin/env python3
"""CPU Sensitivity Sweep over Layer-Wise Compute-Transfer Overlap Factor alpha_ovl in [0.0, 1.0]."""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from optikv.metrics import SUMMARY_COLUMNS, write_csv
from run_full_suite import SIM, _run_one_job, make_cluster
from run_suites_3_to_6_refined import make_agentic_closedloop


def main() -> int:
    t0 = time.perf_counter()
    tr = make_agentic_closedloop(
        name="ovl-128k-bimodal",
        context_tokens=131072,
        hit_rate=255.0 / 256.0,
        n_sessions=24,
        turns_per_session=8,
        gap_s=1.5,
        bimodal_gaps=True,
        workers=8,
        include_short_probes=True,
    )
    alphas = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]
    pols = ["rdma_routing", "rdma_prefetch", "optical_naive", "optikv", "oracle_infinite_hbm"]
    jobs = []
    for a in alphas:
        cl = make_cluster(hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0, links=2, alpha=a)
        jobs.append(
            {
                "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": pols, "slo_ttft_s": 0.25, "seed": 0},
                "trace": tr,
                "meta": {"alpha_ovl": a},
            }
        )
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = [r for res in pool.map(_run_one_job, jobs) for r in res["rows"]]
    rows.sort(key=lambda r: (r["alpha_ovl"], pols.index(r["policy"])))
    out = SIM / "exp_sensitivity_overlap.csv"
    write_csv(out, rows, ["alpha_ovl"] + SUMMARY_COLUMNS)
    for a in alphas:
        sub = {r["policy"]: r for r in rows if r["alpha_ovl"] == a}
        print(
            f"  alpha={a:.2f} | RDMA+Rout={sub['rdma_routing']['p50_ttft_s']*1e3:6.1f}ms "
            f"RDMA+Pref={sub['rdma_prefetch']['p50_ttft_s']*1e3:5.1f}ms "
            f"OptiKV={sub['optikv']['p50_ttft_s']*1e3:5.1f}ms "
            f"(sp vs RDMA={sub['rdma_routing']['p50_ttft_s']/sub['optikv']['p50_ttft_s']:.2f}x, "
            f"duty={sub['optikv']['ocs_duty_cycle']*100:.1f}%)",
            flush=True,
        )
    print(f"Saved {out} in {time.perf_counter()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
