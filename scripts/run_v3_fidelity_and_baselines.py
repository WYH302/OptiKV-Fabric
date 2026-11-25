#!/usr/bin/env python3
"""V3 Physics Fidelity & 2026 Frontier Baselines Suite for OptiKV-Fabric.

Produces:
  1. results/sim/exp_eviction_comparison.csv (6 policies: LRU, RR-Evict '26, SIEVE '24,
     S3-FIFO '23, Prefix-GDSF, Belady on single-GPU 2 GiB LMCache 120 sessions).
  2. results/sim/exp_v3_fidelity.csv (Strict Physics Fidelity Stress Verification:
     Base vs. Bipartite OCS Port Contention vs. Decode Continuous-Batching Queue Coupling
     vs. Full Strict Physics (Bipartite + Decode) vs. WDM 2x/4x Wavelength Striping,
     across LMCache 120, FreeInference 25, and 128K Bimodal).
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from optikv.metrics import SUMMARY_COLUMNS, write_csv
from optikv.simulator import run_simulation
from run_full_suite import (
    SIM,
    load_freeinference_long,
    load_lmcache_tokenized,
    make_cluster,
)
from run_suites_3_to_6_refined import make_agentic_closedloop

_CACHE = {}


def _get_trace(key: str):
    if key not in _CACHE:
        if key == "lm120":
            _CACHE[key] = load_lmcache_tokenized(max_sessions=120, session_launch_gap_s=0.5)
        elif key == "fi25":
            _CACHE[key] = load_freeinference_long(max_sessions=25, session_launch_gap_s=1.0)
        elif key == "bim128":
            _CACHE[key] = make_agentic_closedloop(
                name="sens-128k-bimodal",
                context_tokens=131072,
                hit_rate=0.996,
                n_sessions=24,
                turns_per_session=8,
                gap_s=1.5,
                bimodal_gaps=True,
                workers=8,
                include_short_probes=True,
            )
    return _CACHE[key]


def _run_job(job: dict) -> dict:
    tr = _get_trace(job["trace_key"])
    res = run_simulation(job["config"], trace=tr)
    row = dict(res.summary_rows[0])
    row.update(job["meta"])
    return row


def main() -> int:
    t0 = time.perf_counter()
    print("=== V3 Physics Fidelity & 2026 Frontier Eviction Suite ===", flush=True)

    # 1. Single-GPU 2 GiB Eviction with 6 policies (incl. 2026 RR-Evict + 2024 SIEVE)
    cl_ev = make_cluster(workers=1, hbm_gib=2.0, dram_gib=0.0, ssd_gib=0.0, pool_gib=0.0, rdma_bw=25.0, ocs_bw=200.0, tau_us=25000.0, links=2)
    ev_specs = [
        (0, "LRU (Local-HBM)", "local_hbm"),
        (1, "RR-Evict ('26)", "eviction_rrevict"),
        (2, "SIEVE ('24)", "eviction_sieve"),
        (3, "S3-FIFO ('23)", "eviction_s3fifo"),
        (4, "Prefix-GDSF (Ours)", "eviction_gdsf"),
        (5, "Belady (Optimal)", "eviction_belady"),
    ]
    ev_jobs = [
        {
            "trace_key": "lm120",
            "config": {
                "model": "qwen2.5-7b-instruct-1m",
                "hardware_profile": "h100_peak_spec",
                "cluster": cl_ev,
                "policies": [pol],
                "slo_ttft_s": 0.25,
                "seed": 0,
            },
            "meta": {"ev_idx": idx, "eviction_label": lbl},
        }
        for idx, lbl, pol in ev_specs
    ]

    # 2. Physics Fidelity Stress Suite (Bipartite Port Contention + Decode Coupling + WDM Striping)
    regimes = [
        ("base", "Base (Shared OCS)", False, False, 1),
        ("bipartite", "+Bipartite Port", True, False, 1),
        ("decode", "+Decode Queue", False, True, 1),
        ("strict_full", "Strict (Bip+Dec)", True, True, 1),
        ("strict_wdm2", "Strict + WDM-2x", True, True, 2),
        ("strict_wdm4", "Strict + WDM-4x", True, True, 4),
    ]
    pols = ["rdma_routing", "rdma_prefetch", "concur_rdma", "optical_naive", "optikv", "oracle_infinite_hbm"]
    workloads = [
        ("lm120", "LMCache-120", 4.0, 64.0, 256.0, 0.25),
        ("fi25", "FreeInference-25", 8.0, 64.0, 256.0, 1.0),
        ("bim128", "128K-Bimodal", 10.0, 16.0, 512.0, 0.25),
    ]
    fid_jobs = []
    for wkey, wlbl, hbm_g, dram_g, pool_g, slo in workloads:
        for rkey, rlbl, bip, dec, wdm in regimes:
            cl = make_cluster(
                workers=8,
                hbm_gib=hbm_g,
                dram_gib=dram_g,
                ssd_gib=256.0,
                pool_gib=pool_g,
                rdma_bw=25.0,
                ocs_bw=200.0,
                tau_us=25000.0,
                links=2,
            )
            cl["ocs"]["bipartite_port_contention"] = bip
            cl["ocs"]["wdm_wavelengths"] = wdm
            cl["couple_decode_queue"] = dec
            cl["decode_batch_slots"] = 16
            for pol in pols:
                # WDM-2x/4x only affects optical policies (skip redundant electrical repeats)
                if wdm > 1 and pol not in ("optikv", "optical_naive"):
                    continue
                fid_jobs.append(
                    {
                        "trace_key": wkey,
                        "config": {
                            "model": "qwen2.5-7b-instruct-1m",
                            "hardware_profile": "h100_peak_spec",
                            "cluster": cl,
                            "policies": [pol],
                            "slo_ttft_s": slo,
                            "seed": 0,
                        },
                        "meta": {
                            "workload": wlbl,
                            "fidelity_key": rkey,
                            "fidelity_label": rlbl,
                            "bipartite": bip,
                            "decode_coupled": dec,
                            "wdm_wavelengths": wdm,
                        },
                    }
                )

    with ProcessPoolExecutor(max_workers=8) as pool:
        ev_rows = list(pool.map(_run_job, ev_jobs))
        fid_rows = list(pool.map(_run_job, fid_jobs))

    ev_rows.sort(key=lambda r: r["ev_idx"])
    write_csv(SIM / "exp_eviction_comparison.csv", ev_rows, ["ev_idx", "eviction_label"] + SUMMARY_COLUMNS)
    print("\n--- Single-GPU 2 GiB Eviction Results ---")
    for r in ev_rows:
        print(f"  {r['eviction_label']:20s}: HBM Hit = {r['hbm_hit_rate']*100:6.2f}% | P50 = {r['p50_ttft_s']:6.2f}s")

    fid_cols = ["workload", "fidelity_key", "fidelity_label", "bipartite", "decode_coupled", "wdm_wavelengths", "inflight_joins"] + SUMMARY_COLUMNS
    write_csv(SIM / "exp_v3_fidelity.csv", fid_rows, fid_cols)
    print("\n--- Physics Fidelity Verification Summary (P50 TTFT ms / Duty %) ---")
    for wlbl in ("LMCache-120", "FreeInference-25", "128K-Bimodal"):
        print(f"\n[{wlbl}]")
        for rkey, rlbl, _, _, _ in regimes:
            sub = {r["policy"]: r for r in fid_rows if r["workload"] == wlbl and r["fidelity_key"] == rkey}
            parts = []
            for p in pols:
                if p in sub:
                    r = sub[p]
                    d_str = f" ({r['ocs_duty_cycle']*100:.0f}%)" if p in ("optikv", "optical_naive") else ""
                    parts.append(f"{p}={r['p50_ttft_s']*1e3:.1f}ms{d_str}")
            print(f"  {rlbl:18s}: " + " | ".join(parts))

    print(f"\nDone in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
