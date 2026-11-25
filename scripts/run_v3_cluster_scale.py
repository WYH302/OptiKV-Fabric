#!/usr/bin/env python3
"""V3 Multi-Node Cluster Scale-Out & OCS Port-Oversubscription Sweep under Strict Physics.

Evaluates weak scaling from W=4 to W=32 GPUs (0.5 to 4 full 8-GPU HGX nodes)
under simultaneous strict bipartite source-destination port contention AND
continuous-batching decode queue coupling (16 slots), plus an OCS port
oversubscription sweep (P in {1, 2, 4, 8} at W=16 GPUs).

Outputs:
  results/sim/exp_v3_cluster_scale.csv
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
from run_full_suite import SIM, make_cluster
from run_suites_3_to_6_refined import make_agentic_closedloop

_TRACE_CACHE = {}


def _get_scale_trace(workers: int, rps_per_worker: float = 1.5):
    key = (workers, rps_per_worker)
    if key not in _TRACE_CACHE:
        n_sessions = 4 * workers
        steady_rps = rps_per_worker * workers
        _TRACE_CACHE[key] = make_agentic_closedloop(
            name=f"scale-128k-w{workers}-r{rps_per_worker:.1f}",
            context_tokens=131072,
            hit_rate=0.996,
            n_sessions=n_sessions,
            turns_per_session=6,
            workers=workers,
            steady_rps=steady_rps,
        )
    return _TRACE_CACHE[key]


def _run_job(job: dict) -> dict:
    tr = _get_scale_trace(job["workers"], job.get("rps_per_worker", 1.5))
    t0 = time.perf_counter()
    res = run_simulation(job["config"], trace=tr)
    wall_s = time.perf_counter() - t0
    row = dict(res.summary_rows[0])
    row.update(job["meta"])
    row["sim_wall_s"] = wall_s
    row["ctrl_ms_per_req"] = (wall_s * 1e3) / max(1, int(row["num_requests"]))
    return row


def main() -> int:
    t0 = time.perf_counter()
    print("=== V3 Strict-Physics Multi-Node Cluster Scale-Out (W=4..32) & Port Ratio Sweep ===", flush=True)

    jobs = []
    pols = ["rdma_routing", "concur_rdma", "rdma_prefetch", "optical_naive", "optikv", "oracle_infinite_hbm"]

    # Part 1: Weak scaling at constant 4:1 GPU-to-OCS-port ratio: (W, P) in {(4,1), (8,2), (16,4), (32,8)}
    for w, p in [(4, 1), (8, 2), (16, 4), (32, 8)]:
        cl = make_cluster(
            workers=w,
            hbm_gib=10.0,
            dram_gib=16.0,
            ssd_gib=256.0,
            pool_gib=64.0 * w,
            rdma_bw=25.0,
            ocs_bw=200.0,
            tau_us=25000.0,
            links=p,
        )
        cl["ocs"]["bipartite_port_contention"] = True
        cl["ocs"]["wdm_wavelengths"] = 1
        cl["couple_decode_queue"] = True
        cl["decode_batch_slots"] = 16
        for pol in pols:
            jobs.append(
                {
                    "workers": w,
                    "config": {
                        "model": "qwen2.5-7b-instruct-1m",
                        "hardware_profile": "h100_peak_spec",
                        "cluster": cl,
                        "policies": [pol],
                        "slo_ttft_s": 0.25,
                        "seed": 0,
                    },
                    "meta": {
                        "sweep_type": "weak_scale",
                        "workers": w,
                        "ocs_ports": p,
                        "oversub_ratio": w / p,
                        "offered_rps": 1.5 * w,
                    },
                }
            )

    # Part 2: Port-oversubscription sweep at W=16 GPUs (2 full 8-GPU HGX nodes, heavy 4.0 req/s/GPU = 64 req/s burst): P in {1, 2, 4, 8}
    for p in [1, 2, 4, 8]:
        w = 16
        cl = make_cluster(
            workers=w,
            hbm_gib=10.0,
            dram_gib=16.0,
            ssd_gib=256.0,
            pool_gib=64.0 * w,
            rdma_bw=25.0,
            ocs_bw=200.0,
            tau_us=25000.0,
            links=p,
        )
        cl["ocs"]["bipartite_port_contention"] = True
        cl["ocs"]["wdm_wavelengths"] = 1
        cl["couple_decode_queue"] = True
        cl["decode_batch_slots"] = 16
        for pol in ("optikv", "rdma_prefetch", "rdma_routing"):
            jobs.append(
                {
                    "workers": w,
                    "rps_per_worker": 4.0,
                    "config": {
                        "model": "qwen2.5-7b-instruct-1m",
                        "hardware_profile": "h100_peak_spec",
                        "cluster": cl,
                        "policies": [pol],
                        "slo_ttft_s": 0.25,
                        "seed": 0,
                    },
                    "meta": {
                        "sweep_type": "port_ratio_w16",
                        "workers": w,
                        "ocs_ports": p,
                        "oversub_ratio": w / p,
                        "offered_rps": 4.0 * w,
                    },
                }
            )

    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(_run_job, jobs))

    extra_cols = ["sweep_type", "workers", "ocs_ports", "oversub_ratio", "offered_rps", "sim_wall_s", "ctrl_ms_per_req"]
    out_path = SIM / "exp_v3_cluster_scale.csv"
    write_csv(out_path, rows, extra_cols + SUMMARY_COLUMNS)

    print("\n--- Weak Scaling (Strict Bipartite + Decode, W/P = 4:1, 128K Agentic, 250ms SLO) ---", flush=True)
    for r in rows:
        if r["sweep_type"] == "weak_scale":
            print(
                f"  W={r['workers']:2d} P={r['ocs_ports']} | {r['policy']:20s} | "
                f"P50={r['p50_ttft_s']*1e3:7.1f}ms | SLO={r['slo_attainment']*100:5.1f}% | "
                f"Goodput={r['goodput_rps']:6.2f} req/s | Duty={r['ocs_duty_cycle']*100:5.1f}% | "
                f"Ctrl={r['ctrl_ms_per_req']:.2f} ms/req",
                flush=True,
            )

    print("\n--- Port Oversubscription Sweep at W=16 GPUs (2x 8-GPU Nodes, Strict Bipartite + Decode) ---", flush=True)
    for r in rows:
        if r["sweep_type"] == "port_ratio_w16":
            print(
                f"  W=16 P={r['ocs_ports']} ({r['oversub_ratio']:4.1f}:1) | {r['policy']:15s} | "
                f"P50={r['p50_ttft_s']*1e3:7.1f}ms | SLO={r['slo_attainment']*100:5.1f}% | "
                f"Goodput={r['goodput_rps']:6.2f} req/s | Duty={r['ocs_duty_cycle']*100:5.1f}%",
                flush=True,
            )

    print(f"\nSaved {len(rows)} rows to {out_path} in {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
