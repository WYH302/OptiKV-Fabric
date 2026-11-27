#!/usr/bin/env python3
"""Round-2 (v3.1) Strict-Physics Additions:
  1. Per-worker & per-request queue diagnostic on FreeInference-250 (OptiKV 1:1 vs 4:1)
     to verify the exact quantitative mechanism behind 4:1's lower P99 tail latency.
  2. Run OptiKV (4:1, P=2) and full 400G line-rate RDMA (50 GB/s: rdma50_routing, rdma50_prefetch)
     on the 128K Load Sweep (1..32 req/s) -> updates exp_load_sweep.csv idempotently.
  3. Run full 400G line-rate RDMA (50 GB/s: rdma50_routing, rdma50_prefetch) on the
     Main Real-Trace Suite (Mooncake SiPh/MEMS, Full 767-sess LMCache, 250-sess FreeInference,
     128K Teaser) and 4-32 GPU Weak Scaling -> updates CSVs idempotently.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from optikv.metrics import SUMMARY_COLUMNS, write_csv
from optikv.simulator import run_simulation
from run_full_suite import SIM
from run_suites_3_to_6_refined import make_agentic_closedloop
from run_v3_strict_matching_suite import _get_trace, _strict_cluster

_LOCAL_SYN_CACHE: dict[tuple, object] = {}


def _run_job(job: dict) -> dict:
    if "syn_spec" in job:
        spec = job["syn_spec"]
        k = tuple(sorted(spec.items()))
        if k not in _LOCAL_SYN_CACHE:
            _LOCAL_SYN_CACHE[k] = make_agentic_closedloop(**spec)
        tr = _LOCAL_SYN_CACHE[k]
    else:
        tr = _get_trace(job["trace_key"])
    t0 = time.perf_counter()
    res = run_simulation(job["config"], trace=tr)
    wall_s = time.perf_counter() - t0
    row = dict(res.summary_rows[0])
    if "policy_override" in job:
        row["policy"] = job["policy_override"]
    row.update(job.get("meta", {}))
    row["sim_wall_s"] = wall_s
    row["ctrl_ms_per_req"] = (wall_s * 1e3) / max(1, int(row["num_requests"]))
    return row


def _merge_csv(path: Path, new_rows: list[dict], key_cols: list[str], extra_cols: list[str] | None = None) -> None:
    df_old = pd.read_csv(path)
    new_keys = {tuple(str(r[c]) for c in key_cols) for r in new_rows}
    keep_mask = [
        tuple(str(v) for v in vals) not in new_keys
        for vals in df_old[key_cols].itertuples(index=False, name=None)
    ]
    df_keep = df_old[keep_mask]
    cols = list(df_old.columns)
    if extra_cols:
        for c in extra_cols:
            if c not in cols:
                cols.append(c)
    merged = df_keep.to_dict("records") + new_rows
    write_csv(path, merged, cols)


def diagnose_freeinference_4to1() -> None:
    print("\n=== [1/4] FreeInference-250 Worker-Queue Diagnostic (1:1 vs 4:1) ===", flush=True)
    tr = _get_trace("fi_250")
    for label, links in [("OptiKV 1:1 (P=8)", 8), ("OptiKV 4:1 (P=2)", 2)]:
        cl = _strict_cluster(
            workers=8, hbm_gib=16.0, dram_gib=64.0, pool_gib=1024.0,
            ocs_bw=200.0, tau_us=25000.0, links=links,
        )
        res = run_simulation(
            {
                "model": "qwen2.5-7b-instruct-1m",
                "hardware_profile": "h100_peak_spec",
                "cluster": cl,
                "policies": ["optikv"],
                "slo_ttft_s": 1.0,
                "seed": 0,
            },
            trace=tr,
        )
        sr = res.summary_rows[0]
        reqs = res.request_rows
        by_w_cnt: dict[int, int] = defaultdict(int)
        by_w_pref: dict[int, float] = defaultdict(float)
        by_w_q: dict[int, list[float]] = defaultdict(list)
        q_ge_1 = 0
        q_ge_10 = 0
        ttfts = sorted(float(r["ttft_s"]) for r in reqs)
        top1_idx = int(0.99 * len(reqs))
        top1_reqs = sorted(reqs, key=lambda r: float(r["ttft_s"]))[top1_idx:]
        top1_q_share = sum(float(r["queue_s"]) for r in top1_reqs) / max(1e-9, sum(float(r["ttft_s"]) for r in top1_reqs))
        top1_p_share = sum(float(r["prefill_s"]) for r in top1_reqs) / max(1e-9, sum(float(r["ttft_s"]) for r in top1_reqs))
        top1_t_share = sum(float(r["transfer_s"]) for r in top1_reqs) / max(1e-9, sum(float(r["ttft_s"]) for r in top1_reqs))
        for r in reqs:
            w = int(r["worker"])
            q = float(r["queue_s"])
            by_w_cnt[w] += 1
            by_w_pref[w] += float(r["prefill_s"])
            by_w_q[w].append(q)
            if q >= 1.0:
                q_ge_1 += 1
            if q >= 10.0:
                q_ge_10 += 1
        cnts = [by_w_cnt[w] for w in range(8)]
        prefs = [by_w_pref[w] for w in range(8)]
        w_max_q = [max(by_w_q[w]) if by_w_q[w] else 0.0 for w in range(8)]
        cnt_cv = statistics.pstdev(cnts) / max(1e-9, statistics.mean(cnts))
        pref_cv = statistics.pstdev(prefs) / max(1e-9, statistics.mean(prefs))
        print(
            f"  {label}: P50={sr['p50_ttft_s']*1e3:.1f}ms P99={sr['p99_ttft_s']:.2f}s "
            f"SLO={sr['slo_attainment']*100:.2f}% MeanQ={sr['mean_queue_s']:.3f}s "
            f"Duty={sr['ocs_duty_cycle']*100:.1f}% Aborts={sr['abort_count']} PF={sr['prefetch_bursts']}",
            flush=True,
        )
        print(
            f"    Worker req counts: {cnts} (CV={cnt_cv:.3f}, max/min={max(cnts)/max(1,min(cnts)):.2f}x)",
            flush=True,
        )
        print(
            f"    Worker prefill s : {[round(x, 1) for x in prefs]} (CV={pref_cv:.3f}, max/min={max(prefs)/max(1e-6,min(prefs)):.2f}x)",
            flush=True,
        )
        print(
            f"    Worker max Q (s) : {[round(x, 1) for x in w_max_q]} | Q>=1s: {q_ge_1} ({100*q_ge_1/len(reqs):.2f}%) | Q>=10s: {q_ge_10} ({100*q_ge_10/len(reqs):.2f}%)",
            flush=True,
        )
        print(
            f"    Top-1% tail TTFT composition: Queue={top1_q_share*100:.1f}%, Prefill={top1_p_share*100:.1f}%, Transfer={top1_t_share*100:.1f}%",
            flush=True,
        )


def main() -> int:
    t0_all = time.perf_counter()
    diagnose_freeinference_4to1()

    with ProcessPoolExecutor(max_workers=8) as pool:
        # ------------------------------------------------------------------
        # [2/4] 128K Load Sweep Additions: optikv_4to1 + rdma50_routing + rdma50_prefetch
        # ------------------------------------------------------------------
        print("\n=== [2/4] 128K Load Sweep: OptiKV (4:1) + 400G Line-Rate RDMA (50 GB/s) ===", flush=True)
        t0 = time.perf_counter()
        j_ld = []
        cl_41 = _strict_cluster(workers=8, hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, links=2)
        cl_r50 = _strict_cluster(workers=8, hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, rdma_bw=50.0, ocs_bw=200.0, tau_us=25000.0, links=8)
        for rps in [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]:
            syn = dict(name=f"load-128k-r{int(rps)}", context_tokens=131072, hit_rate=255.0 / 256.0, n_sessions=36, turns_per_session=6, workers=8, steady_rps=rps)
            j_ld.append({
                "syn_spec": syn,
                "policy_override": "optikv_4to1",
                "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_41, "policies": ["optikv"], "slo_ttft_s": 0.25, "seed": 0},
                "meta": {"target_rps": rps},
            })
            for base_pol, ov_pol in [("rdma_routing", "rdma50_routing"), ("rdma_prefetch", "rdma50_prefetch")]:
                j_ld.append({
                    "syn_spec": syn,
                    "policy_override": ov_pol,
                    "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_r50, "policies": [base_pol], "slo_ttft_s": 0.25, "seed": 0},
                    "meta": {"target_rps": rps},
                })
        r_ld = list(pool.map(_run_job, j_ld))
        _merge_csv(SIM / "exp_load_sweep.csv", r_ld, key_cols=["target_rps", "policy"])
        print(f"  Load sweep additions done in {time.perf_counter()-t0:.1f}s:", flush=True)
        for pol in ("optikv_4to1", "rdma50_routing", "rdma50_prefetch"):
            sub = [r for r in r_ld if r["policy"] == pol]
            peak_gp = max(r["goodput_rps"] for r in sub)
            r16 = [r for r in sub if r["target_rps"] == 16.0][0]
            r32 = [r for r in sub if r["target_rps"] == 32.0][0]
            print(
                f"    {pol:16s}: PeakGP={peak_gp:.2f} req/s | @16: P50={r16['p50_ttft_s']*1e3:.1f}ms GP={r16['goodput_rps']:.2f} Duty={r16['ocs_duty_cycle']*100:.1f}% | "
                f"@32: P50={r32['p50_ttft_s']*1e3:.1f}ms GP={r32['goodput_rps']:.2f}",
                flush=True,
            )

        # ------------------------------------------------------------------
        # [3/4] Main Real-Trace Suite: 400G Line-Rate RDMA (50 GB/s)
        # ------------------------------------------------------------------
        print("\n=== [3/4] Main Real-Trace Benchmarks: 400G Line-Rate RDMA (50 GB/s) ===", flush=True)
        t0 = time.perf_counter()
        main_r50_specs = [
            ("exp_main_mooncake_h100_siph.csv", "mc_full", _strict_cluster(workers=8, hbm_gib=4.0, dram_gib=16.0, pool_gib=256.0, rdma_bw=50.0, ocs_bw=200.0, tau_us=10.0, links=8), 0.25),
            ("exp_main_mooncake_h100_mems.csv", "mc_full", _strict_cluster(workers=8, hbm_gib=4.0, dram_gib=16.0, pool_gib=256.0, rdma_bw=50.0, ocs_bw=200.0, tau_us=25000.0, links=8), 0.25),
            ("exp_main_lmcache_full_mems.csv", "lm_full", _strict_cluster(workers=8, hbm_gib=4.0, dram_gib=32.0, pool_gib=512.0, rdma_bw=50.0, ocs_bw=200.0, tau_us=25000.0, links=8), 0.25),
            ("exp_main_freeinference_250sess_mems.csv", "fi_250", _strict_cluster(workers=8, hbm_gib=16.0, dram_gib=64.0, pool_gib=1024.0, rdma_bw=50.0, ocs_bw=200.0, tau_us=25000.0, links=8), 1.0),
            ("fig1_teaser_128k.csv", "teaser_128k", _strict_cluster(workers=8, hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, rdma_bw=50.0, ocs_bw=200.0, tau_us=25000.0, links=8), 0.25),
        ]
        j_main = []
        for fname, tkey, cl, slo in main_r50_specs:
            for base_pol, ov_pol in [("rdma_routing", "rdma50_routing"), ("rdma_prefetch", "rdma50_prefetch")]:
                j_main.append({
                    "trace_key": tkey,
                    "policy_override": ov_pol,
                    "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": [base_pol], "slo_ttft_s": slo, "seed": 0},
                    "meta": {"_target_csv": fname},
                })
        r_main = list(pool.map(_run_job, j_main))
        by_csv: dict[str, list[dict]] = defaultdict(list)
        for r in r_main:
            by_csv[r["_target_csv"]].append(r)
        for fname, rows in by_csv.items():
            _merge_csv(SIM / fname, rows, key_cols=["policy"])
            print(f"  {fname}:", flush=True)
            for r in rows:
                print(
                    f"    {r['policy']:16s}: P50={r['p50_ttft_s']*1e3:7.1f}ms P99={r['p99_ttft_s']*1e3:8.1f}ms "
                    f"Hit={r['hit_rate']*100:5.1f}% SLO={r['slo_attainment']*100:5.1f}% GP={r['goodput_rps']:.2f}",
                    flush=True,
                )
        print(f"  Main suite 50 GB/s RDMA done in {time.perf_counter()-t0:.1f}s", flush=True)

        # ------------------------------------------------------------------
        # [4/4] 4-32 GPU Weak Scaling: 400G Line-Rate RDMA (50 GB/s)
        # ------------------------------------------------------------------
        print("\n=== [4/4] 4-32 GPU Weak Scaling: 400G Line-Rate RDMA (50 GB/s) ===", flush=True)
        t0 = time.perf_counter()
        j_sc = []
        for w in [4, 8, 16, 32]:
            syn_sc = dict(name=f"scale-128k-w{w}", context_tokens=131072, hit_rate=0.996, n_sessions=4 * w, turns_per_session=6, workers=w, steady_rps=1.5 * w)
            cl_r50_w = _strict_cluster(workers=w, hbm_gib=10.0, dram_gib=16.0, pool_gib=64.0 * w, rdma_bw=50.0, ocs_bw=200.0, tau_us=25000.0, links=w, couple_decode=True)
            for base_pol, ov_pol in [("rdma_routing", "rdma50_routing"), ("rdma_prefetch", "rdma50_prefetch")]:
                j_sc.append({
                    "syn_spec": syn_sc,
                    "policy_override": ov_pol,
                    "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_r50_w, "policies": [base_pol], "slo_ttft_s": 0.25, "seed": 0},
                    "meta": {"sweep_type": "weak_scale_1to1", "workers": w, "ocs_ports": w, "oversub_ratio": 1.0, "offered_rps": 1.5 * w},
                })
        r_sc = list(pool.map(_run_job, j_sc))
        _merge_csv(SIM / "exp_v3_cluster_scale.csv", r_sc, key_cols=["sweep_type", "workers", "ocs_ports", "policy"])
        for r in r_sc:
            print(
                f"  W={r['workers']:2d} | {r['policy']:16s} | P50={r['p50_ttft_s']*1e3:6.1f}ms | GP={r['goodput_rps']:5.2f} | SLO={r['slo_attainment']*100:5.1f}%",
                flush=True,
            )
        print(f"  Scale-out 50 GB/s RDMA done in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"\n=== ALL V3.1 ROUND-2 ADDITIONS COMPLETED IN {time.perf_counter()-t0_all:.1f}s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
