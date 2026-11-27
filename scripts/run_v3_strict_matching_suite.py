#!/usr/bin/env python3
"""Master V3 Strict-Bipartite-Matching Re-Simulation Suite for OptiKV-Fabric.

Every simulation enforces 100% strict physical bipartite port matching conservation
(at any instant, at most min(W, P) workers hold standing circuits to the Optical Pool;
stealing a pool port tears down the previous worker's latch and pays tau).

Runs cleanly in parallel across CPU cores and regenerates all core CSVs:
  1. Main Real-Trace Suite:
     - Mooncake Full (23,608 reqs, SiPh 10 us & MEMS 25 ms, P=W 1:1 + P=2 4:1)
     - Mooncake RTX 3090 Calibrated (2,000 reqs, 45 GB/s MEMS, P=W 1:1 + P=2 4:1)
     - LMCache Full (ALL 767 sessions / 24,880 reqs, MEMS 25 ms, P=W 1:1 + P=2 4:1)
     - FreeInference Large-Scale Stratified (250 sessions / 21,230 reqs = 11.14% of week,
       165K median prompt, MEMS 25 ms, P=W 1:1 + P=2 4:1)
     - 128K Agentic Teaser (P=W 1:1 + P=2 4:1)
  2. Corrected 80-Cell Break-Even Heatmap (2K..1M x 0%..99.5%, strict matching)
  3. Hardware Sensitivity (tau, BW, Multi-Arch x FP8), 7-Step Ablation, Noise, Load Sweep (1..32 req/s)
  4. Strict Physics 4-32 GPU Multi-Node Weak Scaling (BOTH 1:1 P=W and 4:1 P=W/4!) +
     W=16 Port-Oversubscription Sweep (P in {1, 2, 4, 8, 16} -> 16:1 down to 1:1)
  5. Strict Fidelity & WDM Striping (1x, 2x, 4x) + 5-Fold Disjoint CV (all 767 LMCache + 250 FreeInference)
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from optikv.metrics import SUMMARY_COLUMNS, write_csv
from optikv.simulator import run_simulation
from optikv.trace import BlockRef, Request, Trace
from run_full_suite import (
    SIM,
    load_lmcache_tokenized,
    load_mooncake_full,
    make_cluster,
    pace_trace,
)
from run_suites_3_to_6_refined import make_agentic_closedloop

ALL_POLS = [
    "no_cache",
    "local_hbm",
    "local_dram",
    "rdma_routing",
    "concur_rdma",
    "rdma_prefetch",
    "optical_naive",
    "optical_reservation",
    "optikv",
    "oracle_infinite_hbm",
]

_TRACE_CACHE: dict[str, Trace] = {}


def load_freeinference_250(max_sessions: int | None = None, launch_gap_s: float = 2.0) -> Trace:
    path = ROOT / "datasets/freeinference-agentic-trace/stratified_250sess_512.parquet"
    tbl = pq.read_table(path)
    s_col = tbl.column("session_id").to_pylist()
    gaps = tbl.column("pre_gap").to_pylist()
    outs = tbl.column("output_length").to_pylist()
    bids = tbl.column("block_ids").to_pylist()
    blens = tbl.column("block_lens").to_pylist()

    uniq_s = list(dict.fromkeys(s_col))
    if max_sessions is not None:
        uniq_s = uniq_s[:max_sessions]
    pick = set(uniq_s)
    sess_t0 = {sid: i * launch_gap_s for i, sid in enumerate(uniq_s)}
    last_arr: dict[str, float] = {}
    turns: dict[str, int] = {}
    reqs: list[Request] = []
    for sid, g, out, bl, ll in zip(s_col, gaps, outs, bids, blens):
        if sid not in pick or not bl:
            continue
        t = turns.get(sid, 0)
        gap = 0.0 if t == 0 else max(0.0, float(g or 0.0))
        arr = sess_t0[sid] if t == 0 else (last_arr[sid] + max(0.05, gap))
        last_arr[sid] = arr
        turns[sid] = t + 1
        blocks = tuple(BlockRef(b, int(l)) for b, l in zip(bl, ll))
        reqs.append(Request(f"{sid}-{t}", arr, sid, t, gap, blocks, output_tokens=max(1, int(out or 1))))
    reqs.sort(key=lambda r: (r.arrival_s, r.request_id))
    return Trace(name=f"freeinference-{len(uniq_s)}sess", kind="freeinference", block_tokens=512, requests=reqs)


def _get_trace(key: str) -> Trace:
    if key in _TRACE_CACHE:
        return _TRACE_CACHE[key]
    if key == "mc_full":
        tr = load_mooncake_full()
    elif key == "mc_3090":
        mc = load_mooncake_full()
        tr = pace_trace(Trace("mooncake-2k-3090", mc.kind, mc.block_tokens, mc.requests[:2000]), target_rps=0.8, name="mooncake-2k-3090-paced")
    elif key == "lm_full":
        tr = load_lmcache_tokenized(max_sessions=None, session_launch_gap_s=0.8)
    elif key == "lm_120":
        tr = load_lmcache_tokenized(max_sessions=120, session_launch_gap_s=0.5)
    elif key == "fi_250":
        tr = load_freeinference_250(max_sessions=None, launch_gap_s=2.0)
    elif key == "fi_50":
        tr = load_freeinference_250(max_sessions=50, launch_gap_s=2.0)
    elif key.startswith("lm_fold_"):
        fold = int(key.split("_")[-1])
        full = _get_trace("lm_full")
        sids = list(dict.fromkeys(r.session_id for r in full.requests))
        f_sids = {s for i, s in enumerate(sids) if i % 5 == fold}
        sub_r = [r for r in full.requests if r.session_id in f_sids]
        tr = Trace(f"lm-fold-{fold}", "lmcache", full.block_tokens, sub_r)
    elif key.startswith("fi_fold_"):
        fold = int(key.split("_")[-1])
        full = _get_trace("fi_250")
        sids = list(dict.fromkeys(r.session_id for r in full.requests))
        f_sids = {s for i, s in enumerate(sids) if i % 5 == fold}
        sub_r = [r for r in full.requests if r.session_id in f_sids]
        tr = Trace(f"fi-fold-{fold}", "freeinference", full.block_tokens, sub_r)
    elif key == "teaser_128k":
        tr = make_agentic_closedloop(
            name="agentic-128k-512delta",
            context_tokens=131072,
            hit_rate=255.0 / 256.0,
            n_sessions=24,
            turns_per_session=8,
            gap_s=1.5,
            bimodal_gaps=True,
            workers=8,
        )
    elif key == "bimodal_128k_probes":
        tr = make_agentic_closedloop(
            name="agentic-128k-bimodal-probes",
            context_tokens=131072,
            hit_rate=255.0 / 256.0,
            n_sessions=24,
            turns_per_session=8,
            gap_s=1.5,
            bimodal_gaps=True,
            workers=8,
            include_short_probes=True,
        )
    else:
        raise ValueError(f"Unknown trace key: {key}")
    _TRACE_CACHE[key] = tr
    return tr


def _run_job(job: dict) -> dict:
    tr = job["trace"] if "trace" in job else _get_trace(job["trace_key"])
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


def _strict_cluster(
    workers: int = 8,
    hbm_gib: float = 10.0,
    dram_gib: float = 16.0,
    ssd_gib: float = 256.0,
    pool_gib: float = 512.0,
    rdma_bw: float = 25.0,
    ocs_bw: float = 200.0,
    tau_us: float = 25000.0,
    links: int = 8,
    wdm: int = 1,
    couple_decode: bool = False,
    dram_bw: float = 50.0,
) -> dict:
    cl = make_cluster(
        workers=workers,
        hbm_gib=hbm_gib,
        dram_gib=dram_gib,
        ssd_gib=ssd_gib,
        pool_gib=pool_gib,
        dram_bw=dram_bw,
        rdma_bw=rdma_bw,
        ocs_bw=ocs_bw,
        tau_us=tau_us,
        links=links,
    )
    cl["ocs"]["bipartite_port_contention"] = True
    cl["ocs"]["wdm_wavelengths"] = wdm
    cl["couple_decode_queue"] = couple_decode
    cl["decode_batch_slots"] = 16
    return cl


def run_batch(pool: ProcessPoolExecutor, jobs: list[dict]) -> list[dict]:
    futs = [pool.submit(_run_job, j) for j in jobs]
    return [f.result() for f in futs]


def main() -> int:
    t_all = time.perf_counter()
    workers_cpu = 8
    print(f"=== V3 STRICT BIPARTITE MATCHING SUITE (workers={workers_cpu}) ===", flush=True)

    with ProcessPoolExecutor(max_workers=workers_cpu) as pool:
        # ------------------------------------------------------------------
        # [1/6] Main Real-Trace Benchmarks (100% strict matching: 1:1 P=8 + 4:1 P=2)
        # ------------------------------------------------------------------
        print("\n[1/6] Main Real-Trace Suite (Mooncake, Full 767-sess LMCache, 250-sess FreeInference, 128K)...", flush=True)
        t0 = time.perf_counter()
        main_specs = [
            (
                "exp_main_mooncake_h100_siph.csv",
                "mc_full",
                "qwen2.5-7b-instruct-1m",
                "h100_peak_spec",
                _strict_cluster(workers=8, hbm_gib=4.0, dram_gib=16.0, pool_gib=256.0, ocs_bw=200.0, tau_us=10.0, links=8),
                _strict_cluster(workers=8, hbm_gib=4.0, dram_gib=16.0, pool_gib=256.0, ocs_bw=200.0, tau_us=10.0, links=2),
                0.25,
            ),
            (
                "exp_main_mooncake_h100_mems.csv",
                "mc_full",
                "qwen2.5-7b-instruct-1m",
                "h100_peak_spec",
                _strict_cluster(workers=8, hbm_gib=4.0, dram_gib=16.0, pool_gib=256.0, ocs_bw=200.0, tau_us=25000.0, links=8),
                _strict_cluster(workers=8, hbm_gib=4.0, dram_gib=16.0, pool_gib=256.0, ocs_bw=200.0, tau_us=25000.0, links=2),
                0.25,
            ),
            (
                "exp_main_mooncake_3090_calibrated.csv",
                "mc_3090",
                "qwen2.5-7b-instruct-1m",
                "rtx3090",
                _strict_cluster(workers=8, hbm_gib=4.0, dram_gib=16.0, pool_gib=256.0, dram_bw=26.28, rdma_bw=25.0, ocs_bw=45.0, tau_us=25000.0, links=8),
                _strict_cluster(workers=8, hbm_gib=4.0, dram_gib=16.0, pool_gib=256.0, dram_bw=26.28, rdma_bw=25.0, ocs_bw=45.0, tau_us=25000.0, links=2),
                2.0,
            ),
            (
                "exp_main_lmcache_full_mems.csv",
                "lm_full",
                "qwen2.5-7b-instruct-1m",
                "h100_peak_spec",
                _strict_cluster(workers=8, hbm_gib=4.0, dram_gib=32.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, links=8),
                _strict_cluster(workers=8, hbm_gib=4.0, dram_gib=32.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, links=2),
                0.25,
            ),
            (
                "exp_main_freeinference_250sess_mems.csv",
                "fi_250",
                "qwen2.5-7b-instruct-1m",
                "h100_peak_spec",
                _strict_cluster(workers=8, hbm_gib=16.0, dram_gib=64.0, pool_gib=1024.0, ocs_bw=200.0, tau_us=25000.0, links=8),
                _strict_cluster(workers=8, hbm_gib=16.0, dram_gib=64.0, pool_gib=1024.0, ocs_bw=200.0, tau_us=25000.0, links=2),
                1.0,
            ),
            (
                "fig1_teaser_128k.csv",
                "teaser_128k",
                "qwen2.5-7b-instruct-1m",
                "h100_peak_spec",
                _strict_cluster(workers=8, hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, links=8),
                _strict_cluster(workers=8, hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, links=2),
                0.25,
            ),
        ]
        for fname, tkey, model, hw, cl_1to1, cl_4to1, slo in main_specs:
            ts = time.perf_counter()
            jobs = []
            for idx, pol in enumerate(ALL_POLS):
                jobs.append({
                    "trace_key": tkey,
                    "config": {"model": model, "hardware_profile": hw, "cluster": cl_1to1, "policies": [pol], "slo_ttft_s": slo, "seed": 0},
                    "meta": {"order": idx},
                })
            # Also add 4:1 oversubscribed OptiKV (P=2 on 8 GPUs) to quantify exact matching oversubscription!
            jobs.append({
                "trace_key": tkey,
                "policy_override": "optikv_4to1",
                "config": {"model": model, "hardware_profile": hw, "cluster": cl_4to1, "policies": ["optikv"], "slo_ttft_s": slo, "seed": 0},
                "meta": {"order": len(ALL_POLS)},
            })
            rows = sorted(run_batch(pool, jobs), key=lambda r: r["order"])
            write_csv(SIM / fname, rows, SUMMARY_COLUMNS)
            print(f"  {fname} ({time.perf_counter()-ts:.1f}s):", flush=True)
            for r in rows:
                if r["policy"] in ("rdma_routing", "concur_rdma", "rdma_prefetch", "optical_naive", "optikv", "optikv_4to1", "oracle_infinite_hbm"):
                    print(f"    {r['policy']:20s}: P50={r['p50_ttft_s']*1e3:7.1f}ms P99={r['p99_ttft_s']*1e3:8.1f}ms Hit={r['hit_rate']*100:5.1f}% Duty={r['ocs_duty_cycle']*100:5.1f}% SLO={r['slo_attainment']*100:5.1f}%", flush=True)
        print(f"[1/6] Done in {time.perf_counter()-t0:.1f}s", flush=True)

        # ------------------------------------------------------------------
        # [2/6] Corrected 80-Cell Break-Even Heatmap (2K..1M x 0%..99.5%)
        # ------------------------------------------------------------------
        print("\n[2/6] 80-Cell Break-Even Heatmap (strict matching, fixed Optical Pool admission)...", flush=True)
        t0 = time.perf_counter()
        contexts = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1000000]
        hit_rates = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.995]
        hm_pols = ["no_cache", "rdma_routing", "optical_naive", "optical_reservation", "optikv"]
        cl_hm = _strict_cluster(workers=8, hbm_gib=12.0, dram_gib=24.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, links=8)
        hm_jobs = []
        for ctx in contexts:
            n_sess = 24 if ctx <= 131072 else (16 if ctx <= 262144 else (12 if ctx <= 524288 else 8))
            gap = 1.5 if ctx <= 262144 else (4.0 if ctx <= 524288 else 8.0)
            for hr in hit_rates:
                tr = make_agentic_closedloop(
                    name=f"hm-{ctx}-{int(round(hr*1000))}",
                    context_tokens=ctx,
                    hit_rate=hr,
                    n_sessions=n_sess,
                    turns_per_session=8,
                    gap_s=gap,
                    workers=8,
                )
                for pol in hm_pols:
                    hm_jobs.append({
                        "trace": tr,
                        "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_hm, "policies": [pol], "slo_ttft_s": 1.0, "seed": 0},
                        "meta": {"context_tokens": ctx, "requested_hit_rate": hr},
                    })
        hm_rows = run_batch(pool, hm_jobs)
        by_cell: dict[tuple[int, float], dict[str, dict]] = {}
        for r in hm_rows:
            by_cell.setdefault((r["context_tokens"], r["requested_hit_rate"]), {})[r["policy"]] = r
        cells = []
        for (ctx, hr), by_p in sorted(by_cell.items()):
            rdma, opt, naive, resv, nc = by_p["rdma_routing"], by_p["optikv"], by_p["optical_naive"], by_p["optical_reservation"], by_p["no_cache"]
            sp_rdma = rdma["p50_ttft_s"] / max(1e-9, opt["p50_ttft_s"])
            cells.append({
                "context_tokens": ctx,
                "requested_hit_rate": hr,
                "nocache_p50_s": nc["p50_ttft_s"],
                "rdma_p50_s": rdma["p50_ttft_s"],
                "naive_p50_s": naive["p50_ttft_s"],
                "reservation_p50_s": resv["p50_ttft_s"],
                "optikv_p50_s": opt["p50_ttft_s"],
                "nocache_p99_s": nc["p99_ttft_s"],
                "rdma_p99_s": rdma["p99_ttft_s"],
                "naive_p99_s": naive["p99_ttft_s"],
                "optikv_p99_s": opt["p99_ttft_s"],
                "speedup_vs_rdma_p50": sp_rdma,
                "speedup_vs_rdma_p99": rdma["p99_ttft_s"] / max(1e-9, opt["p99_ttft_s"]),
                "speedup_vs_naive_p50": naive["p50_ttft_s"] / max(1e-9, opt["p50_ttft_s"]),
                "speedup_vs_nocache_p50": nc["p50_ttft_s"] / max(1e-9, opt["p50_ttft_s"]),
                "optikv_hit_rate": opt["hit_rate"],
                "optikv_ocs_duty": opt["ocs_duty_cycle"],
                "optikv_aborts": opt["abort_count"],
                "optikv_prefetches": opt["prefetch_bursts"],
                "no_benefit_vs_rdma": sp_rdma <= 1.05,
            })
        write_csv(SIM / "exp_breakeven_heatmap.csv", cells, list(cells[0].keys()))
        min_sp = min(c["speedup_vs_rdma_p50"] for c in cells)
        max_sp = max(c["speedup_vs_rdma_p50"] for c in cells)
        print(f"[2/6] Heatmap done in {time.perf_counter()-t0:.1f}s | min_speedup={min_sp:.3f}x max_speedup={max_sp:.2f}x", flush=True)

        # ------------------------------------------------------------------
        # [3/6] Sensitivity (tau, BW, Multi-Arch x FP8), Ablation, Noise, Load Sweep
        # ------------------------------------------------------------------
        print("\n[3/6] Sensitivity, Multi-Arch/FP8, Ablation, Noise, and 1-32 req/s Load Sweep...", flush=True)
        t0 = time.perf_counter()
        sens_pols = ["optikv", "rdma_prefetch", "rdma_routing", "optical_reservation", "optical_naive"]

        # 3a: tau sweep (0..50 ms)
        j_tau = []
        for tau_us in [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 25000.0, 50000.0]:
            cl = _strict_cluster(workers=8, hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=tau_us, links=8)
            for p in sens_pols:
                j_tau.append({"trace_key": "bimodal_128k_probes", "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": [p], "slo_ttft_s": 0.25, "seed": 0}, "meta": {"reconfig_us": tau_us}})
        r_tau = run_batch(pool, j_tau)
        write_csv(SIM / "exp_sens_reconfig.csv", r_tau, ["reconfig_us"] + SUMMARY_COLUMNS)

        # 3b: bandwidth sweep (45..800 GB/s)
        j_bw = []
        for bw in [45.0, 50.0, 100.0, 200.0, 400.0, 800.0]:
            cl = _strict_cluster(workers=8, hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=bw, tau_us=25000.0, links=8)
            for p in sens_pols:
                j_bw.append({"trace_key": "bimodal_128k_probes", "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": [p], "slo_ttft_s": 0.25, "seed": 0}, "meta": {"bandwidth_gb_s": bw}})
        r_bw = run_batch(pool, j_bw)
        write_csv(SIM / "exp_sens_bandwidth.csv", r_bw, ["bandwidth_gb_s"] + SUMMARY_COLUMNS)

        # 3c: Multi-arch x FP8
        j_ma = []
        for mname in ["qwen3.5-9b", "qwen2.5-7b-instruct-1m", "deepseek-v3", "qwen3-8b", "llama-3.1-70b"]:
            for db in (2, 1):
                hbm_g = 4.0 if mname == "qwen3.5-9b" else (10.0 if mname == "qwen2.5-7b-instruct-1m" else (12.0 if mname == "deepseek-v3" else (24.0 if mname == "qwen3-8b" else 48.0)))
                dram_g = hbm_g * 1.6
                pool_g = max(512.0, hbm_g * 40.0)
                tr_m = make_agentic_closedloop(name=f"ma-{mname}-b{db}", model_name=mname, dtype_bytes=db, context_tokens=131072, hit_rate=255.0 / 256.0, n_sessions=24, turns_per_session=8, gap_s=1.5, bimodal_gaps=True, workers=8)
                cl = _strict_cluster(workers=8, hbm_gib=hbm_g, dram_gib=dram_g, pool_gib=pool_g, ocs_bw=200.0, tau_us=25000.0, links=8)
                for p in ("rdma_routing", "optical_naive", "optikv"):
                    j_ma.append({"trace": tr_m, "config": {"model": mname, "dtype_bytes": db, "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": [p], "slo_ttft_s": 0.5, "seed": 0}, "meta": {"model_name": mname, "dtype_bytes": db}})
        r_ma = run_batch(pool, j_ma)
        write_csv(SIM / "exp_sens_multiarch_fp8.csv", r_ma, ["model_name", "dtype_bytes"] + SUMMARY_COLUMNS)

        # 3d: 7-Step Cumulative Ablation
        abl_steps = [
            ("Optical-Naive", "optical_naive", None),
            ("+Latch", "optikv", {"enable_routing": False, "enable_recompute_fallback": False, "enable_placement": False, "enable_prefetch": False, "enable_eviction": False}),
            ("+Co-Route", "optikv", {"enable_routing": True, "enable_recompute_fallback": False, "enable_placement": False, "enable_prefetch": False, "enable_eviction": False}),
            ("+Abort", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": False, "enable_prefetch": False, "enable_eviction": False}),
            ("+TierSpill", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": True, "enable_prefetch": False, "enable_eviction": False}),
            ("+GapFetch", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": True, "enable_prefetch": True, "enable_eviction": False}),
            ("+PrefixGDSF", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": True, "enable_prefetch": True, "enable_eviction": True}),
        ]
        cl_abl = _strict_cluster(workers=8, hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, links=8)
        j_abl = []
        for idx, (lbl, p, flg) in enumerate(abl_steps):
            cfg = {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_abl, "policies": [p], "slo_ttft_s": 0.25, "seed": 0}
            if flg is not None:
                cfg["optikv"] = flg
            j_abl.append({"trace_key": "bimodal_128k_probes", "config": cfg, "meta": {"order": idx, "ablation_step": lbl}})
        r_abl = sorted(run_batch(pool, j_abl), key=lambda r: r["order"])
        write_csv(SIM / "exp_ablation.csv", r_abl, ["ablation_step"] + SUMMARY_COLUMNS)

        # 3e: Noise Robustness (3 seeds)
        j_ns = []
        for sig in [0.0, 0.05, 0.10, 0.20]:
            for sd in (0, 1, 2):
                j_ns.append({"trace_key": "bimodal_128k_probes", "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_abl, "policies": ["optikv"], "slo_ttft_s": 0.25, "seed": sd, "reuse_noise_std": sig}, "meta": {"sigma": sig, "seed_id": sd}})
        r_ns = run_batch(pool, j_ns)
        write_csv(SIM / "exp_noise_robustness.csv", r_ns, ["sigma", "seed_id"] + SUMMARY_COLUMNS)

        # 3f: Offered Load Sweep (1..32 req/s, 36 concurrent 128K sessions, 250ms SLO)
        j_ld = []
        cl_ld = _strict_cluster(workers=8, hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, links=8)
        for rps in [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]:
            tr_l = make_agentic_closedloop(name=f"load-128k-r{int(rps)}", context_tokens=131072, hit_rate=255.0 / 256.0, n_sessions=36, turns_per_session=6, workers=8, steady_rps=rps)
            for p in ALL_POLS:
                j_ld.append({"trace": tr_l, "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_ld, "policies": [p], "slo_ttft_s": 0.25, "seed": 0}, "meta": {"target_rps": rps}})
        r_ld = run_batch(pool, j_ld)
        write_csv(SIM / "exp_load_sweep.csv", r_ld, ["target_rps"] + SUMMARY_COLUMNS)
        print(f"[3/6] Sensitivity + Ablation + Noise + Load done in {time.perf_counter()-t0:.1f}s", flush=True)

        # ------------------------------------------------------------------
        # [4/6] Strict Physics Multi-Node Scale-Out (4-32 GPUs: BOTH 1:1 AND 4:1!) + W=16 Port Sweep
        # ------------------------------------------------------------------
        print("\n[4/6] Strict-Matching 4-32 GPU Weak Scaling (1:1 P=W vs 4:1 P=W/4) + W=16 Port Sweep...", flush=True)
        t0 = time.perf_counter()
        scale_pols = ["rdma_routing", "concur_rdma", "rdma_prefetch", "optical_naive", "optikv", "oracle_infinite_hbm"]
        j_sc = []
        for w in [4, 8, 16, 32]:
            tr_sc = make_agentic_closedloop(name=f"scale-128k-w{w}", context_tokens=131072, hit_rate=0.996, n_sessions=4 * w, turns_per_session=6, workers=w, steady_rps=1.5 * w)
            # 1:1 Non-oversubscribed (P = W): Theorem 1a regime
            cl_11 = _strict_cluster(workers=w, hbm_gib=10.0, dram_gib=16.0, pool_gib=64.0 * w, ocs_bw=200.0, tau_us=25000.0, links=w, couple_decode=True)
            for pol in scale_pols:
                j_sc.append({"trace": tr_sc, "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_11, "policies": [pol], "slo_ttft_s": 0.25, "seed": 0}, "meta": {"sweep_type": "weak_scale_1to1", "workers": w, "ocs_ports": w, "oversub_ratio": 1.0, "offered_rps": 1.5 * w}})
            # 4:1 Oversubscribed (P = W/4): Theorem 1b true physical matching regime
            p4 = max(1, w // 4)
            cl_41 = _strict_cluster(workers=w, hbm_gib=10.0, dram_gib=16.0, pool_gib=64.0 * w, ocs_bw=200.0, tau_us=25000.0, links=p4, couple_decode=True)
            j_sc.append({"trace": tr_sc, "policy_override": "optikv_4to1", "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_41, "policies": ["optikv"], "slo_ttft_s": 0.25, "seed": 0}, "meta": {"sweep_type": "weak_scale_4to1", "workers": w, "ocs_ports": p4, "oversub_ratio": w / p4, "offered_rps": 1.5 * w}})

        # W=16 heavy burst port sweep: P in {1, 2, 4, 8, 16} (16:1 down to 1:1!)
        tr_p16 = make_agentic_closedloop(name="port-128k-w16", context_tokens=131072, hit_rate=0.996, n_sessions=64, turns_per_session=6, workers=16, steady_rps=64.0)
        for p in [1, 2, 4, 8, 16]:
            cl_p = _strict_cluster(workers=16, hbm_gib=10.0, dram_gib=16.0, pool_gib=1024.0, ocs_bw=200.0, tau_us=25000.0, links=p, couple_decode=True)
            for pol in ("optikv", "rdma_prefetch", "rdma_routing"):
                j_sc.append({"trace": tr_p16, "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_p, "policies": [pol], "slo_ttft_s": 0.25, "seed": 0}, "meta": {"sweep_type": "port_ratio_w16", "workers": 16, "ocs_ports": p, "oversub_ratio": 16.0 / p, "offered_rps": 64.0}})

        r_sc = run_batch(pool, j_sc)
        extra_sc = ["sweep_type", "workers", "ocs_ports", "oversub_ratio", "offered_rps", "sim_wall_s", "ctrl_ms_per_req"]
        write_csv(SIM / "exp_v3_cluster_scale.csv", r_sc, extra_sc + SUMMARY_COLUMNS)
        print(f"[4/6] Scale-out + Port Sweep done in {time.perf_counter()-t0:.1f}s:", flush=True)
        for r in r_sc:
            if r["policy"] in ("optikv", "optikv_4to1", "rdma_routing", "rdma_prefetch", "oracle_infinite_hbm"):
                print(f"  {r['sweep_type']:16s} W={r['workers']:2d} P={r['ocs_ports']:2d} ({r['oversub_ratio']:4.1f}:1) | {r['policy']:19s} | P50={r['p50_ttft_s']*1e3:6.1f}ms | GP={r['goodput_rps']:5.2f} | Duty={r['ocs_duty_cycle']*100:5.1f}%", flush=True)

        # ------------------------------------------------------------------
        # [5/6] Strict Physics Fidelity & WDM Striping (1x, 2x, 4x)
        # ------------------------------------------------------------------
        print("\n[5/6] Strict Fidelity (1:1 vs 4:1 Matching + Decode + WDM 1x/2x/4x)...", flush=True)
        t0 = time.perf_counter()
        j_fid = []
        fid_wls = [
            ("lmcache-120", "lm_120", 4.0, 32.0, 512.0, 0.25),
            ("freeinference-50", "fi_50", 16.0, 64.0, 1024.0, 1.0),
            ("agentic-128k-bimodal", "bimodal_128k_probes", 10.0, 16.0, 512.0, 0.25),
        ]
        fid_regimes = [
            ("Match-1:1 (Base)", 8, False, 1),
            ("Match-4:1 (Oversub)", 2, False, 1),
            ("Match-1:1 + Decode", 8, True, 1),
            ("Match-4:1 + Decode", 2, True, 1),
            ("Strict 1:1 + WDM 2x", 8, True, 2),
            ("Strict 1:1 + WDM 4x", 8, True, 4),
        ]
        for wlbl, tkey, hbm_g, dram_g, pool_g, slo in fid_wls:
            for rlbl, links, dec, wdm in fid_regimes:
                cl = _strict_cluster(workers=8, hbm_gib=hbm_g, dram_gib=dram_g, pool_gib=pool_g, ocs_bw=200.0, tau_us=25000.0, links=links, wdm=wdm, couple_decode=dec)
                for pol in ("rdma_routing", "concur_rdma", "rdma_prefetch", "optical_naive", "optikv"):
                    j_fid.append({"trace_key": tkey, "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": [pol], "slo_ttft_s": slo, "seed": 0}, "meta": {"workload": wlbl, "regime": rlbl, "ocs_ports": links, "wdm_wavelengths": wdm, "couple_decode": dec}})
        r_fid = run_batch(pool, j_fid)
        write_csv(SIM / "exp_v3_fidelity.csv", r_fid, ["workload", "regime", "ocs_ports", "wdm_wavelengths", "couple_decode"] + SUMMARY_COLUMNS)
        print(f"[5/6] Fidelity done in {time.perf_counter()-t0:.1f}s", flush=True)

        # ------------------------------------------------------------------
        # [6/6] 5-Fold Disjoint Cross-Validation (All 767 LMCache + 250 FreeInference)
        # ------------------------------------------------------------------
        print("\n[6/6] 5-Fold Disjoint Session CV (767 LMCache + 250 FreeInference)...", flush=True)
        t0 = time.perf_counter()
        cv_pols = ["rdma_routing", "concur_rdma", "rdma_prefetch", "optical_naive", "optikv", "oracle_infinite_hbm"]
        j_cv = []
        for fold in range(5):
            cl_lm = _strict_cluster(workers=8, hbm_gib=4.0, dram_gib=32.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, links=8)
            cl_fi = _strict_cluster(workers=8, hbm_gib=16.0, dram_gib=64.0, pool_gib=1024.0, ocs_bw=200.0, tau_us=25000.0, links=8)
            for pol in cv_pols:
                j_cv.append({"trace_key": f"lm_fold_{fold}", "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_lm, "policies": [pol], "slo_ttft_s": 0.25, "seed": 0}, "meta": {"dataset": "lmcache_767", "fold": fold}})
                j_cv.append({"trace_key": f"fi_fold_{fold}", "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_fi, "policies": [pol], "slo_ttft_s": 1.0, "seed": 0}, "meta": {"dataset": "freeinference_250", "fold": fold}})
        r_cv = run_batch(pool, j_cv)
        write_csv(SIM / "exp_cv5_strict.csv", r_cv, ["dataset", "fold"] + SUMMARY_COLUMNS)
        print(f"[6/6] 5-Fold CV done in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"\n=== ALL 6 STRICT-MATCHING STAGES COMPLETED IN {time.perf_counter()-t_all:.1f}s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
