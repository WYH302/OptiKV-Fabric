#!/usr/bin/env python3
"""Final single-provenance experiment script for OptiKV-Fabric (ACM TACO / IEEE TPDS).

Runs with the final Prefix-Dependency-Aware GDSF + Pinned-Headroom Prefetch engine:
  [0/6] Updates `optikv` across all 6 real-trace CSVs (Mooncake 3090/SiPh/MEMS, LMCache 120/24K,
        FreeInference 25-sess) so 100% of results/sim/ share identical final code provenance.
  [1/6] 128K Closed-Loop Agentic Teaser (Fig 1a).
  [2/6] 8x10 Closed-Loop Break-Even Heatmap (2K..1M x 0%..99.5% reuse, Fig 1c & Fig 5).
  [3/6] Hardware Sensitivity (tau, BW, Links) + Multi-Arch x FP8 (Fig 6).
  [4/6] 7-Step Cumulative Ablation + Single-GPU Eviction (LRU/S3-FIFO/Prefix-GDSF/Belady) + Noise (Fig 7).
  [5/6] Steady-State Interactive Load Sweep (1..32 req/s, 250ms SLO, Fig 8 & Fig 6d).
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from optikv.metrics import SUMMARY_COLUMNS, write_csv
from optikv.modelspec import load_model
from optikv.prefill import build_prefill_model
from optikv.trace import BlockRef, Request, Trace
from run_full_suite import (
    MAX_WORKERS,
    SIM,
    _run_one_job,
    load_freeinference_long,
    load_lmcache_tokenized,
    load_mooncake_full,
    make_cluster,
    pace_trace,
)


def make_agentic_closedloop(
    *,
    name: str,
    model_name: str = "qwen2.5-7b-instruct-1m",
    hardware_profile: str = "h100_peak_spec",
    dtype_bytes: int = 2,
    context_tokens: int = 131072,
    hit_rate: float = 0.996,
    n_sessions: int = 24,
    turns_per_session: int = 8,
    gap_s: float = 1.5,
    bimodal_gaps: bool = False,
    workers: int = 8,
    steady_rps: float | None = None,
    include_short_probes: bool = False,
) -> Trace:
    """Closed-loop multi-turn agentic trace with realistic post-completion turn gaps.

    Enforces n_unique >= 1 whenever hit_rate < 1.0 (avoiding 0-novel-token rounding artifacts
    on short 2K-8K prompts) and paces cold Turn-0 launches below cluster saturation.
    """
    spec = load_model(model_name)
    prefill = build_prefill_model(
        spec,
        dtype_bytes=dtype_bytes,
        calibration_path=ROOT / "configs/hardware/calibration_3090.yaml",
        hardware_profile=hardware_profile,
    )
    block_tokens = 512 if context_tokens >= 512 else context_tokens
    n_blocks = max(1, context_tokens // block_tokens)
    if hit_rate <= 0.0:
        n_shared = 0
    elif hit_rate >= 1.0:
        n_shared = n_blocks
    else:
        max_shared = max(0, n_blocks - 1) if n_blocks >= 2 else n_blocks
        n_shared = min(max_shared, max(0, int(round(hit_rate * n_blocks))))
    n_unique = n_blocks - n_shared

    cold_s = prefill.cold_parts(n_blocks * block_tokens)[0]
    warm_s = prefill.request_times(n_shared * block_tokens, n_unique * block_tokens)[0] if n_shared > 0 else cold_s
    cold_launch_step = max(0.25, (cold_s / max(1, workers)) * 1.40)

    reqs: list[Request] = []
    if steady_rps is not None:
        period = max(warm_s + 0.15, n_sessions / max(0.1, steady_rps))
        eff_gap = max(0.15, period - warm_s)
        warm_start = n_sessions * cold_launch_step + cold_s + 1.0
        for s in range(n_sessions):
            t0 = s * cold_launch_step
            shared_blocks = tuple(BlockRef(f"{name}-s{s}-p{i}", block_tokens) for i in range(n_shared))
            uniq0 = tuple(BlockRef(f"{name}-s{s}-t0-u{j}", block_tokens) for j in range(n_unique))
            reqs.append(
                Request(
                    request_id=f"{name}-s{s}-t0",
                    arrival_s=t0,
                    session_id=f"{name}-sess-{s}",
                    turn_index=0,
                    pre_gap_s=0.0,
                    blocks=shared_blocks + uniq0,
                    output_tokens=64,
                )
            )
            s_warm0 = warm_start + (s / max(1, n_sessions)) * period
            for t in range(1, turns_per_session):
                if n_shared == 0:
                    blks = tuple(BlockRef(f"{name}-s{s}-t{t}-u{j}", block_tokens) for j in range(n_blocks))
                else:
                    uniq = tuple(BlockRef(f"{name}-s{s}-t{t}-u{j}", block_tokens) for j in range(n_unique))
                    blks = shared_blocks + uniq
                reqs.append(
                    Request(
                        request_id=f"{name}-s{s}-t{t}",
                        arrival_s=s_warm0 + (t - 1) * period,
                        session_id=f"{name}-sess-{s}",
                        turn_index=t,
                        pre_gap_s=eff_gap,
                        blocks=blks,
                        output_tokens=64,
                    )
                )
    else:
        for s in range(n_sessions):
            t0 = s * cold_launch_step
            shared_blocks = tuple(BlockRef(f"{name}-s{s}-p{i}", block_tokens) for i in range(n_shared))
            cur_arr = t0
            for t in range(turns_per_session):
                if t == 0:
                    arr = t0
                    pg = 0.0
                    uniq = tuple(BlockRef(f"{name}-s{s}-t0-u{j}", block_tokens) for j in range(n_unique))
                    blks = shared_blocks + uniq
                    cur_arr = t0 + cold_s
                else:
                    # Bimodal gap models LMCache/FreeInference mix of fast automated tool calls (0.04s)
                    # and slower user/code-exec tool pauses (gap_s = 1.5s).
                    pg = (0.04 if (bimodal_gaps and (t % 2 == 0)) else gap_s)
                    arr = cur_arr + pg
                    cur_arr = arr + warm_s
                    if n_shared == 0:
                        blks = tuple(BlockRef(f"{name}-s{s}-t{t}-u{j}", block_tokens) for j in range(n_blocks))
                    else:
                        uniq = tuple(BlockRef(f"{name}-s{s}-t{t}-u{j}", block_tokens) for j in range(n_unique))
                        blks = shared_blocks + uniq
                reqs.append(
                    Request(
                        request_id=f"{name}-s{s}-t{t}",
                        arrival_s=arr,
                        session_id=f"{name}-sess-{s}",
                        turn_index=t,
                        pre_gap_s=pg,
                        blocks=blks,
                        output_tokens=64,
                    )
                )

    if include_short_probes:
        base_t = max(1.0, cold_s + 0.5)
        for k in range(n_sessions):
            anchor_blk = (BlockRef(f"{name}-short-p{k}", 512),)
            t_probe = base_t + k * max(0.4, cold_launch_step)
            reqs.append(
                Request(
                    request_id=f"{name}-short-w{k}",
                    arrival_s=t_probe,
                    session_id=f"{name}-short-s{k}",
                    turn_index=0,
                    pre_gap_s=0.0,
                    blocks=anchor_blk + (BlockRef(f"{name}-short-u0-{k}", 512),),
                    output_tokens=16,
                    assigned_worker=k % workers,
                )
            )
            reqs.append(
                Request(
                    request_id=f"{name}-short-p{k}",
                    arrival_s=t_probe + 0.12,
                    session_id=f"{name}-short-s{k}",
                    turn_index=1,
                    pre_gap_s=0.0,
                    blocks=anchor_blk + (BlockRef(f"{name}-short-u1-{k}", 512),),
                    output_tokens=16,
                    assigned_worker=(k + 1) % workers,
                )
            )

    reqs.sort(key=lambda r: (r.arrival_s, r.request_id))
    return Trace(name=name, kind="synthetic", block_tokens=block_tokens, requests=reqs)


_WORKER_TRACE_CACHE: dict[str, Trace] = {}


def _load_real_trace_in_worker(key: str) -> Trace:
    if key in _WORKER_TRACE_CACHE:
        return _WORKER_TRACE_CACHE[key]
    if key == "mc_full":
        tr = load_mooncake_full()
    elif key == "mc_3090":
        mc = load_mooncake_full()
        tr = pace_trace(Trace("mooncake-2k-3090", mc.kind, mc.block_tokens, mc.requests[:2000]), target_rps=0.8, name="mooncake-2k-3090-paced")
    elif key == "lm_120":
        tr = load_lmcache_tokenized(max_sessions=120, session_launch_gap_s=0.5)
    elif key == "lm_full":
        tr = load_lmcache_tokenized(max_sessions=None, session_launch_gap_s=0.8)
    elif key == "fi_25":
        tr = load_freeinference_long(max_sessions=25, session_launch_gap_s=1.0)
    else:
        raise ValueError(f"unknown trace_key={key}")
    _WORKER_TRACE_CACHE[key] = tr
    return tr


def _run_real_trace_job(job: dict) -> dict:
    tr = _load_real_trace_in_worker(job["trace_key"])
    return _run_one_job({"config": job["config"], "trace": tr, "meta": job.get("meta", {})})


def _patch_policy_row(csv_path: Path, new_row: dict, *, after_policy: str | None = None) -> None:
    """Replace or cleanly insert a policy row in an existing main-suite CSV."""
    df = pd.read_csv(csv_path)
    rows = df.to_dict(orient="records")
    pol = new_row["policy"]
    clean = {c: new_row.get(c) for c in SUMMARY_COLUMNS}
    for idx, r in enumerate(rows):
        if r["policy"] == pol:
            rows[idx] = {c: new_row.get(c, r.get(c)) for c in SUMMARY_COLUMNS}
            write_csv(csv_path, rows, SUMMARY_COLUMNS)
            return
    if after_policy is not None:
        for idx, r in enumerate(rows):
            if r["policy"] == after_policy:
                rows.insert(idx + 1, clean)
                write_csv(csv_path, rows, SUMMARY_COLUMNS)
                return
    rows.append(clean)
    write_csv(csv_path, rows, SUMMARY_COLUMNS)


def main() -> int:
    t_all = time.perf_counter()
    workers = min(8, MAX_WORKERS)
    print(f"=== FINAL Single-Provenance Suite (Pool-Coalesced OptiKV + RDMA+Prefetch SOTA, workers={workers}) ===", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # --- 0. Update `optikv` and add `rdma_prefetch` (2025-2026 Electrical SOTA) across all 6 real-trace CSVs ---
        print("[0/6] Refreshing OptiKV-Fabric & RDMA+Prefetch across all 6 real-trace CSVs...", flush=True)
        t0 = time.perf_counter()

        cl_mc_siph = make_cluster(workers=8, hbm_gib=4.0, dram_gib=64.0, ssd_gib=256.0, pool_gib=256.0, rdma_bw=25.0, ocs_bw=200.0, tau_us=10.0, links=2)
        cl_mc_mems = make_cluster(workers=8, hbm_gib=4.0, dram_gib=64.0, ssd_gib=256.0, pool_gib=256.0, rdma_bw=25.0, ocs_bw=200.0, tau_us=25000.0, links=2)
        cl_mc_3090 = make_cluster(workers=8, hbm_gib=4.0, dram_gib=64.0, ssd_gib=256.0, pool_gib=256.0, dram_bw=26.28, rdma_bw=25.0, ocs_bw=45.0, tau_us=25000.0, links=2)
        cl_lm = make_cluster(workers=8, hbm_gib=4.0, dram_gib=64.0, ssd_gib=256.0, pool_gib=256.0, rdma_bw=25.0, ocs_bw=200.0, tau_us=25000.0, links=2)
        cl_fi = make_cluster(workers=8, hbm_gib=8.0, dram_gib=64.0, ssd_gib=256.0, pool_gib=256.0, rdma_bw=25.0, ocs_bw=200.0, tau_us=25000.0, links=2)

        real_specs = [
            ("exp_main_mooncake_h100_siph.csv", "h100_peak_spec", cl_mc_siph, 0.5, "mc_full"),
            ("exp_main_mooncake_h100_mems.csv", "h100_peak_spec", cl_mc_mems, 0.5, "mc_full"),
            ("exp_main_mooncake_3090_calibrated.csv", "rtx3090", cl_mc_3090, 2.0, "mc_3090"),
            ("exp_main_lmcache_h100_mems.csv", "h100_peak_spec", cl_lm, 0.25, "lm_120"),
            ("exp_main_lmcache_full24k_h100.csv", "h100_peak_spec", cl_lm, 0.25, "lm_full"),
            ("exp_main_freeinference_h100_mems.csv", "h100_peak_spec", cl_fi, 1.0, "fi_25"),
        ]
        f_real = {}
        for fname, hw, cl, slo, tkey in real_specs:
            for pol in ("rdma_prefetch", "optikv"):
                cfg = {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": hw, "cluster": cl, "policies": [pol], "slo_ttft_s": slo, "seed": 0}
                f_real[pool.submit(_run_real_trace_job, {"config": cfg, "trace_key": tkey})] = (fname, pol)
        for fut in as_completed(f_real):
            fname, pol = f_real[fut]
            row = fut.result()["rows"][0]
            _patch_policy_row(SIM / fname, row, after_policy="rdma_routing" if pol == "rdma_prefetch" else None)
            print(f"  {fname:40s} [{pol:13s}]: p50={row['p50_ttft_s']*1e3:6.1f}ms p99={row['p99_ttft_s']*1e3:7.1f}ms hbm={row['hbm_hit_rate']*100:5.1f}% pf={row['prefetch_bursts']:4d} duty={row['ocs_duty_cycle']*100:4.1f}%", flush=True)
        print(f"  All 6 real-trace OptiKV & RDMA+Prefetch rows refreshed in {time.perf_counter()-t0:.1f}s", flush=True)

        # --- 1. Fig 1a Teaser (128K closed-loop multi-turn agentic, 512-tok delta = 99.6% reuse) ---
        print("\n[1/6] Running 128K Closed-Loop Agentic Teaser...", flush=True)
        tr_128k = make_agentic_closedloop(
            name="agentic-128k-512delta",
            context_tokens=131072,
            hit_rate=255.0 / 256.0,
            n_sessions=24,
            turns_per_session=8,
            gap_s=1.5,
            bimodal_gaps=True,
            workers=8,
        )
        cl_128k = make_cluster(workers=8, hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0, links=2)
        pols_t = ["no_cache", "local_hbm", "local_dram", "rdma_routing", "rdma_prefetch", "optical_naive", "optical_reservation", "optikv", "oracle_infinite_hbm"]
        f_t = [
            pool.submit(_run_one_job, {"config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_128k, "policies": [p], "slo_ttft_s": 0.25, "seed": 0}, "trace": tr_128k, "meta": {"order": i}})
            for i, p in enumerate(pols_t)
        ]
        rows_t = sorted([r for f in f_t for r in f.result()["rows"]], key=lambda r: r["order"])
        write_csv(SIM / "fig1_teaser_128k.csv", rows_t, SUMMARY_COLUMNS)
        for r in rows_t:
            print(f"  128K {r['policy']:20s}: p50={r['p50_ttft_s']*1e3:7.1f}ms p99={r['p99_ttft_s']*1e3:7.1f}ms xfer={r['mean_transfer_s']*1e3:6.1f}ms hit={r['hit_rate']*100:5.1f}% pf={r['prefetch_bursts']:3d} duty={r['ocs_duty_cycle']*100:4.1f}%", flush=True)

        # --- 2. 8x10 Break-Even Heatmap (2K..1M x 0, 25, 50, 75, 90, 95, 98, 99.5%) ---
        print("\n[2/6] Running 8x10 Closed-Loop Break-Even Heatmap (2K..1M x 0%..99.5%)...", flush=True)
        t0 = time.perf_counter()
        contexts = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1000000]
        hit_rates = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.995]
        hm_pols = ["no_cache", "rdma_routing", "optical_naive", "optical_reservation", "optikv"]
        cl_hm = make_cluster(workers=8, hbm_gib=12.0, dram_gib=24.0, pool_gib=512.0, rdma_bw=25.0, ocs_bw=200.0, tau_us=25000.0, links=2)
        f_hm = {}
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
                f_hm[
                    pool.submit(
                        _run_one_job,
                        {
                            "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_hm, "policies": hm_pols, "slo_ttft_s": 1.0, "seed": 0},
                            "trace": tr,
                            "meta": {"context_tokens": ctx, "requested_hit_rate": hr},
                        },
                    )
                ] = (ctx, hr)
        cells = []
        for fut in as_completed(f_hm):
            ctx, hr = f_hm[fut]
            by_p = {r["policy"]: r for r in fut.result()["rows"]}
            rdma, opt, naive, resv, nc = by_p["rdma_routing"], by_p["optikv"], by_p["optical_naive"], by_p["optical_reservation"], by_p["no_cache"]
            sp_rdma = rdma["p50_ttft_s"] / max(1e-9, opt["p50_ttft_s"])
            cells.append(
                {
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
                }
            )
        cells.sort(key=lambda r: (r["context_tokens"], r["requested_hit_rate"]))
        write_csv(SIM / "exp_breakeven_heatmap.csv", cells, list(cells[0].keys()))
        nb = sum(1 for c in cells if c["no_benefit_vs_rdma"])
        best = max(cells, key=lambda c: c["speedup_vs_rdma_p50"])
        print(f"  Heatmap 80 cells done in {time.perf_counter()-t0:.1f}s | no-benefit: {nb}/80 | peak vs RDMA: {best['speedup_vs_rdma_p50']:.2f}x at {best['context_tokens']} tok, {best['requested_hit_rate']*100:.1f}%", flush=True)

        # --- 3. Hardware Sensitivity (tau, BW, Links) + Multi-Arch / FP8 ---
        print("\n[3/6] Running Reconfig, BW, Links, and Multi-Arch/FP8 sweeps...", flush=True)
        t0 = time.perf_counter()
        sens_tr = make_agentic_closedloop(
            name="sens-128k-bimodal",
            context_tokens=131072,
            hit_rate=255.0 / 256.0,
            n_sessions=24,
            turns_per_session=8,
            gap_s=1.5,
            bimodal_gaps=True,
            workers=8,
            include_short_probes=True,
        )
        s_pols = ["rdma_routing", "rdma_prefetch", "optical_naive", "optical_reservation", "optikv"]
        f_tau = [
            pool.submit(_run_one_job, {"config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": make_cluster(hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=tu, rdma_bw=25.0, links=2), "policies": s_pols, "slo_ttft_s": 0.25, "seed": 0}, "trace": sens_tr})
            for tu in (0, 1, 10, 100, 1000, 10000, 25000, 50000)
        ]
        f_bw = [
            pool.submit(_run_one_job, {"config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": make_cluster(hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=bw, tau_us=25000.0, rdma_bw=25.0, links=2), "policies": s_pols, "slo_ttft_s": 0.25, "seed": 0}, "trace": sens_tr})
            for bw in (45, 50, 100, 200, 400, 800)
        ]
        f_lk = [
            pool.submit(_run_one_job, {"config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": make_cluster(hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0, links=lk), "policies": s_pols, "slo_ttft_s": 0.25, "seed": 0}, "trace": sens_tr})
            for lk in (1, 2, 4, 8)
        ]
        f_arch = []
        for mname in ("qwen2.5-7b-instruct-1m", "qwen3-8b", "llama-3.1-70b", "deepseek-v3", "qwen3.5-9b"):
            spec = load_model(mname)
            for dtype_b, prec in ((2, "BF16"), (1, "FP8")):
                tr_arch = make_agentic_closedloop(
                    name=f"arch-{mname}-{prec}",
                    model_name=mname,
                    dtype_bytes=dtype_b,
                    context_tokens=131072,
                    hit_rate=255.0 / 256.0,
                    n_sessions=20,
                    turns_per_session=8,
                    gap_s=1.5 if mname not in ("llama-3.1-70b", "deepseek-v3") else 4.0,
                    workers=8,
                )
                sess_bf16_gib = (131072 * spec.kv_bytes_per_token(2)) / (1024**3)
                hbm_gib = max(5.0, round(sess_bf16_gib * 1.4, 1))
                dram_gib = max(10.0, round(sess_bf16_gib * 2.2, 1))
                pool_gib = max(512.0, round(sess_bf16_gib * 30.0, 1))
                cl_a = make_cluster(hbm_gib=hbm_gib, dram_gib=dram_gib, pool_gib=pool_gib, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0, links=2)
                f_arch.append(
                    pool.submit(
                        _run_one_job,
                        {
                            "config": {"model": mname, "dtype_bytes": dtype_b, "hardware_profile": "h100_peak_spec", "cluster": cl_a, "policies": ["no_cache", "rdma_routing", "optical_naive", "optikv"], "slo_ttft_s": 1.0, "seed": 0},
                            "trace": tr_arch,
                            "meta": {"precision": prec, "kv_arch": spec.kv_arch},
                        },
                    )
                )
        write_csv(SIM / "exp_sensitivity_reconfig.csv", [r for f in f_tau for r in f.result()["rows"]], SUMMARY_COLUMNS)
        write_csv(SIM / "exp_sensitivity_bandwidth.csv", [r for f in f_bw for r in f.result()["rows"]], SUMMARY_COLUMNS)
        write_csv(SIM / "exp_sensitivity_links.csv", [r for f in f_lk for r in f.result()["rows"]], SUMMARY_COLUMNS)
        arch_rows = [r for f in f_arch for r in f.result()["rows"]]
        write_csv(SIM / "exp_sensitivity_arch_fp8.csv", arch_rows, ["precision", "kv_arch"] + SUMMARY_COLUMNS)
        print(f"  Sensitivity & Multi-Arch/FP8 done in {time.perf_counter()-t0:.1f}s", flush=True)

        # --- 4. Stepwise Ablation + Single-GPU Apples-to-Apples Eviction + Noise Robustness ---
        print("\n[4/6] Running Stepwise Ablation, Single-GPU Eviction, and Noise Robustness...", flush=True)
        t0 = time.perf_counter()
        cl_abl = make_cluster(hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0, links=2)
        step_specs = [
            ("Optical-Naive", "optical_naive", None),
            ("+StandingCircuit", "optical_reservation", None),
            ("+OpticalRouting", "optikv", {"enable_routing": True, "enable_recompute_fallback": False, "enable_placement": False, "enable_prefetch": False, "enable_eviction": False}),
            ("+RecomputeAbort", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": False, "enable_prefetch": False, "enable_eviction": False}),
            ("+TierPlacement", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": True, "enable_prefetch": False, "enable_eviction": False}),
            ("+WindowedPrefetch", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": True, "enable_prefetch": True, "enable_eviction": False}),
            ("+PrefixGDSF (Full)", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": True, "enable_prefetch": True, "enable_eviction": True}),
        ]
        f_abl = []
        for idx, (label, pol, flags) in enumerate(step_specs):
            cfg = {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_abl, "policies": [pol], "slo_ttft_s": 0.25, "seed": 0}
            if flags is not None:
                cfg["optikv"] = flags
            f_abl.append(pool.submit(_run_one_job, {"config": cfg, "trace": sens_tr, "meta": {"step_idx": idx, "ablation_step": label}}))

        cl_ev = make_cluster(workers=1, hbm_gib=2.0, dram_gib=0.0, ssd_gib=0.0, pool_gib=0.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0)
        ev_specs = [
            ("LRU (Local-HBM)", "local_hbm"),
            ("S3-FIFO", "eviction_s3fifo"),
            ("OptiKV-GDSF (HBM)", "eviction_gdsf"),
            ("Belady (Optimal)", "eviction_belady"),
        ]
        f_ev = [
            pool.submit(_run_real_trace_job, {"config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_ev, "policies": [pol], "slo_ttft_s": 0.25, "seed": 0}, "trace_key": "lm_120", "meta": {"ev_idx": idx, "eviction_label": label}})
            for idx, (label, pol) in enumerate(ev_specs)
        ]

        f_ns = [
            pool.submit(_run_one_job, {"config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_abl, "policies": ["optikv"], "reuse_noise_std": sig, "slo_ttft_s": 0.25, "seed": sd}, "trace": sens_tr, "meta": {"sigma": sig}})
            for sig in (0.0, 0.05, 0.10, 0.20)
            for sd in (0, 1, 2)
        ]
        abl_rows = sorted([r for f in f_abl for r in f.result()["rows"]], key=lambda r: r["step_idx"])
        ev_rows = sorted([r for f in f_ev for r in f.result()["rows"]], key=lambda r: r["ev_idx"])
        ns_rows = [r for f in f_ns for r in f.result()["rows"]]
        write_csv(SIM / "exp_ablation.csv", abl_rows, ["step_idx", "ablation_step"] + SUMMARY_COLUMNS)
        write_csv(SIM / "exp_eviction_comparison.csv", ev_rows, ["ev_idx", "eviction_label"] + SUMMARY_COLUMNS)
        write_csv(SIM / "exp_noise_robustness.csv", ns_rows, ["sigma"] + SUMMARY_COLUMNS)
        for r in abl_rows:
            print(f"  [Abl] {r['ablation_step']:24s}: p50={r['p50_ttft_s']*1e3:7.1f}ms p95={r['p95_ttft_s']*1e3:7.1f}ms xfer={r['mean_transfer_s']*1e3:5.1f}ms abrt={r['abort_count']:3d} pf={r['prefetch_bursts']:3d} duty={r['ocs_duty_cycle']*100:4.1f}%", flush=True)
        for r in ev_rows:
            print(f"  [Ev ] {r['eviction_label']:24s}: hit={r['hit_rate']*100:5.2f}%", flush=True)
        print(f"  Ablation/Eviction/Noise done in {time.perf_counter()-t0:.1f}s", flush=True)

        # --- 5. Steady-State Interactive Load Sweep (1 .. 32 req/s, 128K 512-tok delta, 250ms SLO) ---
        print("\n[5/6] Running Steady-State Load Sweep (1 .. 32 req/s, SLO=250ms)...", flush=True)
        t0 = time.perf_counter()
        rates = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]
        l_pols = ["no_cache", "local_hbm", "local_dram", "rdma_routing", "rdma_prefetch", "optical_naive", "optical_reservation", "optikv", "oracle_infinite_hbm"]
        f_ld = []
        for rps in rates:
            tr_l = make_agentic_closedloop(
                name=f"load-{rps:g}rps",
                context_tokens=131072,
                hit_rate=255.0 / 256.0,
                n_sessions=36,
                turns_per_session=10,
                workers=8,
                steady_rps=rps,
            )
            for p in l_pols:
                f_ld.append(
                    pool.submit(
                        _run_one_job,
                        {
                            "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_128k, "policies": [p], "slo_ttft_s": 0.25, "seed": 0},
                            "trace": tr_l,
                            "meta": {"target_rps": rps},
                        },
                    )
                )
        ld_rows = sorted([r for f in f_ld for r in f.result()["rows"]], key=lambda r: (r["target_rps"], l_pols.index(r["policy"])))
        write_csv(SIM / "exp_load_sweep.csv", ld_rows, ["target_rps"] + SUMMARY_COLUMNS)
        print(f"  Load Sweep done in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"\n=== ALL FINAL SUITES DONE IN {time.perf_counter()-t_all:.1f}s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
