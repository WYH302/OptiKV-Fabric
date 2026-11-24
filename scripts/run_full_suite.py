#!/usr/bin/env python3
"""Master parallel experiment suite for OptiKV-Fabric (ACM TACO / IEEE TPDS edition).

Runs all real-trace, dual-regime (RTX 3090 calibrated + H100 peak-spec scaled),
break-even heatmap, optical hardware sensitivity, multi-architecture/FP8, stepwise
ablation, modern eviction, prediction noise robustness, and load-sweep experiments
using multi-core parallelism. Writes deterministic, provenance-tagged CSV tables
into results/sim/ and results/tables/. Zero fabricated numbers.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optikv.metrics import SUMMARY_COLUMNS, percentile, write_csv
from optikv.modelspec import load_catalog, load_model
from optikv.prefill import build_prefill_model, fit_quadratic
from optikv.simulator import run_simulation
from optikv.trace import (
    FREEINFERENCE_BLOCK_TOKENS,
    MOONCAKE_BLOCK_TOKENS,
    BlockRef,
    Request,
    Trace,
    load_trace,
)

RESULTS = ROOT / "results"
SIM = RESULTS / "sim"
TABLES = RESULTS / "tables"
MAX_WORKERS = min(12, max(2, (os.cpu_count() or 4) - 4))

MAIN_POLICIES = [
    "no_cache",
    "local_hbm",
    "local_dram",
    "local_ssd",
    "rdma_routing",
    "rdma_prefetch",
    "optical_naive",
    "optical_reservation",
    "optikv",
    "oracle_infinite_hbm",
]


def make_cluster(
    *,
    workers: int = 8,
    hbm_gib: float = 8.0,
    dram_gib: float = 64.0,
    ssd_gib: float = 256.0,
    pool_gib: float = 256.0,
    dram_bw: float = 50.0,
    ssd_bw: float = 3.0,
    rdma_bw: float = 25.0,
    rdma_lat_us: float = 5.0,
    ocs_bw: float = 200.0,
    ocs_lat_us: float = 2.0,
    tau_us: float = 25000.0,
    links: int = 2,
    alpha: float = 0.7,
) -> dict:
    return {
        "num_workers": workers,
        "hbm_bytes_per_worker": int(hbm_gib * 1024**3),
        "dram_bytes_per_worker": int(dram_gib * 1024**3),
        "ssd_bytes_per_worker": int(ssd_gib * 1024**3),
        "rdma_pool_bytes": int(pool_gib * 1024**3),
        "dram_bandwidth_gb_s": dram_bw,
        "ssd_bandwidth_gb_s": ssd_bw,
        "rdma_bandwidth_gb_s": rdma_bw,
        "rdma_latency_us": rdma_lat_us,
        "alpha_overlap": alpha,
        "ocs": {
            "concurrent_links": links,
            "bandwidth_gb_s": ocs_bw,
            "latency_us": ocs_lat_us,
            "reconfig_us": tau_us,
        },
    }


def _run_one_job(job: dict) -> dict:
    """Worker entrypoint for ProcessPoolExecutor."""
    cfg = job["config"]
    trace = job["trace"]
    meta = job.get("meta", {})
    res = run_simulation(cfg, trace=trace)
    out_rows = []
    for row in res.summary_rows:
        merged = dict(row)
        merged.update(meta)
        out_rows.append(merged)
    return {"rows": out_rows, "meta": meta}


# ---------------------------------------------------------------------------
# Trace builders (100% deterministic, real datasets + controlled synthetic)
# ---------------------------------------------------------------------------

def load_mooncake_full() -> Trace:
    return load_trace(
        ROOT / "datasets/mooncake-traces/mooncake/train-00000-of-00001.parquet",
        kind="mooncake",
        name="mooncake-full",
    )


def pace_trace(trace: Trace, target_rps: float, name: str) -> Trace:
    """Rescale inter-arrival timestamps linearly to target arrival rate (req/s)."""
    reqs = trace.requests
    if len(reqs) < 2:
        return trace
    t0 = reqs[0].arrival_s
    span = max(1e-6, reqs[-1].arrival_s - t0)
    target_span = len(reqs) / max(1e-6, target_rps)
    scale = target_span / span
    scaled = [
        Request(
            request_id=r.request_id,
            arrival_s=(r.arrival_s - t0) * scale,
            session_id=r.session_id,
            turn_index=r.turn_index,
            pre_gap_s=r.pre_gap_s * scale,
            blocks=r.blocks,
            output_tokens=r.output_tokens,
            assigned_worker=r.assigned_worker,
        )
        for r in reqs
    ]
    return Trace(name=name, kind=trace.kind, block_tokens=trace.block_tokens, requests=scaled)


def load_lmcache_tokenized(max_sessions: int | None = None, session_launch_gap_s: float = 0.5) -> Trace:
    path = ROOT / "datasets/lmcache-agentic-traces/tokenized_512.parquet"
    tbl = pq.read_table(path)
    s_col = tbl.column("session_id").to_pylist()
    gaps = tbl.column("pre_gap").to_pylist()
    outs = tbl.column("output_length").to_pylist()
    bids = tbl.column("block_ids").to_pylist()
    blens = tbl.column("block_lens").to_pylist()

    uniq_s = list(dict.fromkeys(s_col))
    if max_sessions is not None:
        uniq_s = uniq_s[:max_sessions]
    pick_s = set(uniq_s)
    sess_t0 = {sid: i * session_launch_gap_s for i, sid in enumerate(uniq_s)}
    last_arr: dict[str, float] = {}
    turns: dict[str, int] = {}
    reqs: list[Request] = []
    for sid, g, out, bl, ll in zip(s_col, gaps, outs, bids, blens):
        if sid not in pick_s or not bl:
            continue
        t = turns.get(sid, 0)
        gap = 0.0 if t == 0 else max(0.0, float(g or 0.0))
        arr = sess_t0[sid] if t == 0 else (last_arr[sid] + max(0.20, gap))
        last_arr[sid] = arr
        turns[sid] = t + 1
        blocks = tuple(BlockRef(b, int(l)) for b, l in zip(bl, ll))
        reqs.append(Request(f"{sid}-{t}", arr, sid, t, gap, blocks, output_tokens=int(out or 1)))
    reqs.sort(key=lambda r: (r.arrival_s, r.request_id))
    tag = f"lmcache-all{len(uniq_s)}sess" if max_sessions is None else f"lmcache-{len(uniq_s)}sess"
    return Trace(name=tag, kind="lmcache", block_tokens=MOONCAKE_BLOCK_TOKENS, requests=reqs)


def load_freeinference_long(max_sessions: int = 25, session_launch_gap_s: float = 1.0) -> Trace:
    path = ROOT / "datasets/freeinference-agentic-trace/long_sessions_40.parquet"
    tbl = pq.read_table(path)
    s_col = tbl.column("session_id").to_pylist()
    steps = tbl.column("step").to_pylist()
    starts = tbl.column("start_ms").to_pylist()
    ends = tbl.column("end_ms").to_pylist()
    comps = tbl.column("completion_tokens").to_pylist()
    bids = tbl.column("block_ids").to_pylist()

    uniq_sess = list(dict.fromkeys(s_col))[:max_sessions]
    pick_sess = set(uniq_sess)
    sess_t0 = {sid: idx * session_launch_gap_s for idx, sid in enumerate(uniq_sess)}
    last_end: dict[str, float] = {}
    last_arr: dict[str, float] = {}
    reqs: list[Request] = []
    for sid, step, st, en, comp, b_list in zip(s_col, steps, starts, ends, comps, bids):
        if sid not in pick_sess or not b_list:
            continue
        blocks = tuple(BlockRef(str(b), FREEINFERENCE_BLOCK_TOKENS) for b in b_list)
        if sum(b.num_tokens for b in blocks) > 262144:
            continue
        st_f = float(st or 0.0)
        prev = last_end.get(sid)
        gap = 0.0 if prev is None else max(0.0, min(20.0, (st_f - prev) / 1000.0))
        arr = sess_t0[sid] if sid not in last_arr else (last_arr[sid] + max(0.25, gap))
        last_arr[sid] = arr
        last_end[sid] = float(en) if en is not None else (st_f + 1000.0)
        reqs.append(Request(f"{sid}-{step}", arr, str(sid), int(step or 0), gap, blocks, output_tokens=int(comp or 1)))
    reqs.sort(key=lambda r: (r.arrival_s, r.request_id))
    return Trace(name=f"freeinference-{len(uniq_sess)}sess-long", kind="freeinference", block_tokens=FREEINFERENCE_BLOCK_TOKENS, requests=reqs)


def make_agentic_synthetic(
    *,
    name: str,
    context_tokens: int,
    hit_rate: float,
    n_sessions: int = 24,
    turns_per_session: int = 5,
    gap_s: float = 1.5,
    target_rps: float = 4.0,
    seed: int = 0,
    include_short_probes: bool = False,
) -> Trace:
    """Controlled multi-turn agentic cluster workload with exact context length and reuse fraction."""
    block_tokens = 512 if context_tokens >= 512 else context_tokens
    n_blocks = max(1, context_tokens // block_tokens)
    if hit_rate <= 0.0:
        n_shared = 0
    elif hit_rate >= 1.0:
        n_shared = n_blocks
    else:
        n_shared = min(n_blocks, max(0, int(round(hit_rate * n_blocks))))
    n_unique = n_blocks - n_shared

    # Distribute sessions cleanly to hit target_rps while preserving realistic pre_gap_s
    total_reqs = n_sessions * turns_per_session
    total_dur = max(1.0, total_reqs / max(0.1, target_rps))
    session_span = max(0.0, (turns_per_session - 1) * gap_s)
    launch_window = max(0.5, total_dur - session_span)
    launch_step = launch_window / max(1, n_sessions)

    reqs: list[Request] = []
    for s in range(n_sessions):
        t0 = s * launch_step
        shared_blocks = tuple(BlockRef(f"{name}-s{s}-p{i}", block_tokens) for i in range(n_shared))
        for t in range(turns_per_session):
            if t == 0:
                # Turn 0 creates the prefix (or unique blocks if hit_rate == 0)
                uniq = tuple(BlockRef(f"{name}-s{s}-t0-u{j}", block_tokens) for j in range(n_unique))
                blocks = shared_blocks + uniq
            else:
                if n_shared == 0:
                    blocks = tuple(BlockRef(f"{name}-s{s}-t{t}-u{j}", block_tokens) for j in range(n_blocks))
                else:
                    uniq = tuple(BlockRef(f"{name}-s{s}-t{t}-u{j}", block_tokens) for j in range(n_unique))
                    blocks = shared_blocks + uniq
            reqs.append(
                Request(
                    request_id=f"{name}-s{s}-t{t}",
                    arrival_s=t0 + t * gap_s,
                    session_id=f"{name}-sess-{s}",
                    turn_index=t,
                    pre_gap_s=0.0 if t == 0 else gap_s,
                    blocks=blocks,
                    output_tokens=64,
                )
            )
    if include_short_probes:
        # Interleave short 1K-2K tool-check turns that share a 512-token prefix across workers,
        # exercising lossless selective recompute abort under MEMS OCS switching.
        for k in range(n_sessions):
            anchor_blk = (BlockRef(f"{name}-short-p{k}", 512),)
            reqs.append(
                Request(
                    request_id=f"{name}-short-warm-{k}",
                    arrival_s=k * 0.4,
                    session_id=f"{name}-short-w-{k}",
                    turn_index=0,
                    pre_gap_s=0.0,
                    blocks=anchor_blk + (BlockRef(f"{name}-short-wa-{k}", 512),),
                    output_tokens=16,
                )
            )
            reqs.append(
                Request(
                    request_id=f"{name}-short-probe-{k}",
                    arrival_s=k * 0.4 + 0.15,
                    session_id=f"{name}-short-p-{k}",
                    turn_index=0,
                    pre_gap_s=0.0,
                    blocks=anchor_blk + (BlockRef(f"{name}-short-pb-{k}", 512),),
                    output_tokens=16,
                )
            )
    reqs.sort(key=lambda r: (r.arrival_s, r.request_id))
    return Trace(name=name, kind="synthetic", block_tokens=block_tokens, requests=reqs, seed=seed)


# ---------------------------------------------------------------------------
# Suite 1: Capacity & Real 3090 Calibration Ledger CSVs
# ---------------------------------------------------------------------------

def write_suite1_capacity_and_calibration() -> None:
    print("[Suite 1] Writing capacity & RTX 3090 7B calibration tables...", flush=True)
    # 1a. Multi-architecture KV capacity
    cap_rows = []
    for spec in load_catalog().values():
        bf16 = spec.kv_bytes_per_token(2)
        fp8 = spec.kv_bytes_per_token(1)
        row = {
            "model": spec.name,
            "hf_id": spec.hf_id,
            "source_status": spec.source_status,
            "kv_arch": spec.kv_arch,
            "num_hidden_layers": spec.num_hidden_layers,
            "num_full_attention_layers": spec.num_full_attention_layers,
            "num_linear_attention_layers": spec.num_linear_attention_layers,
            "kv_bytes_bf16": bf16,
            "kv_bytes_fp8": fp8,
            "kv_kib_bf16": bf16 / 1024.0,
            "kv_kib_fp8": fp8 / 1024.0,
            "gib_at_128k_bf16": (131072 * bf16) / (1024**3),
            "gib_at_1m_bf16": (1000000 * bf16) / (1024**3),
            "gib_at_1m_fp8": (1000000 * fp8) / (1024**3),
            "tokens_in_24gib_bf16": (24 * 1024**3) // bf16,
            "tokens_in_80gib_bf16": (80 * 1024**3) // bf16,
            "simulation_context_cap": spec.simulation_context_cap,
        }
        cap_rows.append(row)
    write_csv(SIM / "fig1_kv_capacity.csv", cap_rows, list(cap_rows[0].keys()))

    # 1b. RTX 3090 7B calibration & roofline/quadratic validation table
    spec = load_model("qwen2.5-7b-instruct-1m")
    pm_3090 = build_prefill_model(spec, 2, ROOT / "configs/hardware/calibration_3090.yaml", hardware_profile="rtx3090")
    pm_h100 = build_prefill_model(spec, 2, ROOT / "configs/hardware/calibration_3090.yaml", hardware_profile="h100_peak_spec")
    meas_map = dict(pm_3090.study_piecewise)
    a_tr, b_tr, c_tr = fit_quadratic(pm_3090.study_piecewise[:6])  # trained 1K..16K, holdout 24K & 32K
    a_all, b_all, c_all = fit_quadratic(pm_3090.study_piecewise)   # all 8 pts 1K..32K

    cal_rows = []
    for s in (1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 65536, 131072, 262144, 524288, 1000000):
        meas = meas_map.get(s)
        roof_3090 = pm_3090.roofline_prefill_s(s) + pm_3090.roofline_decode_s(s)
        quad_tr = a_tr * s * s + b_tr * s + c_tr
        quad_all = a_all * s * s + b_all * s + c_all
        roof_h100 = pm_h100.roofline_prefill_s(s) + pm_h100.roofline_decode_s(s)
        study_3090 = sum(pm_3090.cold_parts(s))
        cal_rows.append(
            {
                "prompt_tokens": s,
                "measured_3090_7b_ttft_s": "" if meas is None else meas,
                "study_clock_3090_s": study_3090,
                "roofline_3090_s": roof_3090,
                "quad_train_1k_16k_s": quad_tr,
                "quad_all_8pts_s": quad_all,
                "h100_peak_spec_s": roof_h100,
                "roofline_abs_err_pct": "" if meas is None else abs(roof_3090 - meas) / meas * 100.0,
                "quad_train_abs_err_pct": "" if meas is None else abs(quad_tr - meas) / meas * 100.0,
                "split": "measured_train" if s <= 16384 else ("measured_heldout" if s <= 32768 else "extrapolated"),
            }
        )
    write_csv(SIM / "fig3_calibration_points.csv", cal_rows, list(cal_rows[0].keys()))

    # Also update fig3_prefill_model.csv for backward compatibility
    pref_rows = []
    for prof, pm in (("rtx3090", pm_3090), ("h100_peak_spec", pm_h100)):
        for s in (1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 65536, 131072, 262144, 524288, 1000000):
            p_s, _ = pm.cold_parts(s)
            pref_rows.append(
                {
                    "model": spec.name,
                    "hardware_profile": pm.profile,
                    "scaling_label": pm.scaling_label,
                    "calibration_status": pm.calibration_status,
                    "prompt_tokens": s,
                    "prefill_source": pm.source_for(s),
                    "prefill_s": p_s,
                    "quadratic_prefill_s": (a_all * s * s + b_all * s + c_all) if prof == "rtx3090" else "",
                }
            )
    write_csv(SIM / "fig3_prefill_model.csv", pref_rows, list(pref_rows[0].keys()))


# ---------------------------------------------------------------------------
# Suite 2: Main Trace Benchmarks (Mooncake + LMCache + FreeInference, Dual-Regime)
# ---------------------------------------------------------------------------

def run_suite2_main_traces(pool: ProcessPoolExecutor) -> None:
    print("[Suite 2] Loading real traces (Mooncake, LMCache tokenized, FreeInference 40-sess)...", flush=True)
    mc_full = load_mooncake_full()
    mc_3090_paced = pace_trace(
        Trace("mooncake-2k-3090", mc_full.kind, mc_full.block_tokens, mc_full.requests[:2000]),
        target_rps=0.8,
        name="mooncake-2k-3090-paced",
    )
    lm_120 = load_lmcache_tokenized(max_sessions=120, session_launch_gap_s=0.5)
    lm_all = load_lmcache_tokenized(max_sessions=None, session_launch_gap_s=0.8)
    fi_long = load_freeinference_long(max_sessions=25, session_launch_gap_s=1.0)

    # Save exact trace stats summary for paper Table
    trace_meta = [
        {"trace": "Mooncake (full)", "requests": len(mc_full.requests), "sessions": "", "p50_tokens": int(percentile([r.input_tokens for r in mc_full.requests], 0.5)), "p95_tokens": int(percentile([r.input_tokens for r in mc_full.requests], 0.95)), "max_tokens": max(r.input_tokens for r in mc_full.requests)},
        {"trace": "LMCache (all 767 sess, tokenized)", "requests": len(lm_all.requests), "sessions": 767, "p50_tokens": int(percentile([r.input_tokens for r in lm_all.requests], 0.5)), "p95_tokens": int(percentile([r.input_tokens for r in lm_all.requests], 0.95)), "max_tokens": max(r.input_tokens for r in lm_all.requests)},
        {"trace": "LMCache (120-sess benchmark)", "requests": len(lm_120.requests), "sessions": 120, "p50_tokens": int(percentile([r.input_tokens for r in lm_120.requests], 0.5)), "p95_tokens": int(percentile([r.input_tokens for r in lm_120.requests], 0.95)), "max_tokens": max(r.input_tokens for r in lm_120.requests)},
        {"trace": "FreeInference (25 long sess)", "requests": len(fi_long.requests), "sessions": 25, "p50_tokens": int(percentile([r.input_tokens for r in fi_long.requests], 0.5)), "p95_tokens": int(percentile([r.input_tokens for r in fi_long.requests], 0.95)), "max_tokens": max(r.input_tokens for r in fi_long.requests)},
    ]
    write_csv(SIM / "exp_trace_bench_scopes.csv", trace_meta, list(trace_meta[0].keys()))

    # Launch all policy runs in parallel across workers!
    suites = [
        # 1. Mooncake full (23,608 reqs) on H100-scaled cluster (SiPh 10 us, 200 GB/s vs 25 GB/s RDMA)
        ("exp_main_mooncake_h100_siph.csv", mc_full, "h100_peak_spec", make_cluster(hbm_gib=4.0, ocs_bw=200.0, tau_us=10.0, rdma_bw=25.0), 0.5, MAIN_POLICIES),
        # 2. Mooncake full (23,608 reqs) on H100-scaled cluster (MEMS 25 ms, 200 GB/s)
        ("exp_main_mooncake_h100_mems.csv", mc_full, "h100_peak_spec", make_cluster(hbm_gib=4.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0), 0.5, MAIN_POLICIES),
        # 3. Mooncake 3090-calibrated regime (2,000 reqs paced to 0.8 req/s, 3090 PCIe 26.28 GB/s DRAM, 45 GB/s OCS 25ms)
        ("exp_main_mooncake_3090_calibrated.csv", mc_3090_paced, "rtx3090", make_cluster(hbm_gib=4.0, dram_bw=26.28, ocs_bw=45.0, tau_us=25000.0, rdma_bw=25.0), 2.0, MAIN_POLICIES),
        # 4. LMCache multi-turn agentic (120 sessions / 5,046 reqs, H100, 4 GiB HBM/worker, MEMS 25 ms, 200 GB/s OCS)
        ("exp_main_lmcache_h100_mems.csv", lm_120, "h100_peak_spec", make_cluster(hbm_gib=4.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0), 0.25, MAIN_POLICIES),
        # 5. LMCache FULL 767 sessions / 24,880 reqs (H100, 4 GiB HBM/worker, MEMS 25 ms, 200 GB/s OCS)
        ("exp_main_lmcache_full24k_h100.csv", lm_all, "h100_peak_spec", make_cluster(hbm_gib=4.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0), 0.25, [
            "no_cache", "local_hbm", "local_dram", "rdma_routing", "optical_naive", "optical_reservation", "optikv", "oracle_infinite_hbm"
        ]),
        # 6. FreeInference long-context agentic (25 complete sessions, ~900 reqs, 80K median, H100, 8 GiB HBM/worker, MEMS 25 ms)
        ("exp_main_freeinference_h100_mems.csv", fi_long, "h100_peak_spec", make_cluster(hbm_gib=8.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0), 1.0, MAIN_POLICIES),
        # 7. 128K multi-turn agentic teaser workload (for Fig 1a breakdown + main comparison at 128K!)
        ("fig1_teaser_128k.csv", make_agentic_synthetic(name="agentic-128k", context_tokens=131072, hit_rate=0.90, n_sessions=24, turns_per_session=5, gap_s=1.5, target_rps=4.0), "h100_peak_spec", make_cluster(hbm_gib=12.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0), 1.0, [
            "no_cache", "local_hbm", "local_dram", "rdma_routing", "optical_naive", "optical_reservation", "optikv", "oracle_infinite_hbm"
        ]),
    ]

    futures = {}
    for fname, trace, hw, cl, slo, pols in suites:
        for pol in pols:
            job = {
                "config": {
                    "model": "qwen2.5-7b-instruct-1m",
                    "hardware_profile": hw,
                    "cluster": cl,
                    "policies": [pol],
                    "slo_ttft_s": slo,
                    "seed": 0,
                },
                "trace": trace,
                "meta": {"suite_file": fname, "policy_order": pols.index(pol)},
            }
            futures[pool.submit(_run_one_job, job)] = (fname, pol)

    grouped: dict[str, list[dict]] = {}
    expected_counts = {fname: len(pols) for fname, _, _, _, _, pols in suites}
    t0 = time.perf_counter()
    for fut in as_completed(futures):
        fname, pol = futures[fut]
        res = fut.result()
        grouped.setdefault(fname, []).extend(res["rows"])
        if len(grouped[fname]) == expected_counts[fname]:
            rows = sorted(grouped[fname], key=lambda r: r["policy_order"])
            write_csv(SIM / fname, rows, SUMMARY_COLUMNS)
            opt = next((r for r in rows if r["policy"] == "optikv"), None)
            rdma = next((r for r in rows if r["policy"] == "rdma_routing"), None)
            if opt and rdma:
                print(
                    f"  [Suite 2] {fname} ({time.perf_counter()-t0:.1f}s): RDMA p50={rdma['p50_ttft_s']*1e3:.1f}ms p99={rdma['p99_ttft_s']*1e3:.1f}ms | "
                    f"OptiKV p50={opt['p50_ttft_s']*1e3:.1f}ms p99={opt['p99_ttft_s']*1e3:.1f}ms duty={opt['ocs_duty_cycle']*100:.1f}%",
                    flush=True,
                )
    print(f"[Suite 2] Completed all 7 main trace suites in {time.perf_counter()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Suite 3: Break-Even Design Space Heatmap (10 contexts x 6 hit rates = 60 cells)
# ---------------------------------------------------------------------------

def run_suite3_heatmap(pool: ProcessPoolExecutor) -> None:
    print("[Suite 3] Running 10x6 Break-Even Heatmap (2K..1M x 0%..95% reuse)...", flush=True)
    t0 = time.perf_counter()
    contexts = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1000000]
    hit_rates = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95]
    policies = ["no_cache", "rdma_routing", "optical_naive", "optical_reservation", "optikv"]

    # H100-scaled 8-worker cluster with realistic 16 GiB KV HBM/worker + 256 GiB RDMA/Optical Pool
    # Note: 512K (28 GiB) and 1M (53.4 GiB) cleanly spill into the 256 GiB cluster DRAM/Optical/RDMA pool!
    cl = make_cluster(workers=8, hbm_gib=16.0, dram_gib=64.0, pool_gib=512.0, rdma_bw=25.0, ocs_bw=200.0, tau_us=25000.0, links=2)
    futures = {}
    for ctx in contexts:
        # Scale n_sessions moderately for 512K/1M so runtime is crisp while maintaining HBM overflow pressure
        n_sess = 16 if ctx <= 131072 else (10 if ctx <= 524288 else 8)
        n_turns = 4
        # Keep cluster below saturation on cached turns so network transfer dominates cleanly
        target_rps = 4.0 if ctx <= 32768 else (2.5 if ctx <= 131072 else (1.0 if ctx <= 524288 else 0.5))
        for hr in hit_rates:
            tr = make_agentic_synthetic(
                name=f"hm-{ctx}-{int(hr*100)}",
                context_tokens=ctx,
                hit_rate=hr,
                n_sessions=n_sess,
                turns_per_session=n_turns,
                gap_s=1.5,
                target_rps=target_rps,
                seed=0,
            )
            job = {
                "config": {
                    "model": "qwen2.5-7b-instruct-1m",
                    "hardware_profile": "h100_peak_spec",
                    "cluster": cl,
                    "policies": policies,
                    "slo_ttft_s": 2.0,
                    "seed": 0,
                },
                "trace": tr,
                "meta": {"context_tokens": ctx, "requested_hit_rate": hr},
            }
            futures[pool.submit(_run_one_job, job)] = (ctx, hr)

    cells = []
    for fut in as_completed(futures):
        ctx, hr = futures[fut]
        rows = fut.result()["rows"]
        by_p = {r["policy"]: r for r in rows}
        rdma = by_p["rdma_routing"]
        opt = by_p["optikv"]
        naive = by_p["optical_naive"]
        resv = by_p["optical_reservation"]
        nc = by_p["no_cache"]
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
                "speedup_vs_rdma_p50": rdma["p50_ttft_s"] / max(1e-9, opt["p50_ttft_s"]),
                "speedup_vs_rdma_p99": rdma["p99_ttft_s"] / max(1e-9, opt["p99_ttft_s"]),
                "speedup_vs_naive_p50": naive["p50_ttft_s"] / max(1e-9, opt["p50_ttft_s"]),
                "speedup_vs_nocache_p50": nc["p50_ttft_s"] / max(1e-9, opt["p50_ttft_s"]),
                "optikv_hit_rate": opt["hit_rate"],
                "optikv_ocs_duty": opt["ocs_duty_cycle"],
                "optikv_aborts": opt["abort_count"],
                "optikv_prefetches": opt["prefetch_bursts"],
                "no_benefit_vs_rdma": (rdma["p50_ttft_s"] / max(1e-9, opt["p50_ttft_s"])) <= 1.05,
            }
        )
    cells.sort(key=lambda r: (r["context_tokens"], r["requested_hit_rate"]))
    write_csv(SIM / "exp_breakeven_heatmap.csv", cells, list(cells[0].keys()))
    nb = sum(1 for c in cells if c["no_benefit_vs_rdma"])
    best = max(cells, key=lambda c: c["speedup_vs_rdma_p50"])
    print(
        f"[Suite 3] Heatmap 60 cells done in {time.perf_counter()-t0:.1f}s | "
        f"no-benefit cells (<=1.05x vs RDMA): {nb}/60 | peak speedup vs RDMA: "
        f"{best['speedup_vs_rdma_p50']:.2f}x at {best['context_tokens']} tok, {best['requested_hit_rate']*100:.0f}% reuse",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Suite 4: Optical Hardware Sensitivity + Multi-Arch / FP8 KV Sensitivity
# ---------------------------------------------------------------------------

def run_suite4_sensitivity(pool: ProcessPoolExecutor) -> None:
    print("[Suite 4] Running Reconfig, Bandwidth, Links, and Multi-Arch/FP8 sweeps...", flush=True)
    t0 = time.perf_counter()
    # Canonical 128K multi-turn agentic benchmark (24 sessions x 5 turns, 90% reuse, 1.5s gap)
    base_128k = make_agentic_synthetic(
        name="sens-128k",
        context_tokens=131072,
        hit_rate=0.90,
        n_sessions=24,
        turns_per_session=5,
        gap_s=1.5,
        target_rps=3.5,
        seed=0,
        include_short_probes=True,
    )
    policies = ["rdma_routing", "optical_naive", "optical_reservation", "optikv"]

    # 4a. Reconfiguration sweep: 0, 1us (SiPh), 10us, 100us, 1ms, 10ms, 25ms (MEMS), 50ms (MEMS)
    futs_tau = []
    for tau_us in (0, 1, 10, 100, 1000, 10000, 25000, 50000):
        cl = make_cluster(hbm_gib=12.0, ocs_bw=200.0, tau_us=tau_us, rdma_bw=25.0, links=2)
        futs_tau.append(
            pool.submit(
                _run_one_job,
                {
                    "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": policies, "slo_ttft_s": 1.0, "seed": 0},
                    "trace": base_128k,
                    "meta": {"sweep_tau_us": tau_us},
                },
            )
        )

    # 4b. Bandwidth sweep: 45 (Yoon & Yu), 50, 100, 200, 400, 800 GB/s
    futs_bw = []
    for bw in (45, 50, 100, 200, 400, 800):
        cl = make_cluster(hbm_gib=12.0, ocs_bw=bw, tau_us=25000.0, rdma_bw=25.0, links=2)
        futs_bw.append(
            pool.submit(
                _run_one_job,
                {
                    "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": policies, "slo_ttft_s": 1.0, "seed": 0},
                    "trace": base_128k,
                    "meta": {"sweep_bw_gb_s": bw},
                },
            )
        )

    # 4c. Concurrent links sweep: 1, 2, 4, 8
    futs_links = []
    for lk in (1, 2, 4, 8):
        cl = make_cluster(hbm_gib=12.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0, links=lk)
        futs_links.append(
            pool.submit(
                _run_one_job,
                {
                    "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": policies, "slo_ttft_s": 1.0, "seed": 0},
                    "trace": base_128k,
                    "meta": {"sweep_links": lk},
                },
            )
        )

    # 4d. Multi-Architecture x KV Precision (BF16 vs FP8) at 128K multi-turn
    arch_models = [
        "qwen2.5-7b-instruct-1m",
        "qwen3-8b",
        "llama-3.1-70b",
        "deepseek-v3",
        "qwen3.5-9b",
    ]
    # Note: keep 128K exact (131072) so every model stays within its simulation_context_cap!
    tr_arch = make_agentic_synthetic(
        name="arch-128k",
        context_tokens=131072,
        hit_rate=0.90,
        n_sessions=20,
        turns_per_session=4,
        gap_s=1.5,
        target_rps=2.5,
        seed=0,
    )
    futs_arch = []
    for mname in arch_models:
        spec = load_model(mname)
        for dtype_b, prec in ((2, "BF16"), (1, "FP8")):
            # Scale HBM budget proportionally to keep ~2-session HBM/worker overflow pressure consistent across 7B..70B
            hbm_gib = 16.0 if mname != "llama-3.1-70b" else 48.0
            cl = make_cluster(hbm_gib=hbm_gib, dram_gib=128.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0, links=2)
            futs_arch.append(
                pool.submit(
                    _run_one_job,
                    {
                        "config": {
                            "model": mname,
                            "dtype_bytes": dtype_b,
                            "hardware_profile": "h100_peak_spec",
                            "cluster": cl,
                            "policies": ["no_cache", "rdma_routing", "optical_naive", "optikv"],
                            "slo_ttft_s": 2.0,
                            "seed": 0,
                        },
                        "trace": tr_arch,
                        "meta": {"precision": prec, "kv_arch": spec.kv_arch},
                    },
                )
            )

    tau_rows = [r for f in futs_tau for r in f.result()["rows"]]
    bw_rows = [r for f in futs_bw for r in f.result()["rows"]]
    lk_rows = [r for f in futs_links for r in f.result()["rows"]]
    arch_rows = [r for f in futs_arch for r in f.result()["rows"]]

    write_csv(SIM / "exp_sensitivity_reconfig.csv", tau_rows, SUMMARY_COLUMNS)
    write_csv(SIM / "exp_sensitivity_bandwidth.csv", bw_rows, SUMMARY_COLUMNS)
    write_csv(SIM / "exp_sensitivity_links.csv", lk_rows, SUMMARY_COLUMNS)
    write_csv(SIM / "exp_sensitivity_arch_fp8.csv", arch_rows, ["precision", "kv_arch"] + SUMMARY_COLUMNS)
    print(f"[Suite 4] Completed all sensitivity & multi-arch/FP8 sweeps in {time.perf_counter()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Suite 5: Stepwise Ablation, Modern Eviction Baselines, Noise Robustness
# ---------------------------------------------------------------------------

def run_suite5_ablation_eviction_noise(pool: ProcessPoolExecutor) -> None:
    print("[Suite 5] Running Stepwise Ablation, Eviction Comparison, and Noise Robustness...", flush=True)
    t0 = time.perf_counter()
    # Mixed multi-turn agentic workload (128K multi-turn sessions + short 1K tool probes + real 1.5s gaps + tight HBM)
    abl_trace = make_agentic_synthetic(
        name="ablation-agentic",
        context_tokens=131072,
        hit_rate=0.90,
        n_sessions=24,
        turns_per_session=5,
        gap_s=1.5,
        target_rps=3.5,
        seed=0,
        include_short_probes=True,
    )
    cl_abl = make_cluster(hbm_gib=10.0, dram_gib=64.0, pool_gib=256.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0, links=2)

    # Cumulative stepwise progression from Optical-Naive -> Full OptiKV-Fabric
    step_specs = [
        ("Optical-Naive", "optical_naive", None),
        ("+StandingCircuit", "optical_reservation", None),
        ("+OpticalRouting", "optikv", {"enable_routing": True, "enable_recompute_fallback": False, "enable_placement": False, "enable_prefetch": False, "enable_eviction": False}),
        ("+RecomputeAbort", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": False, "enable_prefetch": False, "enable_eviction": False}),
        ("+TierPlacement", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": True, "enable_prefetch": False, "enable_eviction": False}),
        ("+WindowedPrefetch", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": True, "enable_prefetch": True, "enable_eviction": False}),
        ("+ReuseEviction (Full)", "optikv", {"enable_routing": True, "enable_recompute_fallback": True, "enable_placement": True, "enable_prefetch": True, "enable_eviction": True}),
    ]
    futs_abl = []
    for idx, (label, pol, flags) in enumerate(step_specs):
        cfg = {
            "model": "qwen2.5-7b-instruct-1m",
            "hardware_profile": "h100_peak_spec",
            "cluster": cl_abl,
            "policies": [pol],
            "slo_ttft_s": 1.0,
            "seed": 0,
        }
        if flags is not None:
            cfg["optikv"] = flags
        futs_abl.append(pool.submit(_run_one_job, {"config": cfg, "trace": abl_trace, "meta": {"step_idx": idx, "ablation_step": label}}))

    # 5b. Modern Eviction comparison on real LMCache 120-session trace under tight HBM (2 GiB/worker)
    lm_120 = load_lmcache_tokenized(max_sessions=120, session_launch_gap_s=0.5)
    cl_ev = make_cluster(hbm_gib=2.0, dram_gib=0.0, ssd_gib=0.0, pool_gib=0.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0)
    ev_specs = [
        ("LRU (Local-HBM)", "local_hbm", None),
        ("S3-FIFO", "eviction_s3fifo", None),
        ("OptiKV-GDSF (HBM-only)", "optikv", {"enable_routing": False, "enable_recompute_fallback": False, "enable_placement": False, "enable_prefetch": False, "enable_eviction": True}),
        ("Belady (Offline Optimal)", "eviction_belady", None),
    ]
    futs_ev = []
    for idx, (label, pol, flags) in enumerate(ev_specs):
        cfg = {
            "model": "qwen2.5-7b-instruct-1m",
            "hardware_profile": "h100_peak_spec",
            "cluster": cl_ev,
            "policies": [pol],
            "slo_ttft_s": 0.5,
            "seed": 0,
        }
        if flags is not None:
            cfg["optikv"] = flags
        futs_ev.append(pool.submit(_run_one_job, {"config": cfg, "trace": lm_120, "meta": {"ev_idx": idx, "eviction_label": label}}))

    # 5c. P(reuse) Prediction Noise Robustness (sigma = 0, 5%, 10%, 20%, 3 seeds each)
    futs_noise = []
    for sigma in (0.0, 0.05, 0.10, 0.20):
        for seed in (0, 1, 2):
            cfg = {
                "model": "qwen2.5-7b-instruct-1m",
                "hardware_profile": "h100_peak_spec",
                "cluster": cl_abl,
                "policies": ["optikv"],
                "reuse_noise_std": sigma,
                "slo_ttft_s": 1.0,
                "seed": seed,
            }
            futs_noise.append(pool.submit(_run_one_job, {"config": cfg, "trace": abl_trace, "meta": {"sigma": sigma}}))

    abl_rows = sorted([r for f in futs_abl for r in f.result()["rows"]], key=lambda r: r["step_idx"])
    ev_rows = sorted([r for f in futs_ev for r in f.result()["rows"]], key=lambda r: r["ev_idx"])
    noise_rows = [r for f in futs_noise for r in f.result()["rows"]]

    write_csv(SIM / "exp_ablation.csv", abl_rows, ["step_idx", "ablation_step"] + SUMMARY_COLUMNS)
    write_csv(SIM / "exp_eviction_comparison.csv", ev_rows, ["ev_idx", "eviction_label"] + SUMMARY_COLUMNS)
    write_csv(SIM / "exp_noise_robustness.csv", noise_rows, ["sigma"] + SUMMARY_COLUMNS)
    for r in abl_rows:
        print(f"  [Ablation] {r['ablation_step']:24s} p50={r['p50_ttft_s']*1e3:7.1f}ms p99={r['p99_ttft_s']*1e3:7.1f}ms abrt={r['abort_count']:3d} pf={r['prefetch_bursts']:3d}", flush=True)
    print(f"[Suite 5] Completed in {time.perf_counter()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Suite 6: Tail Latency & SLO Goodput Under Load (0.5 -> 16 req/s)
# ---------------------------------------------------------------------------

def run_suite6_load_sweep(pool: ProcessPoolExecutor) -> None:
    print("[Suite 6] Running Load Sweep (0.5 .. 16 req/s, P99 TTFT + SLO Goodput + Goodput/W)...", flush=True)
    t0 = time.perf_counter()
    rates = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    policies = ["no_cache", "local_hbm", "local_dram", "rdma_routing", "optical_naive", "optical_reservation", "optikv", "oracle_infinite_hbm"]
    cl = make_cluster(workers=8, hbm_gib=12.0, dram_gib=64.0, pool_gib=256.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0, links=2)
    futs = []
    for rps in rates:
        tr = make_agentic_synthetic(
            name=f"load-{rps:g}rps",
            context_tokens=131072,
            hit_rate=0.90,
            n_sessions=24,
            turns_per_session=5,
            gap_s=1.5,
            target_rps=rps,
            seed=0,
        )
        for pol in policies:
            futs.append(
                pool.submit(
                    _run_one_job,
                    {
                        "config": {
                            "model": "qwen2.5-7b-instruct-1m",
                            "hardware_profile": "h100_peak_spec",
                            "cluster": cl,
                            "policies": [pol],
                            "slo_ttft_s": 1.0,
                            "seed": 0,
                        },
                        "trace": tr,
                        "meta": {"target_rps": rps},
                    },
                )
            )
    rows = [r for f in futs for r in f.result()["rows"]]
    rows.sort(key=lambda r: (r["target_rps"], policies.index(r["policy"])))
    write_csv(SIM / "exp_load_sweep.csv", rows, ["target_rps"] + SUMMARY_COLUMNS)
    print(f"[Suite 6] Completed load sweep in {time.perf_counter()-t0:.1f}s", flush=True)


def main() -> int:
    SIM.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    print(f"=== OptiKV-Fabric Master Suite (workers={MAX_WORKERS}) ===", flush=True)
    write_suite1_capacity_and_calibration()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        run_suite2_main_traces(pool)
        run_suite3_heatmap(pool)
        run_suite4_sensitivity(pool)
        run_suite5_ablation_eviction_noise(pool)
        run_suite6_load_sweep(pool)
    print(f"\n=== ALL SUITES COMPLETED IN {time.perf_counter()-t_all:.1f}s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
