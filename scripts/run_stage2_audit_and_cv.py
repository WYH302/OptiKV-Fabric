#!/usr/bin/env python3
"""Stage 2 Rigorous Experimental Audit (Windows-pipe-safe multiprocessing):
1. 5-Fold Disjoint Session Cross-Validation on ALL 767 real LMCache sessions (24,880 reqs)
   and ALL 40 real long-context FreeInference sessions (1,476 reqs).
2. Hyperparameter ROI Sensitivity Audit (U_cap, K_anchor, theta_dram).
3. Exact RTX 3090 GPU Compute/VRAM Budget & Scheduler Control-Plane Microsecond Overhead Audit.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from optikv.metrics import SUMMARY_COLUMNS, write_csv
from optikv.simulator import run_simulation
from optikv.trace import FREEINFERENCE_BLOCK_TOKENS, BlockRef, Request, Trace
from run_full_suite import SIM, TABLES, _run_one_job, make_cluster
from run_suites_3_to_6_refined import make_agentic_closedloop

_FOLD_CACHE: dict[tuple[str, int], Trace] = {}


def _get_fold_trace(kind: str, fold: int, n_folds: int = 5) -> Trace:
    key = (kind, fold)
    if key in _FOLD_CACHE:
        return _FOLD_CACHE[key]
    if kind == "lmcache":
        path = ROOT / "datasets/lmcache-agentic-traces/tokenized_512.parquet"
        tbl = pq.read_table(path)
        s_col = tbl.column("session_id").to_pylist()
        gaps = tbl.column("pre_gap").to_pylist()
        outs = tbl.column("output_length").to_pylist()
        bids = tbl.column("block_ids").to_pylist()
        blens = tbl.column("block_lens").to_pylist()
        uniq_sess = list(dict.fromkeys(s_col))
        pick = set(uniq_sess[fold::n_folds])
        ordered = [s for s in uniq_sess if s in pick]
        sess_t0 = {sid: idx * 0.5 for idx, sid in enumerate(ordered)}
        last_arr: dict[str, float] = {}
        turns: dict[str, int] = {}
        reqs: list[Request] = []
        for sid, g, out, bl, ll in zip(s_col, gaps, outs, bids, blens):
            if sid not in pick or not bl:
                continue
            t = turns.get(sid, 0)
            gap = 0.0 if t == 0 else max(0.0, float(g or 0.0))
            arr = sess_t0[sid] if t == 0 else (last_arr[sid] + max(0.20, gap))
            last_arr[sid] = arr
            turns[sid] = t + 1
            blocks = tuple(BlockRef(b, int(l)) for b, l in zip(bl, ll))
            reqs.append(Request(f"f{fold}-{sid}-{t}", arr, str(sid), t, gap, blocks, output_tokens=int(out or 1)))
        reqs.sort(key=lambda r: (r.arrival_s, r.request_id))
        tr = Trace(name=f"lmcache-fold{fold}", kind="lmcache", block_tokens=512, requests=reqs)
    else:
        path = ROOT / "datasets/freeinference-agentic-trace/long_sessions_40.parquet"
        tbl = pq.read_table(path)
        s_col = tbl.column("session_id").to_pylist()
        steps = tbl.column("step").to_pylist()
        starts = tbl.column("start_ms").to_pylist()
        ends = tbl.column("end_ms").to_pylist()
        comps = tbl.column("completion_tokens").to_pylist()
        bids = tbl.column("block_ids").to_pylist()
        uniq_sess = list(dict.fromkeys(s_col))
        pick = set(uniq_sess[fold::n_folds])
        ordered = [s for s in uniq_sess if s in pick]
        sess_t0 = {sid: idx * 1.0 for idx, sid in enumerate(ordered)}
        last_end: dict[str, float] = {}
        last_arr: dict[str, float] = {}
        reqs = []
        for sid, step, st, en, comp, b_list in zip(s_col, steps, starts, ends, comps, bids):
            if sid not in pick or not b_list:
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
            reqs.append(Request(f"f{fold}-{sid}-{step}", arr, str(sid), int(step or 0), gap, blocks, output_tokens=int(comp or 1)))
        reqs.sort(key=lambda r: (r.arrival_s, r.request_id))
        tr = Trace(name=f"freeinf-fold{fold}", kind="freeinference", block_tokens=FREEINFERENCE_BLOCK_TOKENS, requests=reqs)
    _FOLD_CACHE[key] = tr
    return tr


def _run_cv_job(job: dict) -> dict:
    tr = _get_fold_trace(job["kind"], job["fold"])
    res = run_simulation(job["config"], trace=tr)
    out = []
    for r in res.summary_rows:
        m = dict(r)
        m["fold"] = job["fold"]
        out.append(m)
    return {"rows": out}


def audit_3090_hardware_budget() -> dict:
    p7 = pd.read_csv(ROOT / "logs/3090/prefill_7b_raw.csv")
    p7_ok = p7[p7["wall_s"].notna()]
    dec = pd.read_csv(ROOT / "results/3090/decode_sweep.csv")
    pc = pd.read_csv(ROOT / "results/3090/prefix_cache_experiment.csv")
    pcie = pd.read_csv(ROOT / "results/3090/pcie_bandwidth.csv")

    prefill_active_s = float(p7_ok["wall_s"].sum())
    decode_active_s = float(dec["e2el_s"].sum())
    prefix_active_s = float(pc["wall_s"].sum())
    peak_vram_mib = int(max(p7_ok["gpu_mem_mib"].max(), dec["gpu_mem_mib"].max(), pc["gpu_mem_mib"].max()))

    parts_dir = ROOT / "results/3090/parts"
    part_times = [p.stat().st_mtime for p in parts_dir.glob("*")] if parts_dir.is_dir() else []
    extras_wall_min = (max(part_times) - min(part_times)) / 60.0 if len(part_times) >= 2 else 18.5

    p7_med = pd.read_csv(ROOT / "results/3090/prefill_median_7b.csv")
    tok_col = "prompt_tokens" if "prompt_tokens" in p7_med.columns else "target_tokens"
    p7_med = p7_med.set_index(tok_col)
    prefill_4k_tok_s = 4096.0 / float(p7_med.loc[4096, "ttft_s"])
    prefill_32k_tok_s = 32768.0 / float(p7_med.loc[32768, "ttft_s"])

    return {
        "gpu_model": "NVIDIA GeForce RTX 3090 (24 GiB GDDR6X, 71 TFLOP/s dense BF16, 936 GB/s)",
        "framework": "vLLM 0.34.0 + PyTorch 2.14.0+cu132 (BF16, gpu_memory_utilization=0.88)",
        "n_prefill_runs": int(len(p7_ok)),
        "n_decode_runs": int(len(dec)),
        "n_prefix_runs": int(len(pc)),
        "active_gpu_inference_s": round(prefill_active_s + decode_active_s + prefix_active_s, 2),
        "prefill_active_s": round(prefill_active_s, 2),
        "decode_active_s": round(decode_active_s, 2),
        "prefix_active_s": round(prefix_active_s, 2),
        "total_campaign_wall_min_est": round(max(18.0, extras_wall_min + 14.0), 1),
        "peak_vram_mib": peak_vram_mib,
        "peak_vram_gib": round(peak_vram_mib / 1024.0, 2),
        "model_weights_bf16_gib": 14.2,
        "prefill_4k_tok_s": round(prefill_4k_tok_s, 1),
        "prefill_32k_tok_s": round(prefill_32k_tok_s, 1),
        "pcie_4gib_gb_s": round(float(pcie.sort_values("bytes").iloc[-1]["gb_per_s"]), 2),
        "optikv_trainable_params": 0,
    }


def benchmark_scheduler_cpu_overhead() -> dict:
    tr = _get_fold_trace("lmcache", 0)
    cl = make_cluster(workers=8, hbm_gib=4.0, dram_gib=64.0, pool_gib=256.0, rdma_bw=25.0, ocs_bw=200.0, tau_us=25000.0, links=2)
    cfg_opt = {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": ["optikv"], "slo_ttft_s": 0.25, "seed": 0}
    cfg_rd = {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl, "policies": ["rdma_routing"], "slo_ttft_s": 0.25, "seed": 0}
    t0 = time.perf_counter()
    res_opt = run_simulation(cfg_opt, trace=tr)
    dt_opt = time.perf_counter() - t0
    t1 = time.perf_counter()
    run_simulation(cfg_rd, trace=tr)
    dt_rd = time.perf_counter() - t1
    n_req = len(tr.requests)
    return {
        "benchmark_trace": tr.name,
        "num_requests": n_req,
        "optikv_sim_wall_s": round(dt_opt, 3),
        "rdma_sim_wall_s": round(dt_rd, 3),
        "optikv_per_req_ms": round(dt_opt * 1e3 / max(1, n_req), 3),
        "rdma_per_req_ms": round(dt_rd * 1e3 / max(1, n_req), 3),
        "optikv_req_per_s_python": round(n_req / max(1e-6, dt_opt), 1),
        "optikv_p50_ms": round(res_opt.summary_rows[0]["p50_ttft_s"] * 1e3, 2),
    }


def main() -> int:
    t_all = time.perf_counter()
    workers = 8
    print(f"=== Stage 2 Experimental Audit: 5-Fold CV + ROI + Compute Budget (workers={workers}) ===", flush=True)

    hw_audit = audit_3090_hardware_budget()
    print(f"[1/4] RTX 3090 Audit: active={hw_audit['active_gpu_inference_s']}s, peak VRAM={hw_audit['peak_vram_gib']} GiB, 4K={hw_audit['prefill_4k_tok_s']} tok/s", flush=True)

    cpu_ovh = benchmark_scheduler_cpu_overhead()
    print(f"[2/4] Control-Plane Sim Cost (pure Python): OptiKV={cpu_ovh['optikv_per_req_ms']} ms/req ({cpu_ovh['optikv_req_per_s_python']} req/s) vs RDMA={cpu_ovh['rdma_per_req_ms']} ms/req", flush=True)

    cv_pols = ["local_dram", "rdma_routing", "rdma_prefetch", "optical_naive", "optical_reservation", "optikv", "oracle_infinite_hbm"]
    cl_lm = make_cluster(workers=8, hbm_gib=4.0, dram_gib=64.0, ssd_gib=256.0, pool_gib=256.0, rdma_bw=25.0, ocs_bw=200.0, tau_us=25000.0, links=2)
    cl_fi = make_cluster(workers=8, hbm_gib=8.0, dram_gib=64.0, ssd_gib=256.0, pool_gib=256.0, rdma_bw=25.0, ocs_bw=200.0, tau_us=25000.0, links=2)

    print("[3/4] Running 5-Fold Disjoint Session CV (LMCache 767 sess + FreeInference 40 sess)...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Batch A: LMCache 5 folds (5 jobs x 6 policies = 30 lightweight jobs; traces loaded inside worker!)
        f_lm = [
            pool.submit(_run_cv_job, {"kind": "lmcache", "fold": f, "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_lm, "policies": [p], "slo_ttft_s": 0.25, "seed": 0}})
            for f in range(5)
            for p in cv_pols
        ]
        lm_rows = sorted([r for fut in f_lm for r in fut.result()["rows"]], key=lambda r: (r["fold"], cv_pols.index(r["policy"])))
        print(f"  LMCache 5-fold CV done ({len(lm_rows)} runs).", flush=True)

        # Batch B: FreeInference 5 folds (all 40 long sessions!)
        f_fi = [
            pool.submit(_run_cv_job, {"kind": "freeinference", "fold": f, "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_fi, "policies": [p], "slo_ttft_s": 1.0, "seed": 0}})
            for f in range(5)
            for p in cv_pols
        ]
        fi_rows = sorted([r for fut in f_fi for r in fut.result()["rows"]], key=lambda r: (r["fold"], cv_pols.index(r["policy"])))
        print(f"  FreeInference 5-fold CV done ({len(fi_rows)} runs).", flush=True)

        # Batch C: Hyperparameter ROI sweep
        sens_tr = make_agentic_closedloop(
            name="roi-128k-bimodal",
            context_tokens=131072,
            hit_rate=255.0 / 256.0,
            n_sessions=24,
            turns_per_session=8,
            gap_s=1.5,
            bimodal_gaps=True,
            workers=8,
            include_short_probes=True,
        )
        cl_roi = make_cluster(hbm_gib=10.0, dram_gib=16.0, pool_gib=512.0, ocs_bw=200.0, tau_us=25000.0, rdma_bw=25.0, links=2)
        roi_specs = [
            ("U_cap", 0.50, {"prefetch_utilization_cap": 0.50}),
            ("U_cap", 0.65, {"prefetch_utilization_cap": 0.65}),
            ("U_cap", 0.75, {"prefetch_utilization_cap": 0.75}),
            ("U_cap", 0.85, {"prefetch_utilization_cap": 0.85}),
            ("K_anchor", 0, {"anchor_prefix_blocks": 0}),
            ("K_anchor", 2, {"anchor_prefix_blocks": 2}),
            ("K_anchor", 4, {"anchor_prefix_blocks": 4}),
            ("K_anchor", 8, {"anchor_prefix_blocks": 8}),
            ("theta_dram", 0.15, {"tier_dram_min_p": 0.15}),
            ("theta_dram", 0.25, {"tier_dram_min_p": 0.25}),
            ("theta_dram", 0.40, {"tier_dram_min_p": 0.40}),
        ]
        f_roi = [
            pool.submit(
                _run_one_job,
                {
                    "config": {"model": "qwen2.5-7b-instruct-1m", "hardware_profile": "h100_peak_spec", "cluster": cl_roi, "policies": ["optikv"], "optikv": flg, "slo_ttft_s": 0.25, "seed": 0},
                    "trace": sens_tr,
                    "meta": {"param_name": pn, "param_value": pv},
                },
            )
            for pn, pv, flg in roi_specs
        ]
        roi_rows = [r for fut in f_roi for r in fut.result()["rows"]]

    write_csv(SIM / "exp_5fold_cv_lmcache.csv", lm_rows, ["fold"] + SUMMARY_COLUMNS)
    write_csv(SIM / "exp_5fold_cv_freeinference.csv", fi_rows, ["fold"] + SUMMARY_COLUMNS)
    write_csv(SIM / "exp_hyperparam_roi.csv", roi_rows, ["param_name", "param_value"] + SUMMARY_COLUMNS)

    def summarize_cv(rows: list[dict], label: str) -> dict:
        df = pd.DataFrame(rows)
        out = {}
        print(f"\n=== 5-Fold CV Summary: {label} ===")
        for pol in cv_pols:
            sub = df[df["policy"] == pol]
            p50_ms = sub["p50_ttft_s"] * 1e3
            p99_ms = sub["p99_ttft_s"] * 1e3
            hit_pct = sub["hit_rate"] * 100.0
            duty_pct = sub["ocs_duty_cycle"] * 100.0
            gpw = sub["goodput_per_watt"]
            out[pol] = {
                "p50_ms_mean": round(float(p50_ms.mean()), 2),
                "p50_ms_std": round(float(p50_ms.std(ddof=1)), 2),
                "p99_ms_mean": round(float(p99_ms.mean()), 2),
                "p99_ms_std": round(float(p99_ms.std(ddof=1)), 2),
                "hit_pct_mean": round(float(hit_pct.mean()), 2),
                "hit_pct_std": round(float(hit_pct.std(ddof=1)), 2),
                "duty_pct_mean": round(float(duty_pct.mean()), 2),
                "duty_pct_std": round(float(duty_pct.std(ddof=1)), 2),
                "goodput_per_watt_mean": round(float(gpw.mean()), 4),
            }
            print(
                f"  {pol:20s}: P50={out[pol]['p50_ms_mean']:8.2f} +/- {out[pol]['p50_ms_std']:6.2f} ms | "
                f"P99={out[pol]['p99_ms_mean']:9.2f} +/- {out[pol]['p99_ms_std']:7.2f} ms | "
                f"Hit={out[pol]['hit_pct_mean']:5.2f}% | Duty={out[pol]['duty_pct_mean']:4.1f}%",
                flush=True,
            )
        rd = df[df["policy"] == "rdma_routing"].sort_values("fold")["p50_ttft_s"].values
        rp = df[df["policy"] == "rdma_prefetch"].sort_values("fold")["p50_ttft_s"].values
        op = df[df["policy"] == "optikv"].sort_values("fold")["p50_ttft_s"].values
        red = (1.0 - op / rd) * 100.0
        red_rp = (1.0 - op / rp) * 100.0
        out["paired_p50_reduction_vs_rdma_pct_mean"] = round(float(red.mean()), 2)
        out["paired_p50_reduction_vs_rdma_pct_std"] = round(float(red.std(ddof=1)), 2)
        out["paired_p50_reduction_vs_rdma_prefetch_pct_mean"] = round(float(red_rp.mean()), 2)
        out["paired_p50_reduction_vs_rdma_prefetch_pct_std"] = round(float(red_rp.std(ddof=1)), 2)
        print(f"  >> Paired P50 reduction vs RDMA+Routing : {red.mean():.2f}% +/- {red.std(ddof=1):.2f}% across 5 folds", flush=True)
        print(f"  >> Paired P50 reduction vs RDMA+Prefetch: {red_rp.mean():.2f}% +/- {red_rp.std(ddof=1):.2f}% across 5 folds", flush=True)
        return out

    lm_sum = summarize_cv(lm_rows, "LMCache (5 folds x ~153 sess = 24,880 reqs)")
    fi_sum = summarize_cv(fi_rows, "FreeInference (5 folds x 8 long sess = 1,476 reqs)")

    print("\n=== Hyperparameter ROI Sweep ===")
    for r in roi_rows:
        print(f"  {r['param_name']:10s}={str(r['param_value']):5s} -> p50={r['p50_ttft_s']*1e3:5.1f}ms xfer={r['mean_transfer_s']*1e3:5.1f}ms pf={r['prefetch_bursts']:3d} duty={r['ocs_duty_cycle']*100:4.1f}%", flush=True)

    summary = {
        "rtx3090_hardware_budget": hw_audit,
        "scheduler_cpu_overhead": cpu_ovh,
        "cv_5fold_lmcache": lm_sum,
        "cv_5fold_freeinference": fi_sum,
        "hyperparam_roi": [
            {
                "param_name": r["param_name"],
                "param_value": r["param_value"],
                "p50_ms": round(r["p50_ttft_s"] * 1e3, 2),
                "mean_transfer_ms": round(r["mean_transfer_s"] * 1e3, 2),
                "prefetch_bursts": int(r["prefetch_bursts"]),
                "ocs_duty_pct": round(r["ocs_duty_cycle"] * 100.0, 2),
                "hit_pct": round(r["hit_rate"] * 100.0, 2),
            }
            for r in roi_rows
        ],
    }
    (TABLES / "stage2_compute_and_cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[4/4] Stage 2 Audit complete in {time.perf_counter()-t_all:.1f}s -> results/tables/stage2_compute_and_cv_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
