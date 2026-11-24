#!/usr/bin/env python3
"""Real NVIDIA RTX 3090 vLLM 0.34.0 Benchmarks (Qwen2.5-7B-Instruct BF16):
Part 2A: Multi-length Prefix-Hit Incremental Prefill Calibration (4K..30K cached prefix + 128/512 suffix).
Part 2B: Multi-Session HBM Capacity Wall & Prefix Eviction Cliff (N = 1, 2, 4, 8, 12, 16, 20 x 8K sessions).
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import bench_3090  # noqa: E402
from optikv.modelspec import load_model  # noqa: E402
from optikv.prefill import build_prefill_model  # noqa: E402

MODEL_DIR = ROOT / "models" / "Qwen2.5-7B-Instruct"
RES = ROOT / "results" / "3090"
OUT_PREFIX = RES / "prefix_hit_scaling_7b.csv"
OUT_WALL = RES / "hbm_capacity_wall_7b.csv"


def main() -> int:
    t_all = time.perf_counter()
    RES.mkdir(parents=True, exist_ok=True)
    LLM, SamplingParams = bench_3090.import_vllm()
    tokenizer = bench_3090.load_tokenizer(str(MODEL_DIR))

    # Build analytical roofline prefill clock for direct MAPE validation on hot prefix hits
    model_spec = load_model("qwen2.5-7b-instruct")
    prefill_model = build_prefill_model(
        model_spec,
        dtype_bytes=2,
        calibration_path=ROOT / "configs" / "hardware" / "calibration_3090.yaml",
        hardware_profile="rtx3090",
        quadratic_extrapolation=False,
    )

    llm = bench_3090.build_llm(
        LLM,
        str(MODEL_DIR),
        max_model_len=32768,
        prefix_caching=True,
        gpu_mem=0.88,
    )
    try:
        # Global warmup so CUDA graphs and kernels reach steady P0 clock
        warm_ids = bench_3090.build_ids(tokenizer, 1024, salt=999001)
        bench_3090.run_once(llm, SamplingParams, warm_ids, max_tokens=1)
        bench_3090.run_once(llm, SamplingParams, warm_ids, max_tokens=1)

        # =====================================================================
        # Part 2A: Multi-Length Prefix-Hit Incremental Prefill Scaling (4K..30K)
        # =====================================================================
        print("\n=== Part 2A: Real RTX 3090 Prefix-Hit Incremental Prefill Scaling ===", flush=True)
        grid = [
            (4096, 128),
            (4096, 512),
            (8192, 128),
            (8192, 512),
            (12288, 512),
            (16384, 128),
            (16384, 512),
            (24576, 128),
            (24576, 512),
            (30720, 512),
        ]
        salt_base = 10000
        prefix_rows = []
        for idx, (plen, slen) in enumerate(grid):
            cold_vals, hot_vals, cached_vals = [], [], []
            # 1 warmup trial + 3 measured trials
            for rep in range(4):
                salt = salt_base + idx * 100 + rep * 10
                prefix_ids = bench_3090.build_ids(tokenizer, plen, salt=salt)
                suf_a = bench_3090.build_ids(tokenizer, slen, salt=salt + 1)
                suf_b = bench_3090.build_ids(tokenizer, slen, salt=salt + 2)
                r1 = bench_3090.run_once(llm, SamplingParams, prefix_ids + suf_a, max_tokens=1)
                r2 = bench_3090.run_once(llm, SamplingParams, prefix_ids + suf_b, max_tokens=1)
                if rep >= 1:
                    cold_vals.append(float(r1["ttft_s"]))
                    hot_vals.append(float(r2["ttft_s"]))
                    cached_vals.append(int(r2["cached"]) if str(r2["cached"]).isdigit() else 0)
            cold_med = statistics.median(cold_vals)
            hot_med = statistics.median(hot_vals)
            cached_med = int(statistics.median(cached_vals))
            tot_tok = plen + slen
            # Analytical delta prefill prediction + 1st-token decode from our roofline model
            pred_prefill_s, pred_dec_s = prefill_model.request_times(cached_med, tot_tok - cached_med)
            pred_hot_s = pred_prefill_s + pred_dec_s
            pred_cold_p, pred_cold_d = prefill_model.request_times(0, tot_tok)
            pred_cold_s = pred_cold_p + pred_cold_d
            hot_err_pct = abs(pred_hot_s - hot_med) / max(1e-9, hot_med) * 100.0
            cold_err_pct = abs(pred_cold_s - cold_med) / max(1e-9, cold_med) * 100.0
            row = {
                "prefix_tokens": plen,
                "suffix_tokens": slen,
                "total_tokens": tot_tok,
                "cached_tokens_med": cached_med,
                "cold_ttft_ms": round(cold_med * 1e3, 2),
                "hot_ttft_ms": round(hot_med * 1e3, 2),
                "speedup": round(cold_med / max(1e-9, hot_med), 2),
                "roofline_cold_ms": round(pred_cold_s * 1e3, 2),
                "roofline_hot_ms": round(pred_hot_s * 1e3, 2),
                "cold_err_pct": round(cold_err_pct, 2),
                "hot_err_pct": round(hot_err_pct, 2),
            }
            prefix_rows.append(row)
            print(
                f"  prefix={plen:5d} + suf={slen:3d} | cached={cached_med:5d} | "
                f"cold={row['cold_ttft_ms']:7.1f}ms (pred {row['roofline_cold_ms']:7.1f}ms, err {row['cold_err_pct']:4.1f}%) | "
                f"hot={row['hot_ttft_ms']:6.1f}ms (pred {row['roofline_hot_ms']:6.1f}ms, err {row['hot_err_pct']:4.1f}%) | "
                f"speedup={row['speedup']:5.2f}x",
                flush=True,
            )

        with OUT_PREFIX.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(prefix_rows[0].keys()))
            w.writeheader()
            w.writerows(prefix_rows)
        print(f"Saved {OUT_PREFIX} ({len(prefix_rows)} points)", flush=True)

        # =====================================================================
        # Part 2B: Multi-Session HBM Capacity Wall & LRU Prefix Eviction Cliff
        # =====================================================================
        print("\n=== Part 2B: Real RTX 3090 Multi-Session HBM Capacity Wall (8K/session) ===", flush=True)
        sess_plen = 8192
        sess_slen = 128
        kv_bytes_per_tok = 57344  # Qwen2.5-7B BF16: 56 KiB/tok
        wall_rows = []
        for n_idx, n_sess in enumerate((1, 2, 4, 8, 10, 12, 14, 16, 20)):
            sbase = 50000 + n_idx * 1000
            # Build N distinct 8K prefixes
            prefixes = [bench_3090.build_ids(tokenizer, sess_plen, salt=sbase + s * 3) for s in range(n_sess)]
            suf1 = [bench_3090.build_ids(tokenizer, sess_slen, salt=sbase + s * 3 + 1) for s in range(n_sess)]
            suf2 = [bench_3090.build_ids(tokenizer, sess_slen, salt=sbase + s * 3 + 2) for s in range(n_sess)]

            # Turn 1: cold fill all N sessions in order 0..N-1
            t1_ttfts = []
            for s in range(n_sess):
                r1 = bench_3090.run_once(llm, SamplingParams, prefixes[s] + suf1[s], max_tokens=1)
                t1_ttfts.append(float(r1["ttft_s"]))

            # Turn 2: revisit all N sessions in round-robin order 0..N-1
            t2_ttfts = []
            t2_cached = []
            for s in range(n_sess):
                r2 = bench_3090.run_once(llm, SamplingParams, prefixes[s] + suf2[s], max_tokens=1)
                t2_ttfts.append(float(r2["ttft_s"]))
                t2_cached.append(int(r2["cached"]) if str(r2["cached"]).isdigit() else 0)

            tot_pref = n_sess * sess_plen
            tot_cached = sum(t2_cached)
            hit_pct = (tot_cached / tot_pref) * 100.0
            ws_gib = (n_sess * (sess_plen + sess_slen) * kv_bytes_per_tok) / (1024**3)
            row = {
                "num_sessions": n_sess,
                "prefix_per_session": sess_plen,
                "suffix_per_session": sess_slen,
                "working_set_tokens": n_sess * (sess_plen + sess_slen),
                "working_set_kv_gib": round(ws_gib, 3),
                "turn1_cold_p50_ms": round(statistics.median(t1_ttfts) * 1e3, 2),
                "turn2_p50_ms": round(statistics.median(t2_ttfts) * 1e3, 2),
                "turn2_mean_ms": round(statistics.mean(t2_ttfts) * 1e3, 2),
                "turn2_max_ms": round(max(t2_ttfts) * 1e3, 2),
                "turn2_hit_rate_pct": round(hit_pct, 2),
                "turn2_cached_tokens": tot_cached,
                "turn2_prefix_tokens": tot_pref,
            }
            wall_rows.append(row)
            print(
                f"  N={n_sess:2d} sess ({ws_gib:5.2f} GiB KV) | "
                f"Turn1 P50={row['turn1_cold_p50_ms']:7.1f}ms | "
                f"Turn2 P50={row['turn2_p50_ms']:7.1f}ms (mean {row['turn2_mean_ms']:7.1f}ms) | "
                f"Hit={row['turn2_hit_rate_pct']:6.2f}%",
                flush=True,
            )

        with OUT_WALL.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(wall_rows[0].keys()))
            w.writeheader()
            w.writerows(wall_rows)
        print(f"Saved {OUT_WALL} ({len(wall_rows)} concurrency points) | Total elapsed {time.perf_counter()-t_all:.1f}s", flush=True)
    finally:
        bench_3090.shutdown_llm(llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
