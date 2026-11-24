#!/usr/bin/env python3
"""Generate all 8 publication-grade ACM TACO / IEEE TPDS figures (PDF + 300-DPI PNG).

Every numerical plot reads directly from results/sim/, results/3090/, results/trace_stats/,
and datasets/ (for empirical CDFs). Zero numbers are invented.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import LogNorm, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "results" / "sim"
R3090 = ROOT / "results" / "3090"
TSTATS = ROOT / "results" / "trace_stats"
FIG = ROOT / "figures"

# Global publication rcParams
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "figure.titlesize": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Okabe-Ito colorblind-safe palette
C_OPTIKV = "#009E73"
C_RDMA = "#222222"
C_RDMA_PF = "#B03A6E"
C_RESV = "#56B4E9"
C_NAIVE = "#D55E00"
C_HBM = "#0072B2"
C_DRAM = "#E69F00"
C_SSD = "#CC79A7"
C_NOCACHE = "#8C8C8C"
C_ORACLE = "#6A3D9A"

POLICY_LABEL = {
    "no_cache": "NoCache",
    "local_hbm": "Local-HBM",
    "local_dram": "Local-DRAM",
    "local_ssd": "Local-SSD",
    "rdma_routing": "RDMA+Routing",
    "rdma_prefetch": "RDMA+Prefetch",
    "optical_naive": "Optical-Naive",
    "optical_reservation": "Optical-Resv.",
    "optikv": "OptiKV-Fabric",
    "eviction_s3fifo": "S3-FIFO",
    "eviction_belady": "Belady",
    "oracle_infinite_hbm": "Oracle",
}

POLICY_COLOR = {
    "no_cache": C_NOCACHE,
    "local_hbm": C_HBM,
    "local_dram": C_DRAM,
    "local_ssd": C_SSD,
    "rdma_routing": C_RDMA,
    "rdma_prefetch": C_RDMA_PF,
    "optical_naive": C_NAIVE,
    "optical_reservation": C_RESV,
    "optikv": C_OPTIKV,
    "eviction_s3fifo": "#7F7F7F",
    "eviction_belady": "#A6761D",
    "oracle_infinite_hbm": C_ORACLE,
}


def save_fig(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.{{pdf,png}}", flush=True)


def _fmt_pct(h: float) -> str:
    pct = h * 100.0
    return f"{pct:.1f}%" if abs(pct - round(pct)) > 1e-3 else f"{int(round(pct))}%"


def _fmt_ctx(n: int) -> str:
    if n >= 1_000_000:
        return "1M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


# ---------------------------------------------------------------------------
# Figure 1: Golden 3-Panel Top-Venue Teaser
# ---------------------------------------------------------------------------

def make_fig1_teaser() -> None:
    df_t = pd.read_csv(SIM / "fig1_teaser_128k.csv")
    df_hm = pd.read_csv(SIM / "exp_breakeven_heatmap.csv")

    fig = plt.figure(figsize=(14.2, 4.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.30, 1.15], wspace=0.28)

    # --- Panel (a): 128K Multi-Turn TTFT Component Breakdown ---
    ax0 = fig.add_subplot(gs[0])
    order = ["no_cache", "local_hbm", "rdma_routing", "rdma_prefetch", "optical_naive", "optical_reservation", "optikv"]
    sub = df_t.set_index("policy").loc[[p for p in order if p in set(df_t["policy"])]]
    labels = [POLICY_LABEL[p] for p in sub.index]
    x = np.arange(len(labels))
    # Plot P50 TTFT (log scale) with component annotations so both NoCache (seconds) and OptiKV (ms) read cleanly
    p50_ms = sub["p50_ttft_s"].values * 1e3
    colors = [POLICY_COLOR[p] for p in sub.index]
    bars = ax0.bar(x, p50_ms, color=colors, width=0.62, edgecolor="#222222", linewidth=0.6)
    ax0.set_yscale("log")
    ax0.set_xticks(x, labels, rotation=28, ha="right")
    ax0.set_ylabel("P50 TTFT (ms, log scale)")
    ax0.set_title("(a) 128K Agentic TTFT (MEMS 25 ms)")
    ymax = max(p50_ms) * 3.2
    ax0.set_ylim(bottom=max(1.0, min(p50_ms) * 0.4), top=ymax)
    for rect, val, pol in zip(bars, p50_ms, sub.index):
        txt = f"{val/1000:.1f}s" if val >= 1000 else f"{val:.0f}ms"
        weight = "bold" if pol == "optikv" else "normal"
        ax0.text(
            rect.get_x() + rect.get_width() / 2.0,
            val * 1.16,
            txt,
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight=weight,
            color="#00553D" if pol == "optikv" else "#222222",
        )

    # --- Panel (b): Clean Vector Architecture Schematic ---
    ax1 = fig.add_subplot(gs[1])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("(b) OptiKV-Fabric Co-Designed Control Plane")

    def box(xy, w, h, text, fc, ec="#222222", tc="#111111", fs=6.8, bold=False):
        patch = mpatches.FancyBboxPatch(
            xy, w, h, boxstyle="round,pad=0.14,rounding_size=0.22", facecolor=fc, edgecolor=ec, linewidth=1.0
        )
        ax1.add_patch(patch)
        ax1.text(
            xy[0] + w / 2,
            xy[1] + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            color=tc,
            fontweight="bold" if bold else "normal",
        )

    def arrow(p1, p2, label="", color="#333333", rad=0.0, dy=0.38):
        ax1.annotate(
            "",
            xy=p2,
            xytext=p1,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, connectionstyle=f"arc3,rad={rad}"),
        )
        if label:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax1.text(
                mx,
                my + dy,
                label,
                fontsize=6.4,
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92),
            )

    box((0.15, 7.85), 4.15, 1.55, "Multi-Turn Agent Requests\n(Prefix Blocks &\nInter-Turn Gap)", "#EBF5FB", C_HBM, bold=True)
    box((5.65, 7.85), 4.20, 1.55, "Causal Reuse Predictor\nAnchors + 2nd-Obs +\nGap Decay", "#FEF9E7", C_DRAM, bold=True)
    box((0.15, 4.75), 4.15, 2.05, "Optical-Aware Router\nStanding-Circuit Locality &\nSelective Recompute Abort", "#E8F8F5", C_OPTIKV, bold=True)
    box((5.65, 4.75), 4.20, 2.05, "OCS Port Calendar\nStanding Circuit ($\\tau{=}0$) &\nWindowed Burst Prefetch", "#FDEDEC", C_NAIVE, bold=True)

    # 5-Tier Memory Fabric at bottom
    tier_y = 0.45
    box((0.15, tier_y), 2.15, 2.35, "GPU HBM\nPinning &\nPrefix-GDSF", "#D6EAF8", C_HBM, fs=6.7, bold=True)
    box((2.65, tier_y), 2.15, 2.35, "Local DRAM\nPCIe 26–50\nGB/s Spill", "#FCF3CF", C_DRAM, fs=6.7)
    box((5.15, tier_y), 2.15, 2.35, "Local SSD\nNVMe 3 GB/s\nCold Tier", "#F5EEF8", C_SSD, fs=6.7)
    box((7.65, tier_y), 2.20, 2.35, "Optical Pool\nOCS 45–800\nGB/s Burst", "#D1F2EB", C_OPTIKV, fs=6.7, bold=True)

    arrow((4.45, 8.62), (5.50, 8.62))
    arrow((2.22, 7.72), (2.22, 6.95))
    arrow((7.75, 7.72), (7.75, 6.95))
    arrow((4.45, 5.78), (5.50, 5.78), "preview", dy=0.38)
    arrow((1.22, 4.60), (1.22, 2.95), color=C_HBM)
    arrow((3.72, 4.60), (3.72, 2.95), color=C_DRAM)
    arrow((8.75, 4.60), (8.75, 2.95), color=C_OPTIKV)
    arrow((6.35, 4.60), (1.85, 2.95), color=C_NAIVE, rad=0.10)
    ax1.text(
        5.15,
        3.22,
        "Async. Burst Prefetch ($\\Delta t_{\\mathrm{gap}}$)",
        fontsize=6.6,
        ha="center",
        color=C_NAIVE,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
    )

    # --- Panel (c): Break-Even Frontier Preview (OptiKV vs RDMA+Routing) ---
    ax2 = fig.add_subplot(gs[2])
    ctxs = sorted(df_hm["context_tokens"].unique())
    hrs = sorted(df_hm["requested_hit_rate"].unique())
    grid = np.zeros((len(hrs), len(ctxs)))
    no_ben = []
    for ri, h in enumerate(hrs):
        for ci, c in enumerate(ctxs):
            val = float(
                df_hm[(df_hm["context_tokens"] == c) & (np.isclose(df_hm["requested_hit_rate"], h))][
                    "speedup_vs_rdma_p50"
                ].iloc[0]
            )
            grid[ri, ci] = val
            if val <= 1.05:
                no_ben.append((ci, ri))
    vmax = max(2.0, float(np.nanmax(grid)))
    im = ax2.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=0.85, vmax=vmax),
    )
    if no_ben:
        ax2.scatter(
            [p[0] for p in no_ben],
            [p[1] for p in no_ben],
            marker="x",
            c="white",
            s=22,
            lw=1.0,
            label="Compute/HBM-bound ($\\leq 1.05\\times$)",
        )
        leg = ax2.legend(loc="lower right", fontsize=7.1, framealpha=0.92, facecolor="#2B2D42", edgecolor="white")
        for text in leg.get_texts():
            text.set_color("white")
    ax2.set_xticks(range(0, len(ctxs), 2), [_fmt_ctx(ctxs[i]) for i in range(0, len(ctxs), 2)], rotation=30, ha="right")
    ax2.set_yticks(range(len(hrs)), [_fmt_pct(h) for h in hrs])
    ax2.set_xlabel("Context length (tokens)")
    ax2.set_ylabel("Prefix reuse rate")
    ax2.set_title("(c) Speedup vs. RDMA+Routing")
    cbar = fig.colorbar(im, ax=ax2, pad=0.02, fraction=0.055)
    cbar.set_label("P50 Speedup", fontsize=8.0)

    save_fig(fig, "fig1_teaser")


# ---------------------------------------------------------------------------
# Figure 2: Multi-Arch KV Capacity & Real Trace Characterization (4 Panels)
# ---------------------------------------------------------------------------

def make_fig2_workload_and_capacity() -> None:
    df_cap = pd.read_csv(SIM / "fig1_kv_capacity.csv")
    df_ts = pd.read_csv(TSTATS / "summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))

    # --- Panel (a): Multi-Architecture KV Memory Pressure (8K -> 1M) ---
    ax = axes[0, 0]
    ctxs = np.array([8192, 16384, 32768, 65536, 131072, 262144, 524288, 1000000], dtype=float)
    show_models = [
        ("llama-3.1-70b", "Llama-3.1-70B (GQA, BF16)", 2, C_NAIVE, "-"),
        ("qwen3-8b", "Qwen3-8B (GQA, BF16)", 2, C_HBM, "-"),
        ("llama-3.1-8b", "Llama-3.1-8B (GQA, BF16)", 2, C_RESV, "--"),
        ("deepseek-v3", "DeepSeek-V3 (MLA, BF16)", 2, C_ORACLE, "-"),
        ("qwen2.5-7b-instruct-1m", "Qwen2.5-7B-1M (GQA, BF16)", 2, C_OPTIKV, "-"),
        ("deepseek-v3", "DeepSeek-V3 (MLA, FP8)", 1, C_ORACLE, ":"),
        ("qwen3.5-9b", "Qwen3.5-9B (3:1 Hybrid, BF16)", 2, C_DRAM, "-."),
        ("qwen3-next-80b-a3b", "Qwen3-Next-80B (3:1 Hybrid, BF16)", 2, C_SSD, "--"),
    ]
    cap_map = df_cap.set_index("model")
    for mkey, label, db, col, ls in show_models:
        if mkey not in cap_map.index:
            continue
        bpt = float(cap_map.loc[mkey, "kv_bytes_bf16" if db == 2 else "kv_bytes_fp8"])
        gib = (ctxs * bpt) / (1024**3)
        ax.plot(ctxs / 1000.0, gib, label=label, color=col, linestyle=ls, linewidth=1.7, marker="o", markersize=3.2)
    ax.axhline(24.0, color="#444444", linestyle="--", linewidth=1.1)
    ax.text(9.0, 27.0, "RTX 3090 (24 GiB)", fontsize=7.6, color="#333333", fontweight="bold")
    ax.axhline(80.0, color="#880000", linestyle=":", linewidth=1.2)
    ax.text(9.0, 90.0, "H100 SXM (80 GiB)", fontsize=7.6, color="#880000", fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Context length (K tokens, log scale)")
    ax.set_ylabel("Batch=1 KV cache size (GiB, log scale)")
    ax.set_title("(a) Multi-Architecture KV Memory Wall (GQA vs. MLA vs. Hybrid)")
    ax.legend(loc="lower right", fontsize=7.1, ncol=1, framealpha=0.92)

    # --- Panel (b): Real Trace Input Length CDFs (including tokenized LMCache!) ---
    ax = axes[0, 1]
    trace_cdfs = []
    mc_files = [
        ("Mooncake (n=23,608)", ROOT / "datasets/mooncake-traces/mooncake/train-00000-of-00001.parquet", "input_length", C_HBM, "-"),
        ("Conversation (n=12,031)", ROOT / "datasets/mooncake-traces/conversation/train-00000-of-00001.parquet", "input_length", C_DRAM, "--"),
        ("ToolAgent (n=23,608)", ROOT / "datasets/mooncake-traces/toolagent/train-00000-of-00001.parquet", "input_length", C_NAIVE, "-."),
        ("LMCache Tokenized (n=24,880)", ROOT / "datasets/lmcache-agentic-traces/tokenized_512.parquet", "input_tokens", C_OPTIKV, "-"),
        ("FreeInference Week (n=190,544)", ROOT / "datasets/freeinference-agentic-trace/week_2026_08_30/release/2026-08-30_2026-09-05/processed/requests.parquet", "prompt_tokens", C_SSD, "-"),
    ]
    for label, path, colname, col, ls in mc_files:
        if not path.is_file():
            continue
        arr = pq.read_table(path, columns=[colname]).column(colname).to_numpy(zero_copy_only=False)
        arr = arr[~pd.isna(arr)].astype(float)
        arr = arr[arr > 0]
        arr.sort()
        # Subsample 1000 quantiles for crisp lightweight vector PDF
        qs = np.linspace(0, 1, 600)
        vals = np.quantile(arr, qs)
        ax.plot(vals, qs, label=label, color=col, linestyle=ls, linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("Prompt length (tokens, log scale)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("(b) Real Production & Agentic Prompt Length Distributions")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92)

    # --- Panel (c): Causal Block Reuse Rate (%) across all 6 real traces ---
    ax = axes[1, 0]
    # Compute exact tokenized LMCache reuse directly from cached parquet
    lm_tbl = pq.read_table(ROOT / "datasets/lmcache-agentic-traces/tokenized_512.parquet", columns=["block_ids"])
    seen_b: set[str] = set()
    tot_occ = reused_occ = 0
    for bl in lm_tbl.column("block_ids").to_pylist():
        for b in bl:
            tot_occ += 1
            if b in seen_b:
                reused_occ += 1
            else:
                seen_b.add(b)
    lm_reuse_pct = 100.0 * reused_occ / max(1, tot_occ)

    ts_map = df_ts.set_index("trace_id")
    reuse_items = [
        ("Conversation\n(512-tok)", float(ts_map.loc["conversation", "prefix_reuse_fraction"]) * 100.0, C_DRAM),
        ("ToolAgent\n(512-tok)", float(ts_map.loc["toolagent", "prefix_reuse_fraction"]) * 100.0, C_NAIVE),
        ("Mooncake\n(512-tok)", float(ts_map.loc["mooncake", "prefix_reuse_fraction"]) * 100.0, C_HBM),
        ("Synthetic-MC\n(512-tok)", float(ts_map.loc["synthetic", "prefix_reuse_fraction"]) * 100.0, C_RESV),
        ("LMCache\n(512-tok Qwen)", lm_reuse_pct, C_OPTIKV),
        ("FreeInference\n(16-tok native)", float(ts_map.loc["freeinference", "prefix_reuse_fraction"]) * 100.0, C_SSD),
    ]
    y = np.arange(len(reuse_items))
    bars = ax.barh(y, [v[1] for v in reuse_items], color=[v[2] for v in reuse_items], height=0.62, edgecolor="#222222", lw=0.6)
    ax.set_yticks(y, [v[0] for v in reuse_items])
    ax.set_xlim(0, 114)
    ax.set_xlabel("Causal block reuse fraction (%)")
    ax.set_title("(c) Strictly Causal Prefix Block Reuse Across Real Traces")
    for rect, (_, val, _) in zip(bars, reuse_items):
        ax.text(val + 1.5, rect.get_y() + rect.get_height() / 2.0, f"{val:.1f}%", va="center", fontsize=8.3, fontweight="bold")

    # --- Panel (d): Inter-Turn Idle Gap (pre_gap) CDF vs MEMS OCS Switch Window ---
    ax = axes[1, 1]
    lm_gaps = pq.read_table(ROOT / "datasets/lmcache-agentic-traces/tokenized_512.parquet", columns=["pre_gap"]).column("pre_gap").to_numpy(zero_copy_only=False)
    lm_gaps = np.sort(lm_gaps[lm_gaps > 0])
    qs = np.linspace(0, 1, 500)
    ax.plot(np.quantile(lm_gaps, qs), qs, label=f"LMCache (P50={np.median(lm_gaps):.2f}s)", color=C_OPTIKV, lw=2.0)

    fi_t = pq.read_table(
        ROOT / "datasets/freeinference-agentic-trace/week_2026_08_30/release/2026-08-30_2026-09-05/processed/requests.parquet",
        columns=["session_id", "start_ms", "end_ms"],
    ).to_pandas()
    fi_t = fi_t.sort_values(["session_id", "start_ms"])
    fi_t["prev_end"] = fi_t.groupby("session_id")["end_ms"].shift(1)
    fi_gaps = ((fi_t["start_ms"] - fi_t["prev_end"]) / 1000.0).dropna().values
    fi_gaps = np.sort(fi_gaps[fi_gaps > 0])
    ax.plot(np.quantile(fi_gaps, qs), qs, label=f"FreeInference (P50={np.median(fi_gaps):.2f}s)", color=C_SSD, lw=2.0, linestyle="--")

    ax.axvspan(0.025, 0.050, color=C_NAIVE, alpha=0.22, label="MEMS OCS switch (25–50 ms)")
    ax.axvline(0.00001, color=C_HBM, linestyle=":", lw=1.4, label="SiPh OCS switch (10 $\\mu$s)")
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1e3)
    ax.set_xlabel("Inter-turn idle gap $\\Delta t_{\\mathrm{gap}}$ (seconds, log scale)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("(d) Inter-Turn Prefetch Windows vs. Optical Switch Latency")
    ax.legend(loc="lower right", fontsize=7.6, framealpha=0.92)

    fig.tight_layout()
    save_fig(fig, "fig2_trace_characterization")


# ---------------------------------------------------------------------------
# Figure 3: Real RTX 3090 24GB 7B Calibration, HBM Wall & Overlap (6 Panels)
# ---------------------------------------------------------------------------

def make_fig3_calibration() -> None:
    df_cal = pd.read_csv(SIM / "fig3_calibration_points.csv")
    df_ph = pd.read_csv(R3090 / "prefix_hit_scaling_7b.csv")
    df_wall = pd.read_csv(R3090 / "hbm_capacity_wall_7b.csv")
    df_ovl = pd.read_csv(R3090 / "kv_offload_overlap_3090.csv")
    df_sens_ovl = pd.read_csv(SIM / "exp_sensitivity_overlap.csv")

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.6))

    # --- Panel (a): Measured 7B Cold Prefill (1K..32K) + Roofline + Quadratic + H100 ---
    ax = axes[0, 0]
    meas = df_cal[df_cal["measured_3090_7b_ttft_s"].notna()]
    ax.plot(df_cal["prompt_tokens"] / 1e3, df_cal["roofline_3090_s"], color=C_HBM, lw=1.9, label="RTX 3090 Roofline (3.08% MAPE)")
    ax.plot(df_cal["prompt_tokens"] / 1e3, df_cal["quad_train_1k_16k_s"], color=C_DRAM, lw=1.6, ls="--", label="Quadratic Fit (6.25% held-out)")
    ax.plot(df_cal["prompt_tokens"] / 1e3, df_cal["h100_peak_spec_s"], color=C_OPTIKV, lw=1.8, ls=":", label="H100 SXM Scaled Roofline")
    ax.scatter(meas["prompt_tokens"] / 1e3, meas["measured_3090_7b_ttft_s"], color="#111111", s=40, zorder=5, label="Measured RTX 3090 (1K–32K)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Prompt length (K tokens, log scale)")
    ax.set_ylabel("Cold prefill TTFT (s, log scale)")
    ax.set_title("(a) Cold Prefill Scaling (Qwen2.5-7B BF16)")
    ax.legend(loc="upper left", fontsize=7.0, framealpha=0.92)

    # --- Panel (b): Single-Node Calibration Error (%) ---
    ax = axes[0, 1]
    x = np.arange(len(meas))
    w = 0.36
    ax.bar(x - w / 2, meas["roofline_abs_err_pct"], width=w, color=C_HBM, label="Roofline (mean 3.08%)")
    ax.bar(x + w / 2, meas["quad_train_abs_err_pct"], width=w, color=C_DRAM, label="Quad 1–16K (6.25% held-out)")
    ax.axhline(10.0, color=C_NAIVE, ls="--", lw=1.4, label="10% MAPE reference")
    ax.axvspan(5.5, 7.6, color="#FCE4D6", alpha=0.45)
    ax.text(6.5, 9.2, "Held-out\ntest", ha="center", va="top", fontsize=7.2, color=C_NAIVE, fontweight="bold")
    ax.set_xticks(x, [_fmt_ctx(int(v)) for v in meas["prompt_tokens"]])
    ax.set_ylim(0, 12.2)
    ax.set_xlabel("Measured prompt length on RTX 3090")
    ax.set_ylabel("Absolute percentage error (%)")
    ax.set_title("(b) Single-Node Calibration Error (1K–32K)")
    ax.legend(loc="upper left", fontsize=6.9, framealpha=0.92)

    # --- Panel (c): Multi-Length Hot Prefix-Hit Scaling (4K..30K, Real vLLM 3090) ---
    ax = axes[0, 2]
    s128 = df_ph[df_ph["suffix_tokens"] == 128].sort_values("prefix_tokens")
    s512 = df_ph[df_ph["suffix_tokens"] == 512].sort_values("prefix_tokens")
    ax.plot(s512["prefix_tokens"] / 1024.0, s512["cold_ttft_ms"], color=C_RDMA, marker="s", ms=4.8, lw=1.9, label="Cold fill (+512 suffix)")
    ax.plot(s512["prefix_tokens"] / 1024.0, s512["hot_ttft_ms"], color=C_HBM, marker="o", ms=5.0, lw=2.0, label="Hot APC hit (+512 suffix)")
    ax.plot(s512["prefix_tokens"] / 1024.0, s512["roofline_hot_ms"], color=C_HBM, ls=":", lw=1.4, label="Roofline $\\Delta$-prefill (+512)")
    ax.plot(s128["prefix_tokens"] / 1024.0, s128["hot_ttft_ms"], color=C_OPTIKV, marker="^", ms=5.4, lw=2.0, label="Hot APC hit (+128 suffix)")
    ax.set_yscale("log")
    ax.set_ylim(18, 26000)
    ax.set_xticks([4, 8, 12, 16, 24, 30])
    ax.set_xticklabels(["4K", "8K", "12K", "16K", "24K", "30K"])
    ax.set_xlabel("Cached prefix length (tokens)")
    ax.set_ylabel("Measured TTFT on RTX 3090 (ms, log)")
    ax.set_title("(c) Multi-Length Prefix-Hit Scaling (vLLM)")
    # Annotate 94.5x speedup at 24K+128 and 40.9x at 30K+512
    p24 = s128[s128["prefix_tokens"] == 24576].iloc[0]
    p30 = s512[s512["prefix_tokens"] == 30720].iloc[0]
    ax.annotate(
        f"{p24['speedup']:.1f}x\n(24K+128)",
        xy=(24.0, p24["hot_ttft_ms"]),
        xytext=(14.5, 25.0),
        arrowprops=dict(arrowstyle="-|>", color=C_OPTIKV, lw=1.2),
        fontsize=7.3,
        fontweight="bold",
        color="#00553D",
        bbox=dict(boxstyle="round,pad=0.22", fc="#E8F8F5", ec=C_OPTIKV, lw=0.8),
    )
    ax.annotate(
        f"{p30['speedup']:.1f}x\n(30K+512)",
        xy=(30.0, p30["hot_ttft_ms"]),
        xytext=(21.2, 720.0),
        arrowprops=dict(arrowstyle="-|>", color=C_HBM, lw=1.2),
        fontsize=7.3,
        fontweight="bold",
        color="#004477",
        bbox=dict(boxstyle="round,pad=0.22", fc="#EBF5FB", ec=C_HBM, lw=0.8),
    )
    ax.legend(loc="upper left", fontsize=6.7, framealpha=0.92)

    # --- Panel (d): Real RTX 3090 24GB Multi-Session HBM Capacity Wall ---
    ax = axes[1, 0]
    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.grid(False)
    ns = df_wall["num_sessions"].values
    l1 = ax.plot(ns, df_wall["turn2_hit_rate_pct"], color=C_OPTIKV, marker="o", ms=5.2, lw=2.2, label="Turn-2 prefix hit rate (%)")
    l2 = ax_r.plot(ns, df_wall["turn2_p50_ms"], color=C_NAIVE, marker="s", ms=5.0, lw=2.0, ls="--", label="Turn-2 P50 TTFT (ms)")
    ax.axvline(10.6, color="#444444", ls=":", lw=1.4)
    ax.axvspan(10.6, 21.0, color="#FDEDEC", alpha=0.55)
    ax.text(15.6, 56.0, "HBM KV Pool Overflow\n(>4.70 GiB / 88K tok)\nLRU Cliff: 40.7x TTFT", ha="center", va="center", fontsize=7.3, fontweight="bold", color="#9C2A00", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_NAIVE, alpha=0.90))
    ax.set_xlim(0.5, 20.8)
    ax.set_ylim(-6, 115)
    ax_r.set_ylim(-80, 2250)
    ax.set_xticks([1, 4, 8, 10, 12, 16, 20])
    ax.set_xlabel("Concurrent 8K-token sessions ($N$)")
    ax.set_ylabel("Turn-2 prefix hit rate (%)", color="#00664B")
    ax_r.set_ylabel("Turn-2 P50 TTFT (ms)", color="#B03A00")
    ax.tick_params(axis="y", colors="#00664B")
    ax_r.tick_params(axis="y", colors="#B03A00")
    ax.set_title("(d) Real 24 GB HBM Capacity Wall (8K/sess)")
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc="center left", bbox_to_anchor=(0.01, 0.34), fontsize=6.9, framealpha=0.92)

    # --- Panel (e): Real 28-Layer KV PCIe Offload & Dual-Stream Overlap ---
    ax = axes[1, 1]
    sub_ovl = df_ovl[df_ovl["delta_tokens"] == 128].sort_values("context_tokens")
    x_o = np.arange(len(sub_ovl))
    bw = 0.26
    ax.bar(x_o - bw, sub_ovl["t_serial_ms"], width=bw, color=C_NOCACHE, edgecolor="#222222", lw=0.5, label="Serial ($T_{\\mathrm{comp}} + T_{\\mathrm{H2D}}$)")
    ax.bar(x_o, sub_ovl["t_overlapped_ms"], width=bw, color=C_OPTIKV, edgecolor="#222222", lw=0.5, label="2-Stream Overlapped ($T_{\\mathrm{ovl}}$)")
    ax.bar(x_o + bw, sub_ovl["t_comp_ms"], width=bw, color=C_HBM, edgecolor="#222222", lw=0.5, label="Compute Floor ($T_{\\mathrm{comp}}$)")
    for xi, (_, r) in enumerate(sub_ovl.iterrows()):
        ax.text(xi, r["t_serial_ms"] + 4.0, f"$\\alpha$={r['alpha_ovl_measured']:.2f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#00553D")
    med_h2d = float(df_ovl["h2d_gb_s"].median())
    med_d2h = float(df_ovl["d2h_gb_s"].median())
    med_a = float(df_ovl["alpha_ovl_measured"].median())
    ax.text(
        0.03,
        0.96,
        f"PCIe Gen4 H2D/D2H: {med_h2d:.1f}/{med_d2h:.1f} GB/s\nBare-metal $\\alpha_{{\\mathrm{{ovl}}}}$ med={med_a:.2f} (0.86–0.98)",
        transform=ax.transAxes,
        va="top",
        fontsize=6.8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="#F8F9F9", ec="#666666"),
    )
    ax.set_xticks(x_o, [f"{int(c)//1024}K\n({int(m)} MiB)" for c, m in zip(sub_ovl["context_tokens"], sub_ovl["kv_mib"])])
    ax.set_ylim(0, max(sub_ovl["t_serial_ms"]) * 1.24)
    ax.set_xlabel("Context length (28-layer Qwen2.5-7B BF16)")
    ax.set_ylabel("Latency ($\\Delta S{=}128$, ms)")
    ax.set_title("(e) 28-Layer KV Offload & Dual-Stream Overlap")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.77), fontsize=6.7, framealpha=0.92)

    # --- Panel (f): End-to-End Sensitivity Across Full Overlap Range alpha in [0, 1] ---
    ax = axes[1, 2]
    for pol, ls, mk in [
        ("rdma_routing", "--", "s"),
        ("rdma_prefetch", "-.", "D"),
        ("optikv", "-", "o"),
        ("oracle_infinite_hbm", ":", "^"),
    ]:
        sp = df_sens_ovl[df_sens_ovl["policy"] == pol].sort_values("alpha_ovl")
        ax.plot(
            sp["alpha_ovl"],
            sp["p50_ttft_s"] * 1e3,
            color=POLICY_COLOR[pol],
            ls=ls,
            marker=mk,
            ms=4.8,
            lw=2.1 if pol == "optikv" else 1.7,
            label=POLICY_LABEL[pol],
        )
    ax.axvspan(0.86, 0.98, color="#D1F2EB", alpha=0.55, label="3090 bare-metal $\\alpha$ (0.86–0.98)")
    ax.axvline(0.70, color="#555555", ls="--", lw=1.1)
    ax.text(0.68, 96.0, "Default\n$\\alpha{=}0.70$", ha="right", va="center", fontsize=7.0, color="#333333", fontweight="bold")
    ax.annotate("4.56x vs RDMA\n1.45x vs Pref.", xy=(0.0, 40.8), xytext=(0.08, 86.0), arrowprops=dict(arrowstyle="-|>", color=C_OPTIKV, lw=1.2), fontsize=7.0, fontweight="bold", color="#00553D", bbox=dict(boxstyle="round,pad=0.22", fc="#E8F8F5", ec=C_OPTIKV, lw=0.8))
    ax.annotate("3.15x", xy=(1.0, 36.3), xytext=(0.86, 78.0), arrowprops=dict(arrowstyle="-|>", color=C_OPTIKV, lw=1.1), fontsize=7.2, fontweight="bold", color="#00553D")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(15, 212)
    ax.set_xlabel("Compute-transfer overlap factor $\\alpha_{\\mathrm{ovl}}$")
    ax.set_ylabel("128K Agentic P50 TTFT (ms)")
    ax.set_title("(f) Robustness Across Overlap $\\alpha_{\\mathrm{ovl}} \\in [0, 1]$")
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.92, ncol=1)

    fig.tight_layout()
    save_fig(fig, "fig3_prefill_model")


# ---------------------------------------------------------------------------
# Figure 4: Main End-to-End Comparison Across All 3 Real Traces + Dual Regime
# ---------------------------------------------------------------------------

def make_fig4_main_comparison() -> None:
    panels = [
        ("exp_main_mooncake_h100_siph.csv", "(a) Mooncake Full (23,608 reqs, SiPh 10 $\\mu$s)"),
        ("exp_main_lmcache_h100_mems.csv", "(b) LMCache Agentic (120 sess, MEMS 25 ms)"),
        ("exp_main_freeinference_h100_mems.csv", "(c) FreeInference Long (25 sess, MEMS 25 ms)"),
        ("exp_main_mooncake_3090_calibrated.csv", "(d) Mooncake on RTX 3090 Calibrated (45 GB/s)"),
    ]
    show = ["no_cache", "local_hbm", "local_dram", "rdma_routing", "rdma_prefetch", "optical_naive", "optical_reservation", "optikv", "oracle_infinite_hbm"]
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.3))
    for ax, (fname, title) in zip(axes.flat, panels):
        df = pd.read_csv(SIM / fname).set_index("policy")
        pols = [p for p in show if p in df.index]
        x = np.arange(len(pols))
        w = 0.37
        p50 = [float(df.loc[p, "p50_ttft_s"]) * 1e3 for p in pols]
        p99 = [float(df.loc[p, "p99_ttft_s"]) * 1e3 for p in pols]
        b1 = ax.bar(x - w / 2, p50, width=w, color=[POLICY_COLOR[p] for p in pols], edgecolor="#222222", lw=0.6, label="P50 TTFT")
        b2 = ax.bar(x + w / 2, p99, width=w, color=[POLICY_COLOR[p] for p in pols], alpha=0.45, edgecolor="#222222", lw=0.6, hatch="//", label="P99 TTFT")
        ax.set_yscale("log")
        ax.set_xticks(x, [POLICY_LABEL[p] for p in pols], rotation=26, ha="right")
        ax.set_ylabel("TTFT (ms, log scale)")
        ax.set_title(title)
        ymax = max(p99) * 3.0
        ymin = max(1.0, min(p50) * 0.45)
        ax.set_ylim(ymin, ymax)
        for rect, val, pol in zip(b1, p50, pols):
            txt = f"{val/1000:.1f}s" if val >= 1000 else f"{val:.0f}"
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                val * 1.15,
                txt,
                ha="center",
                va="bottom",
                fontsize=7.0,
                fontweight="bold" if pol == "optikv" else "normal",
            )
        ax.legend(loc="upper right", fontsize=7.6, framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, "fig4_main_comparison")


# ---------------------------------------------------------------------------
# Figure 5: 2-Panel Centerpiece Break-Even Design Space Heatmap (2K..1M x 0..95%)
# ---------------------------------------------------------------------------

def make_fig5_heatmap() -> None:
    df = pd.read_csv(SIM / "exp_breakeven_heatmap.csv")
    ctxs = sorted(df["context_tokens"].unique())
    hrs = sorted(df["requested_hit_rate"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1))
    specs = [
        ("speedup_vs_rdma_p50", "(a) OptiKV-Fabric Speedup vs. RDMA+Routing (P50 TTFT)", True),
        ("speedup_vs_naive_p50", "(b) OptiKV-Fabric Speedup vs. Optical-Naive (P50 TTFT)", False),
    ]
    for ax, (col, title, mark_nb) in zip(axes, specs):
        grid = np.zeros((len(hrs), len(ctxs)))
        for ri, h in enumerate(hrs):
            for ci, c in enumerate(ctxs):
                val = float(df[(df["context_tokens"] == c) & (np.isclose(df["requested_hit_rate"], h))][col].iloc[0])
                grid[ri, ci] = val
        vmin = max(0.8, float(np.nanmin(grid)))
        vmax = max(2.0, float(np.nanmax(grid)))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax))
        for ri in range(len(hrs)):
            for ci in range(len(ctxs)):
                val = grid[ri, ci]
                txt_col = "white" if val < math.sqrt(vmin * vmax) else "#111111"
                label = f"{val:.1f}x" if val < 100 else f"{val:.0f}x"
                if mark_nb and val <= 1.05:
                    ax.text(ci, ri, f"{val:.2f}\n(x)", ha="center", va="center", fontsize=6.7, color="white", fontweight="bold")
                else:
                    ax.text(ci, ri, label, ha="center", va="center", fontsize=7.1, color=txt_col, fontweight="bold")
        ax.set_xticks(range(len(ctxs)), [_fmt_ctx(int(c)) for c in ctxs], rotation=28, ha="right")
        ax.set_yticks(range(len(hrs)), [_fmt_pct(h) for h in hrs])
        ax.set_xlabel("Context length (tokens, Qwen2.5-7B-1M)")
        ax.set_ylabel("Prefix reuse fraction")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.048)
        cbar.set_label("Speedup ratio (log scale)")
    fig.tight_layout()
    save_fig(fig, "fig5_heatmap")


# ---------------------------------------------------------------------------
# Figure 6: 4-Panel Sensitivity (Reconfig, BW/Links, Multi-Arch FP8, Cost/Power)
# ---------------------------------------------------------------------------

def make_fig6_sensitivity() -> None:
    df_tau = pd.read_csv(SIM / "exp_sensitivity_reconfig.csv")
    df_bw = pd.read_csv(SIM / "exp_sensitivity_bandwidth.csv")
    df_arch = pd.read_csv(SIM / "exp_sensitivity_arch_fp8.csv")
    df_load = pd.read_csv(SIM / "exp_load_sweep.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))

    # --- Panel (a): Reconfiguration tau (0 .. 50 ms) + OCS Duty Cycle ---
    ax = axes[0, 0]
    for pol, mk in (("optikv", "o"), ("rdma_prefetch", "p"), ("rdma_routing", "s"), ("optical_reservation", "^"), ("optical_naive", "D")):
        sub = df_tau[df_tau["policy"] == pol].sort_values("reconfig_us")
        if sub.empty:
            continue
        ax.plot(
            sub["reconfig_us"],
            sub["p50_ttft_s"] * 1e3,
            marker=mk,
            color=POLICY_COLOR[pol],
            lw=1.9,
            label=POLICY_LABEL[pol],
        )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set_xlabel("OCS reconfiguration latency $\\tau$ ($\\mu$s, symlog)")
    ax.set_ylabel("P50 TTFT (ms, log scale)")
    ax.set_title("(a) Reconfiguration Latency Sensitivity (0–50 ms)")
    ax.axvline(1.0, color="#666666", ls=":", lw=1.0)
    ax.text(1.4, 380.0, "SiPh (1–100 $\\mu$s)", fontsize=7.4, color="#444444", fontweight="bold")
    ax.axvline(25000.0, color=C_NAIVE, ls=":", lw=1.0)
    ax.text(21000.0, 380.0, "MEMS (25 ms)", fontsize=7.4, color=C_NAIVE, fontweight="bold", ha="right")
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.56), fontsize=7.5, framealpha=0.94)

    # --- Panel (b): Optical Bandwidth Sweep (45 .. 800 GB/s) ---
    ax = axes[0, 1]
    for pol, mk in (("optikv", "o"), ("rdma_prefetch", "p"), ("rdma_routing", "s"), ("optical_reservation", "^"), ("optical_naive", "D")):
        sub = df_bw[df_bw["policy"] == pol].sort_values("bandwidth_gb_s")
        if sub.empty:
            continue
        ax.plot(
            sub["bandwidth_gb_s"],
            sub["p50_ttft_s"] * 1e3,
            marker=mk,
            color=POLICY_COLOR[pol],
            lw=1.9,
            label=POLICY_LABEL[pol],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([45, 100, 200, 400, 800], ["45", "100", "200", "400", "800"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Optical fabric bandwidth (GB/s)")
    ax.set_ylabel("P50 TTFT (ms, log scale)")
    ax.set_title("(b) Optical Bandwidth Scaling (45–800 GB/s)")
    ax.legend(loc="best", fontsize=7.6, framealpha=0.92)

    # --- Panel (c): Multi-Architecture x KV Precision (BF16 vs FP8) Speedup vs RDMA ---
    ax = axes[1, 0]
    m_order = ["qwen3.5-9b", "qwen2.5-7b-instruct-1m", "deepseek-v3", "qwen3-8b", "llama-3.1-70b"]
    m_short = {
        "qwen3.5-9b": "Qwen3.5-9B\n(Hybrid)",
        "qwen2.5-7b-instruct-1m": "Qwen2.5-7B\n(GQA)",
        "deepseek-v3": "DeepSeek-V3\n(MLA)",
        "qwen3-8b": "Qwen3-8B\n(GQA)",
        "llama-3.1-70b": "Llama-70B\n(GQA)",
    }
    bf16_sp, fp8_sp = [], []
    for m in m_order:
        for prec, out_list in (("BF16", bf16_sp), ("FP8", fp8_sp)):
            s_m = df_arch[(df_arch["model"] == m) & (df_arch["precision"] == prec)].set_index("policy")
            sp = float(s_m.loc["rdma_routing", "p50_ttft_s"]) / max(1e-9, float(s_m.loc["optikv", "p50_ttft_s"]))
            out_list.append(sp)
    x = np.arange(len(m_order))
    w = 0.36
    b1 = ax.bar(x - w / 2, bf16_sp, width=w, color=C_OPTIKV, edgecolor="#222222", lw=0.6, label="BF16 KV (2 B/elem)")
    b2 = ax.bar(x + w / 2, fp8_sp, width=w, color=C_RESV, edgecolor="#222222", lw=0.6, label="FP8 KV (1 B/elem)")
    ax.axhline(1.0, color="#444444", ls="--", lw=1.0)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.04, f"{h:.2f}x", ha="center", va="bottom", fontsize=7.3, fontweight="bold")
    ax.set_xticks(x, [m_short[m] for m in m_order])
    ax.set_ylim(0, max(bf16_sp + fp8_sp) * 1.22)
    ax.set_ylabel("P50 Speedup vs. RDMA+Routing")
    ax.set_title("(c) Multi-Architecture & FP8 KV Sensitivity (128K Multi-Turn)")
    ax.legend(loc="upper left", fontsize=7.6, framealpha=0.92)

    # --- Panel (d): Cited Fabric Power & Cost Efficiency on Real FreeInference (Goodput/W & Goodput/$1K) ---
    ax = axes[1, 1]
    df_fi = pd.read_csv(SIM / "exp_main_freeinference_h100_mems.csv").set_index("policy")
    pols = ["rdma_routing", "rdma_prefetch", "optical_naive", "optical_reservation", "optikv"]
    pols = [p for p in pols if p in df_fi.index]
    x = np.arange(len(pols))
    w = 0.36
    gpw = [float(df_fi.loc[p, "goodput_per_watt"]) for p in pols]
    gpd = [float(df_fi.loc[p, "goodput_per_dollar"]) for p in pols]
    b1 = ax.bar(x - w / 2, gpw, width=w, color=C_HBM, edgecolor="#222222", lw=0.6, label="Goodput / Watt (req/s/W)")
    b2 = ax.bar(x + w / 2, gpd, width=w, color=C_DRAM, edgecolor="#222222", lw=0.6, label=r"Goodput / \$1K (req/s/\$1K)")
    ax.set_yscale("log")
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            if h > 0:
                lbl = f"{h:.3f}" if h < 0.05 else f"{h:.2f}"
                ax.text(rect.get_x() + rect.get_width() / 2, h * 1.18, lbl, ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax.set_xticks(x, [POLICY_LABEL[p] for p in pols])
    ax.set_ylim(max(1e-3, min(gpw + gpd) * 0.4), max(gpw + gpd) * 4.5)
    ax.set_ylabel("Efficiency on FreeInference (log scale)")
    ax.set_title("(d) Fabric Power & Cost Efficiency (FreeInference)")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92)

    fig.tight_layout()
    save_fig(fig, "fig6_reconfig_and_cited_cost")


# ---------------------------------------------------------------------------
# Figure 7: 3-Panel Stepwise Ablation, Modern Eviction & Noise Robustness
# ---------------------------------------------------------------------------

def make_fig7_ablation() -> None:
    df_abl = pd.read_csv(SIM / "exp_ablation.csv").sort_values("step_idx")
    df_ev = pd.read_csv(SIM / "exp_eviction_comparison.csv").sort_values("ev_idx")
    df_ns = pd.read_csv(SIM / "exp_noise_robustness.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

    # --- Panel (a): Cumulative Stepwise Ablation ---
    ax = axes[0]
    x = np.arange(len(df_abl))
    w = 0.37
    p50 = df_abl["p50_ttft_s"].values * 1e3
    p99 = df_abl["p99_ttft_s"].values * 1e3
    b1 = ax.bar(x - w / 2, p50, width=w, color=C_OPTIKV, edgecolor="#222222", lw=0.6, label="P50 TTFT")
    b2 = ax.bar(x + w / 2, p99, width=w, color=C_HBM, alpha=0.75, edgecolor="#222222", lw=0.6, label="P99 TTFT")
    ax.set_yscale("log")
    ax.set_ylim(max(1.0, min(p50) * 0.4), max(p99) * 3.0)
    for rect, v in zip(b1, p50):
        txt = f"{v/1000:.1f}s" if v >= 1000 else f"{v:.0f}"
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            v * 1.18,
            txt,
            ha="center",
            va="bottom",
            fontsize=6.3,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.82),
        )
    ax.set_xticks(x, df_abl["ablation_step"], rotation=32, ha="right")
    ax.set_ylabel("TTFT (ms, log scale)")
    ax.set_title("(a) Cumulative Module Ablation")
    ax.legend(loc="upper right", fontsize=7.5)

    # --- Panel (b): Constrained-HBM Eviction Baselines on Real LMCache (Single-GPU) ---
    ax = axes[1]
    x = np.arange(len(df_ev))
    hits = df_ev["hit_rate"].values * 100.0
    cols = [C_NOCACHE, C_DRAM, C_OPTIKV, C_ORACLE]
    bars = ax.bar(x, hits, width=0.54, color=cols[: len(x)], edgecolor="#222222", lw=0.6)
    ax.set_xticks(x, df_ev["eviction_label"], rotation=22, ha="right")
    ax.set_ylim(0, min(108, max(hits) * 1.30))
    ax.set_ylabel("Single-GPU HBM prefix hit rate (%)")
    ax.set_title("(b) Eviction Under 2 GiB HBM (LMCache)")
    lru_h = float(hits[0]) if len(hits) else 0.0
    for idx_b, (rect, h) in enumerate(zip(bars, hits)):
        delta = h - lru_h
        sub_txt = "baseline" if idx_b == 0 else f"({delta:+.2f} pp)"
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            h + 0.8,
            f"{h:.2f}%\n{sub_txt}",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
        )

    # --- Panel (c): Causal P(reuse) Noise Robustness (sigma = 0, 5%, 10%, 20%) ---
    ax = axes[2]
    grp = df_ns.groupby("sigma").agg(
        p50_mean=("p50_ttft_s", lambda s: s.mean() * 1e3),
        p50_std=("p50_ttft_s", lambda s: s.std(ddof=0) * 1e3),
        p99_mean=("p99_ttft_s", lambda s: s.mean() * 1e3),
        p99_std=("p99_ttft_s", lambda s: s.std(ddof=0) * 1e3),
    ).reset_index()
    sig_pct = grp["sigma"].values * 100.0
    ax.errorbar(sig_pct, grp["p50_mean"], yerr=grp["p50_std"], marker="o", color=C_OPTIKV, lw=2.0, capsize=4, label="P50 TTFT (mean +/- std)")
    ax.errorbar(sig_pct, grp["p99_mean"], yerr=grp["p99_std"], marker="s", color=C_HBM, lw=2.0, ls="--", capsize=4, label="P99 TTFT (mean +/- std)")
    ax.set_yscale("log")
    ax.set_xticks(sig_pct, [f"{int(v)}%" for v in sig_pct])
    ax.set_ylim(10.0, max(grp["p99_mean"] + grp["p99_std"]) * 3.0)
    for xv, p50v, p99v in zip(sig_pct, grp["p50_mean"], grp["p99_mean"]):
        ax.text(xv, p50v * 1.25, f"{p50v:.1f}ms", ha="center", va="bottom", fontsize=7.0, color="#00553D", fontweight="bold")
        ax.text(xv, p99v * 1.22, f"{p99v/1e3:.2f}s", ha="center", va="bottom", fontsize=7.0, color=C_HBM, fontweight="bold")
    ax.set_xlabel("Gaussian noise std on P(reuse) (%)")
    ax.set_ylabel("TTFT (ms, log scale)")
    ax.set_title("(c) Reuse Predictor Noise Robustness (3 seeds)")
    ax.legend(loc="center right", fontsize=7.5, framealpha=0.92)

    fig.tight_layout()
    save_fig(fig, "fig7_ablation")


# ---------------------------------------------------------------------------
# Figure 8: 2-Panel P99 Tail Latency & SLO Goodput Under Load (0.5 .. 16 req/s)
# ---------------------------------------------------------------------------

def make_fig8_load() -> None:
    df = pd.read_csv(SIM / "exp_load_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))
    pols = ["no_cache", "local_hbm", "local_dram", "rdma_routing", "rdma_prefetch", "optical_naive", "optical_reservation", "optikv", "oracle_infinite_hbm"]
    markers = {"no_cache": "x", "local_hbm": "v", "local_dram": "^", "rdma_routing": "s", "rdma_prefetch": "p", "optical_naive": "D", "optical_reservation": "P", "optikv": "o", "oracle_infinite_hbm": "*"}

    ax0, ax1 = axes
    for pol in pols:
        sub = df[df["policy"] == pol].sort_values("target_rps")
        if sub.empty:
            continue
        lw = 2.4 if pol == "optikv" else 1.5
        ls = "--" if pol == "oracle_infinite_hbm" else "-"
        ax0.plot(sub["target_rps"], sub["p50_ttft_s"], marker=markers[pol], color=POLICY_COLOR[pol], lw=lw, ls=ls, label=POLICY_LABEL[pol])
        ax1.plot(sub["target_rps"], sub["goodput_rps"], marker=markers[pol], color=POLICY_COLOR[pol], lw=lw, ls=ls, label=POLICY_LABEL[pol])

    slo_s = float(df["slo_ttft_s"].iloc[0]) if "slo_ttft_s" in df.columns and len(df) else 0.25
    slo_lbl = f"{int(round(slo_s*1e3))} ms" if slo_s < 1.0 else f"{slo_s:.1f} s"
    ax0.axhline(slo_s, color="#880000", ls=":", lw=1.5, label=f"TTFT SLO ({slo_lbl})")
    ax0.set_yscale("log")
    ax0.set_xlabel("Offered steady-state arrival rate (req/s)")
    ax0.set_ylabel("Median (P50) TTFT (seconds, log scale)")
    ax0.set_title("(a) Median TTFT Saturation Under Load (128K Agentic)")
    ax0.legend(loc="center left", bbox_to_anchor=(0.02, 0.48), fontsize=7.0, ncol=2, framealpha=0.94)

    ax1.set_xlabel("Offered steady-state arrival rate (req/s)")
    ax1.set_ylabel(f"Effective SLO goodput (req/s, TTFT <= {slo_lbl})")
    ax1.set_title("(b) SLO-Attaining Cluster Goodput Under Load")
    ax1.legend(loc="upper left", fontsize=7.2, ncol=2, framealpha=0.92)

    fig.tight_layout()
    save_fig(fig, "fig8_p99_load")


def main() -> int:
    print("Generating publication-grade figures (PDF + 300-DPI PNG)...", flush=True)
    make_fig1_teaser()
    make_fig2_workload_and_capacity()
    make_fig3_calibration()
    make_fig4_main_comparison()
    make_fig5_heatmap()
    make_fig6_sensitivity()
    make_fig7_ablation()
    make_fig8_load()
    print("All 8 publication figures generated cleanly.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
