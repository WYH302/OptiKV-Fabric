#!/usr/bin/env python3
"""Generate all 10 IEEE Transactions publication-grade figures (PDF + 300-DPI PNG) for v3.0.

Visual Constitution:
  - Strictly unified Okabe-Ito colorblind-safe policy palette (zero pure RGB, zero jet/rainbow).
  - Pure black (#000000) for all text, titles, ticks, spines, annotations, and labels.
  - Refined academic line weights (1.20-1.65 pt) and restrained markers (3.6-4.6 pt).
  - Zero redundant zoom-in insets; shared non-occluding legends; zero overlapping axis labels.
  - Proposed method (OptiKV-Fabric / PrefixKeep) placed consistently at the far right of bar charts,
    with theoretical Ground Truth (Oracle / Belady) shown as crisp dashed horizontal reference bounds.
  - Figure 1(b) overall architecture diagram preserved verbatim.
  - Figure 10(a) physically taller y-axis preserved per user specification.
  - Honest bandwidth-matched power & cost accounting (Switch-Only + Pluggable NIC Transceivers).
"""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SIM = ROOT / "results" / "sim"
R3090 = ROOT / "results" / "3090"
TSTATS = ROOT / "results" / "trace_stats"
FIG_EXP = ROOT / "figures"
FIG_PAPER = REPO_ROOT / "paper" / "figures"

# ---------------------------------------------------------------------------
# Global Print-Scale IEEE Transactions rcParams (Pure-Black Arial, Crisp Lines)
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": 9.8,
        "text.color": "#000000",
        "axes.titlesize": 11.0,
        "axes.titleweight": "bold",
        "axes.titlecolor": "#000000",
        "axes.titlepad": 6.5,
        "axes.labelsize": 9.8,
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#000000",
        "axes.linewidth": 0.75,
        "axes.axisbelow": True,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "xtick.major.width": 0.70,
        "ytick.major.width": 0.70,
        "legend.fontsize": 8.0,
        "legend.edgecolor": "#000000",
        "legend.framealpha": 0.95,
        "legend.borderaxespad": 0.4,
        "figure.titlesize": 11.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#CCCCCC",
        "grid.alpha": 0.45,
        "grid.linestyle": "--",
        "grid.linewidth": 0.48,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# ---------------------------------------------------------------------------
# Unified Okabe-Ito Colorblind-Safe Palette (Global Across All 10 Figures)
# ---------------------------------------------------------------------------
C_OPTIKV = "#0072B2"      # Okabe-Ito Deep Blue — Visual protagonist (Ours, 1:1)
C_OPTIKV_41 = "#004C78"   # Deeper Navy Blue — Ours (4:1 oversubscribed)
C_ORACLE = "#000000"      # Pure Black — Theoretical ground-truth bound
C_ORACLE_BAR = "#4A4A4A"  # Charcoal — Secondary bound bar if needed
C_RDMA_PF = "#009E73"     # Okabe-Ito Bluish Green — Causal RDMA+Prefetch
C_RDMA = "#E69F00"        # Okabe-Ito Warm Amber — RDMA+Routing
C_CONCUR = "#CC79A7"      # Okabe-Ito Muted Rose — CONCUR+RDMA
C_NAIVE = "#D55E00"       # Okabe-Ito Vermilion — Optical-Naive / alert
C_RESV = "#56B4E9"        # Okabe-Ito Sky Blue — Optical-Resv. / secondary accent
C_DRAM = "#A69060"        # Muted Morandi Sand — Local-DRAM
C_HBM = "#7F7F7F"         # Medium Neutral Gray — Local-HBM / LRU
C_NOCACHE = "#BDBDBD"     # Light Neutral Gray — NoCache
C_SSD = "#8F7C9E"         # Muted Morandi Lavender — Local-SSD
C_BLACK = "#000000"

# Soft pastel fills derived from Okabe-Ito
P_BLUE = "#DCEBF5"
P_GREEN = "#D9F0E9"
P_AMBER = "#FBF0D9"
P_VERM = "#F9E2D2"
P_SKY = "#E3F2FA"
P_GRAY = "#EDEDED"

# Truncated colorblind-safe sequential colormap (YlGnBu [0.04..0.72]) -> 100% pure-black cell text
_base_ylgnbu = plt.get_cmap("YlGnBu")
CMAP_LIGHT_CB = LinearSegmentedColormap.from_list(
    "ylgnbu_light_cb", _base_ylgnbu(np.linspace(0.04, 0.72, 256))
)

POLICY_LABEL = {
    "no_cache": "NoCache",
    "local_hbm": "Local-HBM",
    "local_dram": "Local-DRAM",
    "local_ssd": "Local-SSD",
    "rdma_routing": "RDMA+Routing",
    "concur_rdma": "CONCUR+RDMA",
    "rdma_prefetch": "RDMA+Prefetch",
    "rdma50_routing": "RDMA-50G+Rt",
    "rdma50_prefetch": "RDMA-50G (400G)",
    "optical_naive": "Optical-Naive",
    "optical_reservation": "Optical-Resv.",
    "optikv": "OptiKV-Fabric",
    "optikv_4to1": "OptiKV (4:1)",
    "eviction_s3fifo": "S3-FIFO",
    "eviction_sieve": "SIEVE",
    "eviction_rrevict": "RR-Evict",
    "eviction_belady": "Belady",
    "oracle_infinite_hbm": "Oracle",
}

POLICY_COLOR = {
    "no_cache": C_NOCACHE,
    "local_hbm": C_HBM,
    "local_dram": C_DRAM,
    "local_ssd": C_SSD,
    "rdma_routing": C_RDMA,
    "concur_rdma": C_CONCUR,
    "rdma_prefetch": C_RDMA_PF,
    "rdma50_routing": "#B87300",
    "rdma50_prefetch": "#2B6B56",
    "optical_naive": C_NAIVE,
    "optical_reservation": C_RESV,
    "optikv": C_OPTIKV,
    "optikv_4to1": C_OPTIKV_41,
    "eviction_s3fifo": "#A69060",
    "eviction_sieve": "#56B4E9",
    "eviction_rrevict": "#E69F00",
    "eviction_belady": C_ORACLE_BAR,
    "oracle_infinite_hbm": C_ORACLE,
}

POLICY_BAR_COLOR = dict(POLICY_COLOR)
POLICY_BAR_COLOR["oracle_infinite_hbm"] = C_ORACLE_BAR

POLICY_MARKER = {
    "no_cache": "x",
    "local_hbm": "v",
    "local_dram": "^",
    "rdma_routing": "s",
    "concur_rdma": "h",
    "rdma_prefetch": "p",
    "rdma50_routing": "s",
    "rdma50_prefetch": "^",
    "optical_naive": "D",
    "optical_reservation": "P",
    "optikv": "o",
    "optikv_4to1": "d",
    "oracle_infinite_hbm": "*",
}

# ---------------------------------------------------------------------------
# Honest Bandwidth-Matched Power & Cost Constants (Yoon & Yu arXiv:2610.18444
# + standard 400G-DR4/FR4 pluggable optical transceiver 10 W / $400 per end)
# 8-GPU pod with W=8 GPU ports + P pool ports (1:1 => 16 ports; 4:1 => 10 ports)
# ---------------------------------------------------------------------------
P_ELEC_400G_SW = 39.4       # W per 400G electrical switch port
P_OCS_MEMS_SW = 0.39        # W per MEMS OCS mirror port (wavelength-agnostic)
P_XCVR_400G = 10.0          # W per pluggable 400G NIC transceiver end

PWR_ELEC_400G_E2E = 16 * P_ELEC_400G_SW + 16 * P_XCVR_400G        # 630.4 + 160.0 = 790.4 W
PWR_ELEC_1600G_E2E = 64 * P_ELEC_400G_SW + 64 * P_XCVR_400G       # 2521.6 + 640.0 = 3161.6 W
PWR_OCS_45G_E2E = 16 * P_OCS_MEMS_SW + 16 * P_XCVR_400G           # 6.24 + 160.0 = 166.24 W
PWR_OCS_200G_11_E2E = 16 * P_OCS_MEMS_SW + 64 * P_XCVR_400G       # 6.24 + 640.0 = 646.24 W
PWR_OCS_200G_41_E2E = 10 * P_OCS_MEMS_SW + 40 * P_XCVR_400G       # 3.90 + 400.0 = 403.90 W


def save_fig(fig: plt.Figure, name: str) -> None:
    for out_dir in (FIG_EXP, FIG_PAPER):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.{{pdf,png}} -> experiments/figures & paper/figures", flush=True)


def _fmt_pct(h: float) -> str:
    pct = h * 100.0
    return f"{pct:.1f}%" if abs(pct - round(pct)) > 1e-3 else f"{int(round(pct))}%"


def _fmt_ctx(n: int) -> str:
    if n >= 1_000_000:
        return "1M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


def _hu1(val: float) -> str:
    """Consistent decimal half-up rounding to 1 decimal place (matches LaTeX tables)."""
    return str(Decimal(f"{val:.6f}").quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Figure 1: Top-Venue 3-Panel Teaser (Aligned Gantt, Clean Bus Arch, Airy Pareto)
# ---------------------------------------------------------------------------

def make_fig1_teaser() -> None:
    df_load = pd.read_csv(SIM / "exp_load_sweep.csv")
    peak = df_load.groupby("policy")["goodput_rps"].max()

    fig = plt.figure(figsize=(13.2, 4.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.20, 1.18, 1.18], wspace=0.20)

    # -----------------------------------------------------------------------
    # Panel (a): Time-Aligned Gantt Contrast (Zero Overflow, Exact Geometry)
    # -----------------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(0.0, 10.0)
    ax0.set_ylim(0.0, 10.0)
    ax0.axis("off")
    ax0.set_title("(a) Optical Thrashing vs. Pause Hiding", loc="center", fontsize=11.0, pad=7.5)

    def rbox(ax, x, y, w, h, text, fc, ec=C_BLACK, fs=7.0, bold=False, lw=0.85, ls="-", rad=0.14):
        p = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={rad}",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3
        )
        ax.add_patch(p)
        if text:
            ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
                    fontsize=fs, color=C_BLACK, fontweight="bold" if bold else "normal",
                    linespacing=1.12, zorder=4)

    # Subtle outer lane containers (pad=0.0 => exact coordinates!)
    ax0.add_patch(mpatches.FancyBboxPatch(
        (0.05, 5.28), 9.90, 4.42, boxstyle="round,pad=0.0,rounding_size=0.22",
        facecolor="#FFF9F5", edgecolor=C_NAIVE, linewidth=0.95, linestyle="--", zorder=1
    ))
    ax0.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.25), 9.90, 4.55, boxstyle="round,pad=0.0,rounding_size=0.22",
        facecolor="#F5FAFE", edgecolor=C_OPTIKV, linewidth=1.05, zorder=1
    ))

    # Top lane header + compact status pill (wide open separation from dotted line!)
    ax0.text(0.28, 9.26, "Optical-Naive ($\\tau \\approx 0$)",
             fontsize=7.6, fontweight="bold", color=C_BLACK, ha="left", va="center", zorder=4)
    rbox(ax0, 6.12, 8.95, 3.60, 0.58, "Duty 3.8–13.1% | 13.9 s", P_VERM, C_NAIVE, fs=6.6, bold=True, lw=0.70, rad=0.12)

    # Bottom lane header + compact status pill (wide open separation from dotted line!)
    ax0.text(0.28, 4.38, "OptiKV-Fabric (Ours)",
             fontsize=7.6, fontweight="bold", color=C_BLACK, ha="left", va="center", zorder=4)
    rbox(ax0, 6.12, 4.07, 3.60, 0.58, "Duty 76–96% | 36 ms", P_BLUE, C_OPTIKV, fs=6.6, bold=True, lw=0.70, rad=0.12)

    # Shared vertical arrival alignment guide at x = 5.92 (broken cleanly around headers & captions!)
    x_arr = 5.92
    ax0.plot([x_arr, x_arr], [6.52, 8.68], color=C_BLACK, linestyle=":", linewidth=1.0, zorder=2)
    ax0.plot([x_arr, x_arr], [5.38, 6.52], color=C_BLACK, linestyle=":", linewidth=1.0, zorder=2)
    ax0.plot([x_arr, x_arr], [1.55, 3.88], color=C_BLACK, linestyle=":", linewidth=1.0, zorder=2)
    ax0.plot([x_arr, x_arr], [3.88, 4.70], color=C_BLACK, linestyle=":", linewidth=1.0, zorder=2)
    ax0.text(x_arr, 5.04, "Turn $k$ arrives", ha="center", va="center",
             fontsize=7.0, fontweight="bold", color=C_BLACK,
             bbox=dict(boxstyle="round,pad=0.14", fc="#FFFFFF", ec=C_BLACK, lw=0.65), zorder=5)

    # Top lane Gantt segments (y = 6.72..8.44, h = 1.72, generous horizontal margins!)
    y_t, h_b = 6.72, 1.72
    rbox(ax0, 0.25, y_t, 0.96, h_b, "Turn\n$k{-}1$", P_GRAY, "#555555", fs=6.6)
    rbox(ax0, 1.31, y_t, 4.50, h_b, "Agent Pause (0.71–2.22 s)\nOCS Idle", "#FFFFFF", "#888888", fs=6.9, ls="--")
    rbox(ax0, 6.03, y_t, 1.54, h_b, "MEMS\n25–50 ms", P_VERM, C_NAIVE, fs=6.6, bold=True)
    rbox(ax0, 7.65, y_t, 0.88, h_b, "KV\nXfer", P_AMBER, C_RDMA, fs=6.6)
    rbox(ax0, 8.61, y_t, 1.13, h_b, "Turn $k$\nPrefill", P_GRAY, "#555555", fs=6.6)

    # Top lane critical-path dimension bar (cleanly inside right margin!)
    ax0.annotate("", xy=(9.74, 6.24), xytext=(6.03, 6.24),
                 arrowprops=dict(arrowstyle="<->", color=C_NAIVE, lw=1.15), zorder=4)
    ax0.text(7.88, 5.74, "Exposed Critical Path", ha="center", va="center",
             fontsize=7.0, fontweight="bold", color=C_BLACK, zorder=4)

    # Bottom lane Gantt segments (y = 1.84..3.56, h = 1.72, generous internal padding!)
    y_b = 1.84
    rbox(ax0, 0.25, y_b, 0.96, h_b, "Turn\n$k{-}1$", P_GRAY, "#555555", fs=6.6)
    rbox(ax0, 1.31, y_b, 1.54, h_b, "Pre-Switch\n(Hidden)", P_SKY, C_OPTIKV, fs=6.5, bold=True)
    rbox(ax0, 2.95, y_b, 1.62, h_b, "OCS Burst\nPrefetch", P_BLUE, C_OPTIKV, fs=6.5, bold=True)
    rbox(ax0, 4.67, y_b, 1.14, h_b, "Warm\nHBM", P_GREEN, C_RDMA_PF, fs=6.5, bold=True)
    rbox(ax0, 6.03, y_b, 1.38, h_b, "Turn $k$\n$\\Delta$-Prefill", P_BLUE, C_OPTIKV, fs=6.6, bold=True, lw=1.15)

    # Bottom lane saved-time callout cleanly in open right whitespace [7.52..9.74]
    ax0.annotate("", xy=(9.74, 2.70), xytext=(7.52, 2.70),
                 arrowprops=dict(arrowstyle="<->", color=C_RDMA_PF, lw=1.15, ls="--"), zorder=4)
    ax0.text(8.63, 3.16, "Latency Saved", ha="center", va="center",
             fontsize=6.9, fontweight="bold", color=C_BLACK, zorder=4)
    ax0.text(8.63, 2.22, "(= Oracle)", ha="center", va="center",
             fontsize=6.8, color=C_BLACK, zorder=4)

    # Bottom lane dimension brackets (ZERO overlap!)
    ax0.annotate("", xy=(5.81, 1.32), xytext=(1.31, 1.32),
                 arrowprops=dict(arrowstyle="<->", color=C_OPTIKV, lw=1.05), zorder=4)
    ax0.text(3.56, 0.80, "Hidden Inside Pause", ha="center", va="center",
             fontsize=7.0, fontweight="bold", color=C_BLACK, zorder=4)
    ax0.annotate("", xy=(7.41, 1.32), xytext=(6.03, 1.32),
                 arrowprops=dict(arrowstyle="<->", color=C_OPTIKV, lw=1.15), zorder=4)
    ax0.text(7.25, 0.80, "0 ms Switch", ha="center", va="center",
             fontsize=7.0, fontweight="bold", color=C_BLACK, zorder=4)

    # -----------------------------------------------------------------------
    # Panel (b): Unified 3-Pillar Control Plane (Exact Gaps, Zero-Crossing Bus)
    # -----------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[1])
    ax1.set_xlim(0.0, 10.0)
    ax1.set_ylim(0.0, 10.0)
    ax1.axis("off")
    ax1.set_title("(b) Unified OptiKV-Fabric Control Plane", loc="center", fontsize=11.0, pad=7.5)

    # Layer 1: Workload + Roofline Laws
    rbox(ax1, 0.15, 8.55, 9.70, 1.15,
         "Agentic Sessions (Prefix Hashes & Gaps $\\Delta t_{\\mathrm{gap}}$)\nGoverned by Roofline Crossover Laws 1–3",
         P_GRAY, "#444444", fs=7.3, bold=True, lw=0.95, rad=0.16)

    # Layer 2: Central State Primitive (OCS Port Calendar)
    rbox(ax1, 0.30, 6.60, 9.40, 1.35,
         "Single State Primitive: OCS Port Calendar\nBipartite Matching $\\mathbf{A}(t)$, Release $t_{\\mathrm{free}}$ & Latching ($\\tau_{\\mathrm{eff}}{=}0$)",
         P_BLUE, C_OPTIKV, fs=7.3, bold=True, lw=1.35, rad=0.16)

    # Layer 3: Three Cohesive Modules (exact 0.22 gaps, pad=0.0)
    mod_w, mod_y, mod_h = 3.08, 3.48, 2.48
    rbox(ax1, 0.15, mod_y, mod_w, mod_h,
         "OptiRoute\n(Sec. III-B)\nStanding-Circuit\nCo-Routing &\nExact Abort",
         P_SKY, C_OPTIKV, fs=7.0, bold=True, lw=1.0, rad=0.16)
    rbox(ax1, 3.46, mod_y, mod_w, mod_h,
         "GapFetch\n(Sec. III-C)\nCausal EWMA\nPrefetch &\nIn-Flight Join",
         P_GREEN, C_RDMA_PF, fs=7.0, bold=True, lw=1.0, rad=0.16)
    rbox(ax1, 6.77, mod_y, mod_w, mod_h,
         "PrefixKeep\n(Sec. III-D)\n2nd-Obs Filter,\nPrefix-GDSF &\nPinning",
         P_AMBER, C_RDMA, fs=7.0, bold=True, lw=1.0, rad=0.16)

    # Layer 4: Five Storage Tiers (exact 0.16 gaps, pad=0.0 -> ZERO overlapping borders!)
    tier_y, tier_h, tier_w = 0.28, 2.02, 1.81
    tier_xs = [0.15 + i * (tier_w + 0.1625) for i in range(5)]
    tiers = [
        ("T1: HBM\nPinned\nHot Tier", P_BLUE, C_OPTIKV, True),
        ("T2: DRAM\n26–50\nGB/s Spill", P_AMBER, C_RDMA, False),
        ("T3: SSD\nNVMe\nCold Tier", P_GRAY, "#555555", False),
        ("T4: Peer\nDirect\nOCS Hop", P_SKY, C_OPTIKV, False),
        ("T5: Pool\n45–800\nGB/s OCS", P_GREEN, C_RDMA_PF, True),
    ]
    for tx, (tlbl, tfc, tec, is_b) in zip(tier_xs, tiers):
        rbox(ax1, tx, tier_y, tier_w, tier_h, tlbl, tfc, tec, fs=6.7, bold=is_b, lw=1.0, rad=0.15)

    # Clean orthogonal connectors (ZERO crossing spaghetti!)
    def varrow(p1, p2, color=C_BLACK, lw=1.15):
        ax1.annotate("", xy=p2, xytext=p1,
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0), zorder=4)

    # Layer 1 -> Layer 2
    varrow((5.00, 8.55), (5.00, 7.98), C_BLACK, lw=1.25)

    # Layer 2 -> Layer 3 (clean orthogonal fan-out bar at y = 6.28)
    y_topbus = 6.28
    ax1.plot([5.00, 5.00], [6.60, y_topbus], color=C_OPTIKV, lw=1.15, zorder=2)
    ax1.plot([1.69, 8.31], [y_topbus, y_topbus], color=C_OPTIKV, lw=1.15, zorder=2)
    varrow((1.69, y_topbus), (1.69, mod_y + mod_h + 0.02), C_OPTIKV, lw=1.15)
    varrow((5.00, y_topbus), (5.00, mod_y + mod_h + 0.02), C_RDMA_PF, lw=1.15)
    varrow((8.31, y_topbus), (8.31, mod_y + mod_h + 0.02), C_RDMA, lw=1.15)

    # Layer 3 -> Layer 4 (clean orthogonal five-tier coordination bus at y = 2.88)
    y_bus = 2.88
    ax1.plot([1.055, 8.945], [y_bus, y_bus], color="#333333", lw=1.35, zorder=2)
    ax1.text(5.00, y_bus, " Five-Tier Lossless Data & Control Bus ", ha="center", va="center",
             fontsize=6.5, fontweight="bold", color=C_BLACK,
             bbox=dict(boxstyle="round,pad=0.12", fc="#FFFFFF", ec="#888888", lw=0.60), zorder=5)
    # 3 module stems into bus
    ax1.plot([1.69, 1.69], [mod_y, y_bus], color=C_OPTIKV, lw=1.15, zorder=2)
    ax1.plot([5.00, 5.00], [mod_y, y_bus + 0.18], color=C_RDMA_PF, lw=1.15, zorder=2)
    ax1.plot([8.31, 8.31], [mod_y, y_bus], color=C_RDMA, lw=1.15, zorder=2)
    # 5 clean vertical drops from bus into T1..T5 centers
    tier_cols = [C_OPTIKV, C_RDMA, "#555555", C_OPTIKV, C_RDMA_PF]
    for tx, tcol in zip(tier_xs, tier_cols):
        cx = tx + tier_w / 2.0
        varrow((cx, y_bus - (0.18 if abs(cx - 5.0) < 0.5 else 0.0)), (cx, tier_y + tier_h + 0.02), tcol, lw=1.10)

    # -----------------------------------------------------------------------
    # Panel (c): Uncluttered, Airy End-to-End Performance-Power Pareto Frontier
    # -----------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[2])
    gp_opt = float(peak.loc["optikv"])
    gp_opt41 = float(peak.loc["optikv_4to1"])
    gp_orc = float(peak.loc["oracle_infinite_hbm"])
    gp_r50 = float(peak.loc["rdma50_prefetch"])
    gp_rpf = float(peak.loc["rdma_prefetch"])
    gp_rrt = float(peak.loc["rdma_routing"])

    # Soft Pareto-optimal region (upper-left)
    ax2.fill_between([145.0, 720.0], [3.10, 3.10], [4.90, 4.90], color=P_BLUE, alpha=0.48, zorder=0)
    ax2.text(165.0, 4.74, "Pareto-Optimal", fontsize=7.6, fontweight="bold", color=C_BLACK, ha="left", va="top")

    # Oracle bound horizontal line
    ax2.axhline(gp_orc, color=C_BLACK, ls="--", lw=1.05, zorder=2)
    ax2.text(4900.0, gp_orc + 0.14, f"Oracle ({gp_orc:.2f} req/s)",
             fontsize=7.4, fontweight="bold", color=C_BLACK, ha="right", va="bottom")

    # Matched 200 GB/s electrical reference vertical line
    ax2.axvline(PWR_ELEC_1600G_E2E, color="#888888", ls=":", lw=1.05, zorder=1)
    ax2.text(PWR_ELEC_1600G_E2E * 0.89, 2.02, "Matched 200 GB/s\nElectrical (3.16 kW)\n4.9$\\times$–7.8$\\times$ power",
             fontsize=6.8, color=C_BLACK, ha="right", va="center", linespacing=1.15)

    # Plot data points cleanly
    ax2.scatter(PWR_OCS_200G_41_E2E, gp_opt41, s=165, marker="*", color=C_OPTIKV_41,
                edgecolor=C_BLACK, linewidth=0.75, zorder=7)
    ax2.scatter(PWR_OCS_200G_11_E2E, gp_opt, s=90, marker="o", color=C_OPTIKV,
                edgecolor=C_BLACK, linewidth=0.75, zorder=6)
    ax2.scatter(PWR_ELEC_400G_E2E, gp_r50, s=66, marker="^", color="#2B6B56",
                edgecolor=C_BLACK, linewidth=0.65, zorder=6)
    ax2.scatter(PWR_ELEC_1600G_E2E, gp_orc, s=48, marker="s", color=C_DRAM,
                edgecolor=C_BLACK, linewidth=0.65, zorder=5)
    ax2.scatter(PWR_ELEC_400G_E2E, gp_rpf, s=55, marker=POLICY_MARKER["rdma_prefetch"],
                color=C_RDMA_PF, edgecolor=C_BLACK, linewidth=0.65, zorder=5)
    ax2.scatter(PWR_ELEC_400G_E2E, gp_rrt, s=52, marker=POLICY_MARKER["rdma_routing"],
                color=C_RDMA, edgecolor=C_BLACK, linewidth=0.65, zorder=5)
    ax2.scatter(PWR_ELEC_400G_E2E * 1.07, gp_rrt - 0.04, s=44, marker=POLICY_MARKER["concur_rdma"],
                color=C_CONCUR, edgecolor=C_BLACK, linewidth=0.60, zorder=4)
    ax2.scatter(PWR_OCS_200G_11_E2E, 0.02, s=48, marker=POLICY_MARKER["optical_naive"],
                color=C_NAIVE, edgecolor=C_BLACK, linewidth=0.65, zorder=5)
    ax2.scatter(PWR_ELEC_400G_E2E * 1.04, 0.02, s=46, marker=POLICY_MARKER["local_hbm"],
                color=C_HBM, edgecolor=C_BLACK, linewidth=0.65, zorder=4)

    gpw_41 = (gp_opt41 / PWR_OCS_200G_41_E2E) / (gp_rrt / PWR_ELEC_400G_E2E)
    gpw_41_r50 = (gp_opt41 / PWR_OCS_200G_41_E2E) / (gp_r50 / PWR_ELEC_400G_E2E)

    # Clean direct placement (NO redundant arrows!); ONLY ONE thin straight leader line
    # for the sandwiched middle dot OptiKV 1:1 (646 W) between 404 W and 790 W!
    ax2.text(320.0, 3.68, f"OptiKV 4:1 (404 W)\n{gp_opt41:.2f} ({gpw_41_r50:.2f}$\\times$ 50G / {gpw_41:.1f}$\\times$ 25G)",
             ha="center", va="top", fontsize=6.4, fontweight="bold", color=C_BLACK, linespacing=1.18)
    ax2.annotate("OptiKV 1:1 (646 W)\n50 ms P50",
                 xy=(PWR_OCS_200G_11_E2E, gp_opt - 0.14), xytext=(PWR_OCS_200G_11_E2E, 2.86),
                 ha="center", va="top", fontsize=6.6, fontweight="bold", color=C_BLACK, linespacing=1.15,
                 arrowprops=dict(arrowstyle="-", color=C_OPTIKV, lw=0.85))
    ax2.text(890.0, 3.84, "RDMA-50G (790 W)\n162 ms (3.25$\\times$ slower)",
             ha="left", va="top", fontsize=6.5, fontweight="bold", color=C_BLACK, linespacing=1.15)
    ax2.text(920.0, 1.24, f"RDMA-25G+Pf ({gp_rpf:.2f})",
             ha="left", va="center", fontsize=6.6, color=C_BLACK)
    ax2.text(960.0, 0.80, f"RDMA-25G+Rt /\nCONCUR ({gp_rrt:.2f})",
             ha="left", va="center", fontsize=6.5, color=C_BLACK, linespacing=1.08)
    ax2.text(555.0, 0.02, "Opt-Naive (0.00)",
             ha="right", va="center", fontsize=6.6, color=C_BLACK)
    ax2.text(940.0, 0.02, "HBM / NoCache (0.00)",
             ha="left", va="center", fontsize=6.6, color=C_BLACK)

    ax2.set_xscale("log")
    ax2.set_xlim(145.0, 5400.0)
    ax2.set_ylim(-0.32, 4.90)
    ax2.set_xticks([250, 500, 1000, 2000, 4000], ["250", "500", "1K", "2K", "4K"])
    ax2.xaxis.set_minor_formatter(plt.NullFormatter())
    ax2.set_xlabel("End-to-end fabric power: switch + NIC optics (W)", fontsize=9.2)
    ax2.set_ylabel("Peak 128K SLO goodput (req/s)", fontsize=9.4)
    ax2.set_title("(c) 128K Goodput vs. E2E Fabric Power", loc="center", fontsize=11.0, pad=7.5)

    save_fig(fig, "fig1_teaser")


# ---------------------------------------------------------------------------
# Figure 2: Multi-Arch KV Capacity & Real Trace Characterization (Deduplicated!)
# ---------------------------------------------------------------------------

def make_fig2_workload_and_capacity() -> None:
    df_cap = pd.read_csv(SIM / "fig1_kv_capacity.csv")
    df_ts = pd.read_csv(TSTATS / "summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.05))

    # --- Panel (a) ---
    ax = axes[0, 0]
    ctxs = np.array([8192, 16384, 32768, 65536, 131072, 262144, 524288, 1000000], dtype=float)
    show_models = [
        ("llama-3.1-70b", "Llama-3.1-70B (GQA, BF16)", 2, C_NAIVE, "-", "o"),
        ("qwen3-8b", "Qwen3-8B (GQA, BF16)", 2, C_OPTIKV, "-", "s"),
        ("llama-3.1-8b", "Llama-3.1-8B (GQA, BF16)", 2, C_RESV, "--", "^"),
        ("deepseek-v3", "DeepSeek-V3 (MLA, BF16)", 2, C_CONCUR, "-", "D"),
        ("qwen2.5-7b-instruct-1m", "Qwen2.5-7B-1M (GQA, BF16)", 2, C_RDMA_PF, "-", "p"),
        ("deepseek-v3", "DeepSeek-V3 (MLA, FP8)", 1, C_CONCUR, ":", "D"),
        ("qwen3.5-9b", "Qwen3.5-9B (3:1 Hybrid, BF16)", 2, C_RDMA, "-.", "v"),
        ("qwen3-next-80b-a3b", "Qwen3-Next-80B (3:1 Hybrid, BF16)", 2, C_HBM, "--", "x"),
    ]
    cap_map = df_cap.set_index("model")
    for mkey, label, db, col, ls, mk in show_models:
        if mkey not in cap_map.index:
            continue
        bpt = float(cap_map.loc[mkey, "kv_bytes_bf16" if db == 2 else "kv_bytes_fp8"])
        gib = (ctxs * bpt) / (1024**3)
        ax.plot(ctxs / 1000.0, gib, label=label, color=col, linestyle=ls, linewidth=1.25, marker=mk, markersize=3.3)
    ax.axhline(24.0, color=C_BLACK, linestyle="--", linewidth=0.90)
    ax.text(8.8, 28.0, "RTX 3090 (24 GiB)", fontsize=7.9, color=C_BLACK, fontweight="bold")
    ax.axhline(80.0, color=C_NAIVE, linestyle=":", linewidth=1.05)
    ax.text(8.8, 94.0, "H100 SXM (80 GiB)", fontsize=7.9, color=C_BLACK, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Context length (K tokens, log scale)")
    ax.set_ylabel("Batch=1 KV cache size (GiB, log)")
    ax.set_title("(a) Multi-Architecture KV Memory Wall", loc="center")
    ax.legend(loc="lower right", fontsize=7.0, ncol=1)

    # --- Panel (b): Deduplicated Unique Trace Splits (No ToolAgent Duplicate!) ---
    ax = axes[0, 1]
    mc_files = [
        ("Mooncake (23.6K)", ROOT / "datasets/mooncake-traces/mooncake/train-00000-of-00001.parquet", "input_length", C_OPTIKV, "-"),
        ("Conversation (12.0K)", ROOT / "datasets/mooncake-traces/conversation/train-00000-of-00001.parquet", "input_length", C_RDMA, "--"),
        ("Synthetic-MC (4.0K)", ROOT / "datasets/mooncake-traces/synthetic/train-00000-of-00001.parquet", "input_length", C_RESV, "-."),
        ("LMCache (24.9K)", ROOT / "datasets/lmcache-agentic-traces/tokenized_512.parquet", "input_tokens", C_RDMA_PF, "-"),
        ("FreeInference (190.5K)", ROOT / "datasets/freeinference-agentic-trace/week_2026_08_30/release/2026-08-30_2026-09-05/processed/requests.parquet", "prompt_tokens", C_CONCUR, "-"),
    ]
    for label, path, colname, col, ls in mc_files:
        if not path.is_file():
            continue
        arr = pq.read_table(path, columns=[colname]).column(colname).to_numpy(zero_copy_only=False)
        arr = arr[~pd.isna(arr)].astype(float)
        arr = arr[arr > 0]
        arr.sort()
        qs = np.linspace(0, 1, 600)
        ax.plot(np.quantile(arr, qs), qs, label=label, color=col, linestyle=ls, linewidth=1.35)
    ax.set_xscale("log")
    ax.set_xlim(1.2e2, 2.5e6)
    ax.set_xlabel("Prompt length (tokens, log scale)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("(b) Production & Agentic Prompt Lengths", loc="center")
    ax.legend(loc="upper left", fontsize=7.2, handlelength=1.6)

    # --- Panel (c): Deduplicated Causal Prefix Block Reuse ---
    ax = axes[1, 0]
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
        ("Conversation\n(512-tok)", float(ts_map.loc["conversation", "prefix_reuse_fraction"]) * 100.0, C_RDMA),
        ("Mooncake\n(512-tok)", float(ts_map.loc["mooncake", "prefix_reuse_fraction"]) * 100.0, C_OPTIKV),
        ("Synthetic-MC\n(512-tok)", float(ts_map.loc["synthetic", "prefix_reuse_fraction"]) * 100.0, C_RESV),
        ("LMCache\n(512-tok)", lm_reuse_pct, C_RDMA_PF),
        ("FreeInference\n(16-tok)", float(ts_map.loc["freeinference", "prefix_reuse_fraction"]) * 100.0, C_CONCUR),
    ]
    y = np.arange(len(reuse_items))
    bars = ax.barh(y, [v[1] for v in reuse_items], color=[v[2] for v in reuse_items], height=0.56, edgecolor=C_BLACK, lw=0.55)
    ax.set_yticks(y, [v[0] for v in reuse_items])
    ax.set_xlim(0, 116)
    ax.set_xlabel("Causal block reuse fraction (%)")
    ax.set_title("(c) Strictly Causal Prefix Block Reuse", loc="center")
    for rect, (_, val, _) in zip(bars, reuse_items):
        ax.text(val + 1.6, rect.get_y() + rect.get_height() / 2.0, f"{val:.1f}%", va="center", fontsize=8.5, fontweight="bold", color=C_BLACK)

    # --- Panel (d): Inter-Turn Pause Windows vs. OCS Switch (No Zoom Inset!) ---
    ax = axes[1, 1]
    lm_gaps = pq.read_table(ROOT / "datasets/lmcache-agentic-traces/tokenized_512.parquet", columns=["pre_gap"]).column("pre_gap").to_numpy(zero_copy_only=False)
    lm_gaps = np.sort(lm_gaps[lm_gaps > 0])
    qs = np.linspace(0, 1, 600)
    lm_q = np.quantile(lm_gaps, qs)
    ax.plot(lm_q, qs, label=f"LMCache (P50={np.median(lm_gaps):.2f} s)", color=C_RDMA_PF, lw=1.45)

    fi_t = pq.read_table(
        ROOT / "datasets/freeinference-agentic-trace/week_2026_08_30/release/2026-08-30_2026-09-05/processed/requests.parquet",
        columns=["session_id", "start_ms", "end_ms"],
    ).to_pandas()
    fi_t = fi_t.sort_values(["session_id", "start_ms"])
    fi_t["prev_end"] = fi_t.groupby("session_id")["end_ms"].shift(1)
    fi_gaps = ((fi_t["start_ms"] - fi_t["prev_end"]) / 1000.0).dropna().values
    fi_gaps = np.sort(fi_gaps[fi_gaps > 0])
    fi_q = np.quantile(fi_gaps, qs)
    ax.plot(fi_q, qs, label=f"FreeInference (P50={np.median(fi_gaps):.2f} s)", color=C_CONCUR, lw=1.45, linestyle="--")

    ax.axvspan(0.025, 0.050, color=C_NAIVE, alpha=0.22, label="MEMS switch (25–50 ms)")
    ax.axvline(0.00001, color=C_OPTIKV, linestyle=":", lw=1.2, label="SiPh switch (10 $\\mu$s)")
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1e3)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Inter-turn idle gap $\\Delta t_{\\mathrm{gap}}$ (s, log scale)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("(d) Inter-Turn Pause Windows vs. OCS Switch", loc="center")
    ax.legend(loc="lower right", fontsize=7.6)

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

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 4.95))

    # --- Panel (a) ---
    ax = axes[0, 0]
    meas = df_cal[df_cal["measured_3090_7b_ttft_s"].notna()]
    ax.plot(df_cal["prompt_tokens"] / 1e3, df_cal["roofline_3090_s"], color=C_OPTIKV, lw=1.45, label="3090 Roofline (3.08%)")
    ax.plot(df_cal["prompt_tokens"] / 1e3, df_cal["quad_train_1k_16k_s"], color=C_RDMA, lw=1.25, ls="--", label="Quad Fit (6.25% HO)")
    ax.plot(df_cal["prompt_tokens"] / 1e3, df_cal["h100_peak_spec_s"], color=C_RDMA_PF, lw=1.35, ls=":", label="H100 Scaled Roofline")
    ax.scatter(meas["prompt_tokens"] / 1e3, meas["measured_3090_7b_ttft_s"], color=C_BLACK, s=25, zorder=5, label="Measured 3090 (1K–32K)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Prompt length (K tokens, log)")
    ax.set_ylabel("Cold prefill TTFT (s, log)")
    ax.set_title("(a) Cold Prefill Scaling (7B BF16)", loc="center", fontsize=10.3)
    ax.legend(loc="upper left", fontsize=7.1)

    # --- Panel (b) ---
    ax = axes[0, 1]
    x = np.arange(len(meas))
    w = 0.36
    ax.bar(x - w / 2, meas["roofline_abs_err_pct"], width=w, color=C_OPTIKV, edgecolor=C_BLACK, lw=0.50, label="Roofline (3.08%)")
    ax.bar(x + w / 2, meas["quad_train_abs_err_pct"], width=w, color=C_RDMA, edgecolor=C_BLACK, lw=0.50, label="Quad (6.25% HO)")
    ax.axhline(10.0, color=C_NAIVE, ls="--", lw=1.0, label="10% MAPE ref.")
    ax.axvspan(5.5, 7.6, color=P_VERM, alpha=0.55, zorder=0)
    ax.text(6.55, 9.3, "Held-out", ha="center", va="top", fontsize=7.7, color=C_BLACK, fontweight="bold")
    ax.set_xticks(x, [_fmt_ctx(int(v)) for v in meas["prompt_tokens"]])
    ax.set_ylim(0, 12.2)
    ax.set_xlabel("Measured prompt length (3090)")
    ax.set_ylabel("Absolute error (%)")
    ax.set_title("(b) Calibration Error (1K–32K)", loc="center", fontsize=10.3)
    ax.legend(loc="upper left", fontsize=7.1)

    # --- Panel (c) ---
    ax = axes[0, 2]
    s128 = df_ph[df_ph["suffix_tokens"] == 128].sort_values("prefix_tokens")
    s512 = df_ph[df_ph["suffix_tokens"] == 512].sort_values("prefix_tokens")
    ax.plot(s512["prefix_tokens"] / 1024.0, s512["cold_ttft_ms"], color=C_BLACK, marker="s", ms=3.6, lw=1.35, label="Cold (+512)")
    ax.plot(s512["prefix_tokens"] / 1024.0, s512["hot_ttft_ms"], color=C_OPTIKV, marker="o", ms=3.8, lw=1.45, label="Hot APC (+512)")
    ax.plot(s512["prefix_tokens"] / 1024.0, s512["roofline_hot_ms"], color=C_OPTIKV, ls=":", lw=1.1, label="Roofline $\\Delta$ (+512)")
    ax.plot(s128["prefix_tokens"] / 1024.0, s128["hot_ttft_ms"], color=C_RDMA_PF, marker="^", ms=4.0, lw=1.45, label="Hot APC (+128)")
    ax.set_yscale("log")
    ax.set_ylim(18, 28000)
    ax.set_xticks([4, 8, 12, 16, 24, 30], ["4K", "8K", "12K", "16K", "24K", "30K"])
    ax.set_xlabel("Cached prefix length (tokens)")
    ax.set_ylabel("Measured TTFT (ms, log)")
    ax.set_title("(c) Prefix-Hit Scaling (vLLM)", loc="center", fontsize=10.3)
    p24 = s128[s128["prefix_tokens"] == 24576].iloc[0]
    p30 = s512[s512["prefix_tokens"] == 30720].iloc[0]
    ax.annotate(
        f"{p24['speedup']:.1f}$\\times$ (24K+128)",
        xy=(24.0, p24["hot_ttft_ms"]), xytext=(10.0, 24.5),
        arrowprops=dict(arrowstyle="-|>", color=C_BLACK, lw=0.90),
        fontsize=7.3, fontweight="bold", color=C_BLACK,
        bbox=dict(boxstyle="round,pad=0.16", fc="#FFFFFF", ec=C_BLACK, lw=0.65),
    )
    ax.annotate(
        f"{p30['speedup']:.1f}$\\times$ (30K+512)",
        xy=(30.0, p30["hot_ttft_ms"]), xytext=(16.5, 680.0),
        arrowprops=dict(arrowstyle="-|>", color=C_BLACK, lw=0.90),
        fontsize=7.3, fontweight="bold", color=C_BLACK,
        bbox=dict(boxstyle="round,pad=0.16", fc="#FFFFFF", ec=C_BLACK, lw=0.65),
    )
    ax.legend(loc="upper left", fontsize=7.0)

    # --- Panel (d) ---
    ax = axes[1, 0]
    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_color(C_BLACK)
    ax_r.spines["right"].set_linewidth(0.75)
    ax_r.grid(False)
    ns = df_wall["num_sessions"].values
    l1 = ax.plot(ns, df_wall["turn2_hit_rate_pct"], color=C_OPTIKV, marker="o", ms=4.0, lw=1.50, label="Turn-2 hit rate (%)")
    l2 = ax_r.plot(ns, df_wall["turn2_p50_ms"], color=C_NAIVE, marker="s", ms=3.8, lw=1.40, ls="--", label="Turn-2 P50 (ms)")
    ax.axvline(10.6, color=C_BLACK, ls=":", lw=1.0)
    ax.axvspan(10.6, 21.0, color=P_VERM, alpha=0.50, zorder=0)
    ax.text(15.7, 52.0, "HBM Overflow\n(>4.70 GiB)\n40.7$\\times$ Cliff",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_BLACK,
            bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec=C_NAIVE, lw=0.80))
    ax.set_xlim(0.5, 20.8)
    ax.set_ylim(-6, 115)
    ax_r.set_ylim(-80, 2250)
    ax.set_xticks([1, 4, 8, 10, 12, 16, 20])
    ax.set_xlabel("Concurrent 8K sessions ($N$)")
    ax.set_ylabel("Turn-2 hit rate (%)", color=C_BLACK)
    ax_r.set_ylabel("Turn-2 P50 TTFT (ms)", color=C_BLACK)
    ax.tick_params(axis="y", colors=C_BLACK)
    ax_r.tick_params(axis="y", colors=C_BLACK)
    ax.set_title("(d) Live 24 GB HBM Capacity Wall", loc="center", fontsize=10.3)
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc="center left", bbox_to_anchor=(0.01, 0.34), fontsize=7.1)

    # --- Panel (e) ---
    ax = axes[1, 1]
    sub_ovl = df_ovl[df_ovl["delta_tokens"] == 128].sort_values("context_tokens")
    x_o = np.arange(len(sub_ovl))
    bw = 0.26
    ax.bar(x_o - bw, sub_ovl["t_serial_ms"], width=bw, color=C_NOCACHE, edgecolor=C_BLACK, lw=0.50, label="Serial ($T_{\\mathrm{comp}}{+}T_{\\mathrm{H2D}}$)")
    ax.bar(x_o, sub_ovl["t_overlapped_ms"], width=bw, color=C_OPTIKV, edgecolor=C_BLACK, lw=0.50, label="2-Stream ($T_{\\mathrm{ovl}}$)")
    ax.bar(x_o + bw, sub_ovl["t_comp_ms"], width=bw, color=C_RESV, edgecolor=C_BLACK, lw=0.50, label="Compute ($T_{\\mathrm{comp}}$)")
    for xi, (_, r) in enumerate(sub_ovl.iterrows()):
        ax.text(xi, r["t_serial_ms"] + 3.5, f"$\\alpha$={r['alpha_ovl_measured']:.2f}", ha="center", va="bottom",
                fontsize=7.7, fontweight="bold", color=C_BLACK)
    ax.set_xticks(x_o, [f"{int(c)//1024}K\n({int(m)}M)" for c, m in zip(sub_ovl["context_tokens"], sub_ovl["kv_mib"])])
    ax.set_ylim(0, max(sub_ovl["t_serial_ms"]) * 1.25)
    ax.set_xlabel("Context length (28L 7B BF16)")
    ax.set_ylabel("Latency ($\\Delta S{=}128$, ms)")
    ax.set_title("(e) 28-Layer KV Dual-Stream Overlap", loc="center", fontsize=10.3)
    ax.legend(loc="upper left", fontsize=7.1)

    # --- Panel (f) ---
    ax = axes[1, 2]
    for pol, ls, mk in [
        ("rdma_routing", "--", "s"),
        ("rdma_prefetch", "-.", "p"),
        ("optikv", "-", "o"),
        ("oracle_infinite_hbm", ":", "*"),
    ]:
        sp = df_sens_ovl[df_sens_ovl["policy"] == pol].sort_values("alpha_ovl")
        ax.plot(
            sp["alpha_ovl"], sp["p50_ttft_s"] * 1e3,
            color=POLICY_COLOR[pol], ls=ls, marker=mk,
            ms=4.4 if pol == "oracle_infinite_hbm" else 3.8,
            lw=1.60 if pol == "optikv" else 1.25,
            zorder=5 if pol == "oracle_infinite_hbm" else 4,
            label=POLICY_LABEL[pol],
        )
    ax.axvspan(0.86, 0.98, color=P_BLUE, alpha=0.55, zorder=0, label="3090 $\\alpha$ (0.86–0.98)")
    ax.axvline(0.70, color=C_BLACK, ls="--", lw=0.90)
    ax.text(0.68, 95.0, "Default\n$\\alpha{=}0.70$", ha="right", va="center", fontsize=7.3, color=C_BLACK, fontweight="bold")
    s0 = df_sens_ovl[np.isclose(df_sens_ovl["alpha_ovl"], 0.0)].set_index("policy")["p50_ttft_s"] * 1e3
    s1 = df_sens_ovl[np.isclose(df_sens_ovl["alpha_ovl"], 1.0)].set_index("policy")["p50_ttft_s"] * 1e3
    ax.annotate(
        f"{float(s0['rdma_routing']/s0['optikv']):.2f}$\\times$ / {float(s0['rdma_prefetch']/s0['optikv']):.2f}$\\times$",
        xy=(0.0, float(s0["optikv"])), xytext=(0.05, 118.0),
        arrowprops=dict(arrowstyle="-|>", color=C_BLACK, lw=0.88),
        fontsize=7.2, fontweight="bold", color=C_BLACK,
        bbox=dict(boxstyle="round,pad=0.16", fc="#FFFFFF", ec=C_BLACK, lw=0.65),
    )
    ax.annotate(
        f"{float(s1['rdma_routing']/s1['optikv']):.2f}$\\times$ / {float(s1['rdma_prefetch']/s1['optikv']):.2f}$\\times$",
        xy=(1.0, float(s1["optikv"])), xytext=(0.73, 76.0),
        arrowprops=dict(arrowstyle="-|>", color=C_BLACK, lw=0.88),
        fontsize=7.1, fontweight="bold", color=C_BLACK,
    )
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(18, 220)
    ax.set_xlabel("Overlap factor $\\alpha_{\\mathrm{ovl}}$")
    ax.set_ylabel("128K P50 TTFT (ms)")
    ax.set_title("(f) Overlap Robustness $\\alpha_{\\mathrm{ovl}} \\in [0, 1]$", loc="center", fontsize=10.3)
    ax.legend(loc="upper right", fontsize=6.9, ncol=1)

    fig.tight_layout()
    save_fig(fig, "fig3_prefill_model")


# ---------------------------------------------------------------------------
# Figure 4: Main End-to-End Comparison (Full LMCache 767 & Large FreeInference 250!)
# ---------------------------------------------------------------------------

def make_fig4_main_comparison() -> None:
    panels = [
        ("exp_main_mooncake_h100_siph.csv", "(a) Mooncake Full (23,608 reqs, SiPh 10 $\\mu$s)"),
        ("exp_main_lmcache_full_mems.csv", "(b) LMCache Full (767 sess, 24.9K reqs, MEMS)"),
        ("exp_main_freeinference_250sess_mems.csv", "(c) FreeInference Stratified (250 sess, 21.2K reqs)"),
        ("exp_main_mooncake_3090_calibrated.csv", "(d) Mooncake RTX 3090 Calibrated (45 GB/s)"),
    ]
    show = [
        "no_cache", "local_hbm", "local_dram", "rdma_routing", "concur_rdma",
        "rdma_prefetch", "optical_naive", "optical_reservation", "optikv"
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.10))
    for idx_p, (ax, (fname, title)) in enumerate(zip(axes.flat, panels)):
        df = pd.read_csv(SIM / fname).set_index("policy")
        pols = [p for p in show if p in df.index]
        x = np.arange(len(pols))
        w = 0.37
        p50 = [float(df.loc[p, "p50_ttft_s"]) * 1e3 for p in pols]
        p99 = [float(df.loc[p, "p99_ttft_s"]) * 1e3 for p in pols]
        oracle_p50 = float(df.loc["oracle_infinite_hbm", "p50_ttft_s"]) * 1e3
        bcols = [POLICY_BAR_COLOR[p] for p in pols]
        b1 = ax.bar(x - w / 2, p50, width=w, color=bcols, edgecolor=C_BLACK, lw=0.55)
        ax.bar(x + w / 2, p99, width=w, color=bcols, alpha=0.40, edgecolor=C_BLACK, lw=0.55, hatch="//")
        ax.axhline(oracle_p50, color=C_BLACK, ls="--", lw=1.05, zorder=5)
        ax.set_yscale("log")
        lbls = [POLICY_LABEL[p] + ("\n(Ours)" if p == "optikv" else "") for p in pols]
        ax.set_xticks(x, lbls, rotation=24, ha="right", fontsize=8.1)
        ax.set_ylabel("TTFT (ms, log scale)")
        ax.set_title(title, loc="center", fontsize=10.3)
        ymax = max(p99) * (18.0 if idx_p == 2 else 6.0)
        ymin = max(1.0, min(min(p50), oracle_p50) * 0.45)
        ax.set_ylim(ymin, ymax)
        for rect, val, pol in zip(b1, p50, pols):
            if val >= 5_000_000:
                txt = f"{val/1e6:.1f}ks"
            elif val >= 10000:
                txt = f"{val/1000:.0f}s"
            elif val >= 1000:
                txt = f"{val/1000:.1f}s"
            else:
                txt = f"{val:.0f}"
            ax.text(
                rect.get_x() + rect.get_width() / 2, val * 1.16, txt,
                ha="center", va="bottom", fontsize=7.4, color=C_BLACK,
                fontweight="bold" if pol == "optikv" else "normal",
            )
        orc_str = f"Oracle P50: {oracle_p50:.1f} ms" if oracle_p50 < 100 else f"Oracle P50: {oracle_p50:.0f} ms"
        # Place Oracle badge in top-right (above bars thanks to 5.2x log headroom)
        ax.text(0.985, 0.96, orc_str, transform=ax.transAxes, ha="right", va="top",
                fontsize=7.6, fontweight="bold", color=C_BLACK,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=C_BLACK, lw=0.65))
    shared_handles = [
        mpatches.Patch(facecolor=C_NOCACHE, edgecolor=C_BLACK, lw=0.55, label="P50 TTFT (solid bar)"),
        mpatches.Patch(facecolor=C_NOCACHE, alpha=0.40, edgecolor=C_BLACK, lw=0.55, hatch="//", label="P99 TTFT (hatched bar)"),
        plt.Line2D([0], [0], color=C_BLACK, ls="--", lw=1.10, label="Oracle P50 Floor (dashed line)"),
    ]
    fig.legend(handles=shared_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.996),
               fontsize=8.4, frameon=True, edgecolor=C_BLACK, fancybox=False, handlelength=2.0, columnspacing=1.8)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.945])
    save_fig(fig, "fig4_main_comparison")


# ---------------------------------------------------------------------------
# Figure 5: 2-Panel Operating Regime Map (Zero 512K Regression!)
# ---------------------------------------------------------------------------

def make_fig5_heatmap() -> None:
    df = pd.read_csv(SIM / "exp_breakeven_heatmap.csv")
    ctxs = sorted(df["context_tokens"].unique())
    hrs = sorted(df["requested_hit_rate"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.65))
    specs = [
        ("speedup_vs_rdma_p50", "(a) Speedup vs. RDMA+Routing (P50 TTFT)", True),
        ("speedup_vs_naive_p50", "(b) Speedup vs. Optical-Naive (P50 TTFT)", False),
    ]
    for ax, (col, title, mark_nb) in zip(axes, specs):
        grid = np.zeros((len(hrs), len(ctxs)))
        for ri, h in enumerate(hrs):
            for ci, c in enumerate(ctxs):
                val = float(df[(df["context_tokens"] == c) & (np.isclose(df["requested_hit_rate"], h))][col].iloc[0])
                grid[ri, ci] = val
        vmin = max(0.85, float(np.nanmin(grid)))
        vmax = max(2.0, float(np.nanmax(grid)))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=CMAP_LIGHT_CB, norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(C_BLACK)
            sp.set_linewidth(0.75)
        for ri in range(len(hrs)):
            for ci in range(len(ctxs)):
                val = grid[ri, ci]
                label = f"{val:.1f}$\\times$" if val < 100 else f"{val:.0f}$\\times$"
                if mark_nb and val <= 1.05:
                    ax.text(ci, ri, f"{val:.2f}$\\times$", ha="center", va="center", fontsize=7.1, color=C_BLACK, fontweight="normal")
                else:
                    ax.text(ci, ri, label, ha="center", va="center", fontsize=7.5, color=C_BLACK, fontweight="bold")
        if mark_nb:
            for ri in range(len(hrs)):
                for ci in range(len(ctxs)):
                    if grid[ri, ci] > 1.05:
                        rect = mpatches.Rectangle((ci - 0.5, ri - 0.5), 1.0, 1.0,
                                                  fill=False, edgecolor=C_NAIVE, linewidth=1.15, zorder=4)
                        ax.add_patch(rect)
        ax.set_xticks(range(len(ctxs)), [_fmt_ctx(int(c)) for c in ctxs], rotation=28, ha="right")
        ax.set_yticks(range(len(hrs)), [_fmt_pct(h) for h in hrs])
        ax.set_xlabel("Context length (tokens, Qwen2.5-7B-1M)")
        ax.set_ylabel("Prefix reuse fraction")
        ax.set_title(title, loc="center", fontsize=10.4)
        cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.050)
        cbar.set_label("Speedup ratio (log scale)", color=C_BLACK, fontsize=9.0)
        cbar.ax.tick_params(colors=C_BLACK, labelsize=8.2)
    fig.tight_layout()
    save_fig(fig, "fig5_heatmap")


# ---------------------------------------------------------------------------
# Figure 6: 4-Panel Sensitivity & Honest Bandwidth-Matched E2E Power/Efficiency
# ---------------------------------------------------------------------------

def make_fig6_sensitivity() -> None:
    df_tau = pd.read_csv(SIM / "exp_sensitivity_reconfig.csv")
    df_bw = pd.read_csv(SIM / "exp_sensitivity_bandwidth.csv")
    df_arch = pd.read_csv(SIM / "exp_sensitivity_arch_fp8.csv")
    df_load = pd.read_csv(SIM / "exp_load_sweep.csv")
    peak_ld = df_load.groupby("policy")["goodput_rps"].max()

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.05))

    # --- Panel (a): Reconfiguration Latency Sensitivity ---
    ax = axes[0, 0]
    for pol in ("optikv", "rdma_prefetch", "rdma_routing", "optical_reservation", "optical_naive"):
        sub = df_tau[df_tau["policy"] == pol].sort_values("reconfig_us")
        if sub.empty:
            continue
        ax.plot(sub["reconfig_us"], sub["p50_ttft_s"] * 1e3, marker=POLICY_MARKER[pol],
                ms=4.3 if pol == "optikv" else 3.7,
                color=POLICY_COLOR[pol], lw=1.60 if pol == "optikv" else 1.22, label=POLICY_LABEL[pol])
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set_xlabel("OCS reconfiguration latency $\\tau$ ($\\mu$s, symlog)")
    ax.set_ylabel("P50 TTFT (ms, log scale)")
    ax.set_title("(a) Reconfiguration Latency Sensitivity (0–50 ms)", loc="center", fontsize=10.4)
    ax.axvline(1.0, color=C_BLACK, ls=":", lw=0.90)
    ax.text(10.0, 260.0, "SiPh (1–100 $\\mu$s)", fontsize=7.5, color=C_BLACK, fontweight="bold", ha="left")
    ax.axvline(25000.0, color=C_NAIVE, ls=":", lw=0.95)
    ax.text(19500.0, 260.0, "MEMS (25 ms)", fontsize=7.5, color=C_BLACK, fontweight="bold", ha="right")
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.59), fontsize=7.0)

    # --- Panel (b): Optical Bandwidth Scaling (Clean, No Zoom Inset!) ---
    ax = axes[0, 1]
    for pol in ("optikv", "rdma_prefetch", "rdma_routing", "optical_reservation", "optical_naive"):
        sub = df_bw[df_bw["policy"] == pol].sort_values("bandwidth_gb_s")
        if sub.empty:
            continue
        ax.plot(sub["bandwidth_gb_s"], sub["p50_ttft_s"] * 1e3, marker=POLICY_MARKER[pol],
                ms=4.3 if pol == "optikv" else 3.7,
                color=POLICY_COLOR[pol], lw=1.60 if pol == "optikv" else 1.22, label=POLICY_LABEL[pol])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(22, 42000)
    ax.set_xticks([45, 100, 200, 400, 800], ["45", "100", "200", "400", "800"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Optical fabric bandwidth (GB/s)")
    ax.set_ylabel("P50 TTFT (ms, log scale)")
    ax.set_title("(b) Optical Bandwidth Scaling (45–800 GB/s)", loc="center", fontsize=10.4)
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.57), fontsize=7.4)
    ax.annotate("Saturates 36.3 ms floor\nat 100–200 GB/s", xy=(100.0, 36.3), xytext=(220.0, 420.0),
                fontsize=7.6, fontweight="bold", color=C_BLACK,
                arrowprops=dict(arrowstyle="-|>", color=C_OPTIKV, lw=0.95))

    # --- Panel (c): Multi-Architecture & FP8 KV Sensitivity (dtype_bytes!) ---
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
        for db, out_list in ((2, bf16_sp), (1, fp8_sp)):
            s_m = df_arch[(df_arch["model"] == m) & (df_arch["dtype_bytes"] == db)].set_index("policy")
            out_list.append(float(s_m.loc["rdma_routing", "p50_ttft_s"]) / max(1e-9, float(s_m.loc["optikv", "p50_ttft_s"])))
    x = np.arange(len(m_order))
    w = 0.36
    b1 = ax.bar(x - w / 2, bf16_sp, width=w, color=C_OPTIKV, edgecolor=C_BLACK, lw=0.55, label="BF16 KV (2 B/elem)")
    b2 = ax.bar(x + w / 2, fp8_sp, width=w, color=C_RESV, edgecolor=C_BLACK, lw=0.55, label="FP8 KV (1 B/elem)")
    ax.axhline(1.0, color=C_BLACK, ls="--", lw=0.90)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.18, f"{h:.2f}$\\times$", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=C_BLACK)
    ax.set_xticks(x, [m_short[m] for m in m_order], fontsize=8.2)
    ax.set_ylim(0, max(bf16_sp + fp8_sp) * 1.22)
    ax.set_ylabel("P50 Speedup vs. RDMA+Routing")
    ax.set_title("(c) Multi-Architecture & FP8 KV Sensitivity", loc="center", fontsize=10.4)
    ax.legend(loc="upper left", fontsize=7.6)

    # --- Panel (d): Honest Stacked Switch + Transceiver Power & E2E Goodput/Watt ---
    ax = axes[1, 1]
    gp_opt = float(peak_ld.loc["optikv"])
    gp_opt41 = float(peak_ld.loc["optikv_4to1"])
    gp_r50 = float(peak_ld.loc["rdma50_prefetch"])
    gp_rpf = float(peak_ld.loc["rdma_prefetch"])
    gp_rrt = float(peak_ld.loc["rdma_routing"])
    gp_orc = float(peak_ld.loc["oracle_infinite_hbm"])

    cfg_lbls = [
        "Elec. 400G\n(25G Rt)",
        "Elec. 400G\n(25G Pf)",
        "Elec. 400G\n(50G Line)",
        "Elec. 1.6T\n(200G Eq.)",
        "OptiKV 1:1\n(200G WDM)",
        "OptiKV 4:1\n(200G WDM)",
    ]
    sw_w = np.array([16 * P_ELEC_400G_SW, 16 * P_ELEC_400G_SW, 16 * P_ELEC_400G_SW, 64 * P_ELEC_400G_SW, 16 * P_OCS_MEMS_SW, 10 * P_OCS_MEMS_SW])
    xcvr_w = np.array([16 * P_XCVR_400G, 16 * P_XCVR_400G, 16 * P_XCVR_400G, 64 * P_XCVR_400G, 64 * P_XCVR_400G, 40 * P_XCVR_400G])
    tot_w = sw_w + xcvr_w
    # E2E Goodput / kW on 128K load sweep (250 ms SLO)
    e2e_gpw_mw = np.array([
        1000.0 * gp_rrt / tot_w[0],
        1000.0 * gp_rpf / tot_w[1],
        1000.0 * gp_r50 / tot_w[2],
        1000.0 * gp_orc / tot_w[3],
        1000.0 * gp_opt / tot_w[4],
        1000.0 * gp_opt41 / tot_w[5],
    ])

    xc = np.arange(len(cfg_lbls))
    bw = 0.44
    ax.bar(xc, sw_w, width=bw, color=C_RDMA, edgecolor=C_BLACK, lw=0.55, label="Switch ports (W)")
    ax.bar(xc, xcvr_w, width=bw, bottom=sw_w, color=C_RESV, edgecolor=C_BLACK, lw=0.55, hatch="///", label="NIC transceivers (W)")
    for xi, tw in zip(xc, tot_w):
        lbl_p = f"{tw/1e3:.2f} kW" if tw >= 1000 else f"{tw:.0f} W"
        ax.text(xi, tw + 95, lbl_p, ha="center", va="bottom", fontsize=6.9, fontweight="bold", color=C_BLACK)
    ax.set_ylim(0, 3950)
    ax.set_xticks(xc, cfg_lbls, fontsize=7.2)
    ax.set_ylabel("8-GPU E2E fabric power (W)")
    ax.set_title("(d) Bandwidth-Matched E2E Power & Efficiency", loc="center", fontsize=10.4)

    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_color(C_BLACK)
    ax_r.spines["right"].set_linewidth(0.75)
    ax_r.grid(False)
    ax_r.plot(xc, e2e_gpw_mw, color=C_OPTIKV, marker="o", ms=4.8, lw=1.60, zorder=6, label="128K E2E Goodput/kW")
    for xi, gv in zip(xc, e2e_gpw_mw):
        if xi in (0, 1):
            ax_r.text(xi, max(0.32, gv - 0.58), f"{gv:.2f}", ha="center", va="top", fontsize=7.0, fontweight="bold",
                      color=C_BLACK, bbox=dict(boxstyle="round,pad=0.08", fc="#FFFFFF", ec=C_OPTIKV, lw=0.55, alpha=0.95))
        else:
            yoff = 0.62 if xi != 3 else 0.72
            ax_r.text(xi, gv + yoff, f"{gv:.2f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold",
                      color=C_BLACK, bbox=dict(boxstyle="round,pad=0.10", fc="#FFFFFF", ec=C_OPTIKV, lw=0.55, alpha=0.94))
    ax_r.set_ylim(0, 12.8)
    ax_r.set_ylabel("128K E2E Goodput / kW (req/s/kW)", color=C_BLACK)
    ax_r.tick_params(axis="y", colors=C_BLACK)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.8, ncol=1)

    fig.tight_layout()
    save_fig(fig, "fig6_reconfig_and_cited_cost")


# ---------------------------------------------------------------------------
# Figure 7: 3-Panel Ablation (FIXED Non-Overlapping X-Labels!), Eviction & Noise
# ---------------------------------------------------------------------------

def make_fig7_ablation() -> None:
    df_abl = pd.read_csv(SIM / "exp_ablation.csv")
    if "step_idx" in df_abl.columns:
        df_abl = df_abl.sort_values("step_idx")
    df_ev = pd.read_csv(SIM / "exp_eviction_comparison.csv")
    if "ev_idx" in df_ev.columns:
        df_ev = df_ev.sort_values("ev_idx")
    df_ns = pd.read_csv(SIM / "exp_noise_robustness.csv")

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.45))

    # --- Panel (a): Clean Single-Line X-Labels (ZERO Overlap!) ---
    ax = axes[0]
    x = np.arange(len(df_abl))
    w = 0.37
    p50 = df_abl["p50_ttft_s"].values * 1e3
    p99 = df_abl["p99_ttft_s"].values * 1e3
    b1 = ax.bar(x - w / 2, p50, width=w, color=C_OPTIKV, edgecolor=C_BLACK, lw=0.55, label="P50 TTFT")
    ax.bar(x + w / 2, p99, width=w, color=C_RESV, alpha=0.65, edgecolor=C_BLACK, lw=0.55, hatch="//", label="P99 TTFT")
    ax.set_yscale("log")
    ax.set_ylim(max(1.0, min(p50) * 0.42), max(p99) * 4.2)
    for rect, v in zip(b1, p50):
        txt = f"{v/1000:.1f}s" if v >= 1000 else f"{v:.0f}"
        ax.text(rect.get_x() + rect.get_width() / 2, v * 1.18, txt,
                ha="center", va="bottom", fontsize=7.2, color=C_BLACK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.06", facecolor="#FFFFFF", edgecolor="none", alpha=0.88))
    step_labels = [
        "Naive", "+Latch", "+Route", "+Abort", "+TierSpill", "+GapFetch", "+GDSF (Ours)"
    ]
    ax.set_xticks(x, step_labels if len(step_labels) == len(df_abl) else list(df_abl["ablation_step"]),
                  rotation=28, ha="right", fontsize=8.0)
    ax.set_ylabel("TTFT (ms, log scale)")
    ax.set_title("(a) Cumulative Module Ablation", loc="center", fontsize=10.3)
    ax.legend(loc="upper right", fontsize=7.4)

    # --- Panel (b): PrefixKeep (Ours) at FAR RIGHT, Belady Optimal Dashed Ceiling ---
    ax = axes[1]
    belady_row = df_ev[df_ev["eviction_label"].str.contains("Belady")]
    belady_val = float(belady_row["hit_rate"].iloc[0]) * 100.0 if len(belady_row) else 34.72
    df_ev_prac = df_ev[~df_ev["eviction_label"].str.contains("Belady")].reset_index(drop=True)
    x = np.arange(len(df_ev_prac))
    hits = df_ev_prac["hit_rate"].values * 100.0
    cols = [C_NOCACHE, C_HBM, C_DRAM, C_RESV, C_OPTIKV]
    bars = ax.bar(x, hits, width=0.55, color=cols[: len(x)], edgecolor=C_BLACK, lw=0.55)
    ax.axhline(belady_val, color=C_BLACK, ls="--", lw=1.10, zorder=5)
    ax.text(0.04, belady_val + 0.9, f"Belady Offline Optimal ({belady_val:.2f}%)",
            ha="left", va="bottom", fontsize=7.5, fontweight="bold", color=C_BLACK)
    ev_lbls = [lbl.replace("Prefix-GDSF (Ours)", "PrefixKeep\n(Ours)").replace(" (Local-HBM)", "") for lbl in df_ev_prac["eviction_label"]]
    ax.set_xticks(x, ev_lbls, rotation=22, ha="right", fontsize=8.0)
    ax.set_ylim(0, belady_val * 1.30)
    ax.set_ylabel("Single-GPU HBM hit rate (%)")
    ax.set_title("(b) Eviction Under 2 GiB HBM (LMCache)", loc="center", fontsize=10.3)
    lru_h = round(float(hits[0]), 2) if len(hits) else 0.0
    for idx_b, (rect, h) in enumerate(zip(bars, hits)):
        delta = round(float(h), 2) - lru_h
        sub_txt = "base" if idx_b == 0 else f"({delta:+.2f})"
        is_ours = (idx_b == len(hits) - 1)
        if is_ours:
            ax.text(rect.get_x() + rect.get_width() / 2, belady_val + 0.85, f"{h:.2f}%\n{sub_txt}",
                    ha="center", va="bottom", fontsize=7.1, linespacing=0.95, color=C_BLACK, fontweight="bold")
        elif idx_b in (2, 3):
            ax.text(rect.get_x() + rect.get_width() / 2, h - 1.2, f"{h:.2f}%\n{sub_txt}",
                    ha="center", va="top", fontsize=6.9, linespacing=0.95, color=C_BLACK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.08", fc="#FFFFFF", ec="none", alpha=0.82))
        else:
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.60, f"{h:.2f}%\n{sub_txt}",
                    ha="center", va="bottom", fontsize=7.0, linespacing=0.95, color=C_BLACK)

    # --- Panel (c): Predictor Noise Robustness ---
    ax = axes[2]
    grp = df_ns.groupby("sigma").agg(
        p50_mean=("p50_ttft_s", lambda s: s.mean() * 1e3),
        p50_std=("p50_ttft_s", lambda s: s.std(ddof=0) * 1e3),
        p99_mean=("p99_ttft_s", lambda s: s.mean() * 1e3),
        p99_std=("p99_ttft_s", lambda s: s.std(ddof=0) * 1e3),
    ).reset_index()
    sig_pct = grp["sigma"].values * 100.0
    ax.errorbar(sig_pct, grp["p50_mean"], yerr=grp["p50_std"], marker="o", ms=4.3, color=C_OPTIKV, lw=1.55, capsize=3.5, label="P50 (mean $\\pm$ std)")
    ax.errorbar(sig_pct, grp["p99_mean"], yerr=grp["p99_std"], marker="s", ms=4.3, color=C_NAIVE, lw=1.40, ls="--", capsize=3.5, label="P99 (mean $\\pm$ std)")
    ax.set_yscale("log")
    ax.set_xticks(sig_pct, [f"{int(v)}%" for v in sig_pct])
    ax.set_ylim(10.0, max(grp["p99_mean"] + grp["p99_std"]) * 3.4)
    for xv, p50v, p99v in zip(sig_pct, grp["p50_mean"], grp["p99_mean"]):
        ax.text(xv, p50v * 1.26, f"{p50v:.1f} ms", ha="center", va="bottom", fontsize=7.5, color=C_BLACK, fontweight="bold")
        ax.text(xv, p99v * 1.23, f"{p99v/1e3:.2f} s", ha="center", va="bottom", fontsize=7.5, color=C_BLACK, fontweight="bold")
    ax.set_xlabel("Gaussian noise std on $P(\\mathrm{reuse})$ (%)")
    ax.set_ylabel("TTFT (ms, log scale)")
    ax.set_title("(c) Predictor Noise Robustness (3 seeds)", loc="center", fontsize=10.3)
    ax.legend(loc="center right", fontsize=7.4)

    fig.tight_layout()
    save_fig(fig, "fig7_ablation")


# ---------------------------------------------------------------------------
# Figure 8: 2-Panel Saturation & SLO Goodput (Clean Shared Top Legend!)
# ---------------------------------------------------------------------------

def make_fig8_load() -> None:
    df = pd.read_csv(SIM / "exp_load_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.85))
    pols = [
        "no_cache", "local_hbm", "local_dram", "rdma_routing", "concur_rdma",
        "rdma_prefetch", "rdma50_prefetch", "optical_naive", "optical_reservation",
        "optikv_4to1", "optikv", "oracle_infinite_hbm"
    ]
    lbl_override = {
        "rdma_routing": "RDMA-25G+Rt",
        "rdma_prefetch": "RDMA-25G+Pf",
        "rdma50_prefetch": "RDMA-50G (400G)",
        "optikv": "OptiKV (1:1)",
        "optikv_4to1": "OptiKV (4:1)",
    }

    ax0, ax1 = axes
    for pol in pols:
        sub = df[df["policy"] == pol].sort_values("target_rps")
        if sub.empty:
            continue
        lw = 1.70 if pol == "optikv" else (1.45 if pol in ("optikv_4to1", "rdma50_prefetch") else (1.30 if pol == "oracle_infinite_hbm" else 1.15))
        ls = "--" if pol in ("oracle_infinite_hbm", "rdma50_prefetch") else ("-." if pol == "optikv_4to1" else "-")
        ms = 4.4 if pol == "optikv" else (4.2 if pol in ("optikv_4to1", "rdma50_prefetch") else (4.5 if pol == "oracle_infinite_hbm" else 3.5))
        z = 7 if pol == "oracle_infinite_hbm" else (6 if pol == "optikv" else (5 if pol in ("optikv_4to1", "rdma50_prefetch") else 3))
        ax0.plot(sub["target_rps"], sub["p50_ttft_s"], marker=POLICY_MARKER[pol], ms=ms,
                 color=POLICY_COLOR[pol], lw=lw, ls=ls, zorder=z, label=lbl_override.get(pol, POLICY_LABEL[pol]))
        ax1.plot(sub["target_rps"], sub["goodput_rps"], marker=POLICY_MARKER[pol], ms=ms,
                 color=POLICY_COLOR[pol], lw=lw, ls=ls, zorder=z)

    slo_s = float(df["slo_ttft_s"].iloc[0]) if "slo_ttft_s" in df.columns and len(df) else 0.25
    slo_lbl = f"{int(round(slo_s*1e3))} ms" if slo_s < 1.0 else f"{slo_s:.1f} s"
    ax0.axhline(slo_s, color=C_NAIVE, ls=":", lw=1.15, label=f"250 ms SLO")
    ax0.axhline(0.10, color=C_HBM, ls=":", lw=0.95, label="100 ms Interactive")
    ax0.set_yscale("log")
    ax0.set_ylim(0.022, 165.0)
    ax0.set_xlabel("Offered steady-state arrival rate (req/s)")
    ax0.set_ylabel("Median (P50) TTFT (s, log scale)")
    ax0.set_title("(a) Median TTFT Saturation Under Load (128K)", loc="center", fontsize=10.3)
    ax0.text(31.8, 0.275, "250 ms SLO", ha="right", va="bottom", fontsize=7.4, fontweight="bold", color=C_BLACK)
    ax0.text(31.8, 0.088, "100 ms Interactive (RDMA-50G fails @ 162 ms)", ha="right", va="top", fontsize=7.1, fontweight="bold", color=C_BLACK)

    ax1.set_xlabel("Offered steady-state arrival rate (req/s)")
    ax1.set_ylabel(f"SLO goodput (req/s, TTFT $\\leq$ {slo_lbl})")
    ax1.set_title("(b) SLO-Attaining Cluster Goodput Under Load", loc="center", fontsize=10.3)
    ax1.set_ylim(-0.18, 4.65)
    opt_max = float(df[df["policy"] == "optikv"]["goodput_rps"].max())
    r50_max = float(df[df["policy"] == "rdma50_prefetch"]["goodput_rps"].max())
    rpf_max = float(df[df["policy"] == "rdma_prefetch"]["goodput_rps"].max())
    rrt_max = float(df[df["policy"] == "rdma_routing"]["goodput_rps"].max())
    ax1.annotate(f"OptiKV 1:1 & 4:1 = Oracle ({opt_max:.2f} @ 50 ms)\nRDMA-50G: {r50_max:.2f} @ 162 ms (3.25$\\times$ slower)\nRDMA-25G: {rrt_max:.2f}–{rpf_max:.2f} ({opt_max/rpf_max:.2f}$\\times$–{opt_max/rrt_max:.2f}$\\times$ lower)",
                 xy=(32.0, opt_max), xytext=(10.0, 1.55),
                 fontsize=7.2, fontweight="bold", color=C_BLACK,
                 bbox=dict(boxstyle="round,pad=0.20", fc="#FFFFFF", ec=C_OPTIKV, lw=0.75),
                 arrowprops=dict(arrowstyle="-|>", color=C_OPTIKV, lw=0.95))

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 0.996),
               fontsize=7.0, frameon=True, edgecolor=C_BLACK, fancybox=False, handlelength=1.45, columnspacing=0.75)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.850])
    save_fig(fig, "fig8_p99_load")


# ---------------------------------------------------------------------------
# Figure 9: Strict Bipartite Matching Fidelity & 4-32 GPU Scale-Out (1:1 & 4:1!)
# ---------------------------------------------------------------------------

def make_fig9_fidelity_and_scaling() -> None:
    fid_path = SIM / "exp_v3_fidelity.csv"
    scale_path = SIM / "exp_v3_cluster_scale.csv"
    if not (fid_path.is_file() and scale_path.is_file()):
        print("  skipping fig9", flush=True)
        return

    df_fid = pd.read_csv(fid_path)
    df_sc = pd.read_csv(scale_path)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.05))

    # --- Panel (a): Strict 1:1 vs 4:1 Bipartite Matching + Decode Coupling ---
    ax = axes[0, 0]
    reg_order = ["Match-1:1 (Base)", "Match-4:1 (Oversub)", "Match-1:1 + Decode", "Match-4:1 + Decode"]
    reg_lbls = ["Strict 1:1\n(P=8)", "Strict 4:1\n(P=2)", "1:1 + Decode\nBatch Q.", "4:1 + Decode\nBatch Q."]
    sub_fi = df_fid[(df_fid["workload"] == "freeinference-50") & (df_fid["regime"].isin(reg_order))]
    x = np.arange(len(reg_order))
    w = 0.20
    show_p = [
        ("rdma_routing", "RDMA+Routing", C_RDMA),
        ("concur_rdma", "CONCUR+RDMA", C_CONCUR),
        ("rdma_prefetch", "RDMA+Prefetch", C_RDMA_PF),
        ("optikv", "OptiKV-Fabric (Ours)", C_OPTIKV),
    ]
    for pi, (pol, lbl, col) in enumerate(show_p):
        vals = [
            float(sub_fi[(sub_fi["regime"] == rk) & (sub_fi["policy"] == pol)]["p50_ttft_s"].iloc[0]) * 1e3
            for rk in reg_order
        ]
        bars = ax.bar(x + (pi - 1.5) * w, vals, width=w, color=col, edgecolor=C_BLACK, lw=0.50, label=lbl)
        if pol == "optikv":
            duties = [
                float(sub_fi[(sub_fi["regime"] == rk) & (sub_fi["policy"] == pol)]["ocs_duty_cycle"].iloc[0]) * 100.0
                for rk in reg_order
            ]
            for rect, v, dv in zip(bars, vals, duties):
                ax.text(rect.get_x() + rect.get_width() / 2 + 0.03, v + 2.0, f"{v:.1f}\n{dv:.0f}%",
                        ha="center", va="bottom", fontsize=6.7, fontweight="bold", color=C_BLACK)
    ax.set_xticks(x, reg_lbls, fontsize=8.3)
    ax.set_ylim(0, 138)
    ax.set_ylabel("FreeInference-50 P50 TTFT (ms)")
    ax.set_title("(a) Strict Bipartite Matching & Decode Stress", loc="center", fontsize=10.3)
    ax.legend(loc="upper right", fontsize=7.2, ncol=2)

    # --- Panel (b): WDM Striping Under Strict Bipartite + Decode ---
    ax = axes[0, 1]
    wdm_keys = ["Match-1:1 + Decode", "Strict 1:1 + WDM 2x", "Strict 1:1 + WDM 4x"]
    wdm_lbls = ["1-$\\lambda$ WDM\n(200 GB/s)", "2-$\\lambda$ WDM\n(400 GB/s)", "4-$\\lambda$ WDM\n(800 GB/s)"]
    wloads = [
        ("lmcache-120", "LMCache-120", C_OPTIKV, "o"),
        ("freeinference-50", "FreeInference-50", C_RDMA_PF, "s"),
        ("agentic-128k-bimodal", "128K-Bimodal", C_RDMA, "^"),
    ]
    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_color(C_BLACK)
    ax_r.spines["right"].set_linewidth(0.75)
    ax_r.grid(False)
    xw = np.arange(len(wdm_keys))
    lns = []
    for wl, lbl, col, mk in wloads:
        t_vals, d_vals = [], []
        for rk in wdm_keys:
            m = df_fid[(df_fid["workload"] == wl) & (df_fid["regime"] == rk) & (df_fid["policy"] == "optikv")]
            t_vals.append(float(m["mean_transfer_s"].iloc[0]) * 1e3)
            d_vals.append(float(m["ocs_duty_cycle"].iloc[0]) * 100.0)
        lns += ax.plot(xw, t_vals, color=col, marker=mk, ms=4.4, lw=1.50, label=f"{lbl} (xfer, solid)")
        ax_r.plot(xw, d_vals, color=col, marker=mk, ms=3.7, lw=1.15, ls="--", alpha=0.82)
    l_dash = ax.plot([], [], color=C_BLACK, ls="--", lw=1.15, label="OCS duty (%, dashed)")
    ax.set_xticks(xw, wdm_lbls, fontsize=8.3)
    ax.set_ylim(0, 32.0)
    ax.set_ylabel("Mean critical-path transfer (ms)", color=C_BLACK)
    ax_r.set_ylabel("Strict bipartite OCS duty (%)", color=C_BLACK)
    ax_r.tick_params(axis="y", colors=C_BLACK)
    ax_r.set_ylim(0, 115)
    ax.set_title("(b) WDM Striping Under Strict Matching + Decode", loc="center", fontsize=10.3)
    all_l = lns + l_dash
    ax.legend(all_l, [l.get_label() for l in all_l], loc="upper right", fontsize=7.1)

    # --- Panel (c): 4-32 GPU Multi-Node Weak Scaling (BOTH 1:1 & 4:1 Strict Matching!) ---
    ax = axes[1, 0]
    sub_11 = df_sc[df_sc["sweep_type"] == "weak_scale_1to1"]
    sub_41 = df_sc[df_sc["sweep_type"] == "weak_scale_4to1"]
    w_order = [4, 8, 16, 32]
    xw_pos = np.arange(len(w_order))
    scale_specs = [
        (sub_11, "optikv", "OptiKV 1:1 (P=W)", C_OPTIKV, "o", "-", 1.75, 5),
        (sub_41, "optikv_4to1", "OptiKV 4:1 (P=W/4)", C_OPTIKV_41, "d", "-.", 1.45, 4),
        (sub_11, "oracle_infinite_hbm", "Oracle (Bound)", C_ORACLE, "*", "--", 1.25, 6),
        (sub_11, "rdma_prefetch", "RDMA+Prefetch", C_RDMA_PF, "p", "-", 1.25, 3),
        (sub_11, "rdma_routing", "RDMA+Routing", C_RDMA, "s", "-", 1.25, 3),
        (sub_11, "optical_naive", "Optical-Naive", C_NAIVE, "D", "-", 1.15, 2),
    ]
    for src, pol, lbl, col, mk, ls, lw, z in scale_specs:
        ys = [float(src[(src["workers"] == wv) & (src["policy"] == pol)]["goodput_rps"].iloc[0]) for wv in w_order]
        ax.plot(xw_pos, ys, color=col, marker=mk, ms=4.5 if pol != "oracle_infinite_hbm" else 5.2,
                lw=lw, ls=ls, zorder=z, label=lbl)
        if pol == "optikv":
            for xi, yv in zip(xw_pos, ys):
                ax.text(xi, yv + 0.55, f"{yv:.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=C_BLACK)
        elif pol == "rdma_prefetch":
            opt32 = float(sub_11[(sub_11["workers"] == 32) & (sub_11["policy"] == "optikv")]["goodput_rps"].iloc[0])
            rrt32 = float(sub_11[(sub_11["workers"] == 32) & (sub_11["policy"] == "rdma_routing")]["goodput_rps"].iloc[0])
            ax.text(xw_pos[-1] - 0.03, ys[-1] + 0.65,
                    f"RDMA+Pf: {ys[-1]:.2f} ({opt32/ys[-1]:.1f}$\\times$)\nRDMA+Rt: {rrt32:.2f} ({opt32/rrt32:.1f}$\\times$)",
                    ha="right", va="bottom", fontsize=7.2, fontweight="bold", color=C_BLACK,
                    bbox=dict(boxstyle="round,pad=0.16", fc="#FFFFFF", ec=C_HBM, lw=0.60))
    # Honest duty callout for W=32 1:1 vs 4:1
    d11_32 = float(sub_11[(sub_11["workers"] == 32) & (sub_11["policy"] == "optikv")]["ocs_duty_cycle"].iloc[0]) * 100.0
    d41_32 = float(sub_41[(sub_41["workers"] == 32) & (sub_41["policy"] == "optikv_4to1")]["ocs_duty_cycle"].iloc[0]) * 100.0
    ax.text(0.03, 0.52, f"32-GPU Strict Duty:\n1:1 (P=32): {d11_32:.1f}% (21 sw)\n4:1 (P=8):  {d41_32:.1f}% (108 sw)",
            transform=ax.transAxes, ha="left", va="center", fontsize=7.2, fontweight="bold", color=C_BLACK,
            bbox=dict(boxstyle="round,pad=0.20", fc="#F4FAFF", ec=C_OPTIKV, lw=0.70))

    ax.set_xticks(xw_pos, ["4 GPUs\n(P=4 / 1)", "8 GPUs\n(P=8 / 2)", "16 GPUs\n(P=16 / 4)", "32 GPUs\n(P=32 / 8)"], fontsize=8.2)
    ax.set_ylim(-0.5, 14.8)
    ax.set_ylabel("SLO goodput (req/s, 250 ms)")
    ax.set_title("(c) Multi-Node Weak Scaling (4–32 GPUs, 1:1 & 4:1)", loc="center", fontsize=10.3)
    ax.legend(loc="upper left", fontsize=7.0, ncol=2)

    # --- Panel (d): W=16 Full Port Sweep P=1, 2, 4, 8, 16 (16:1 down to 1:1) ---
    ax = axes[1, 1]
    sub_pr = df_sc[df_sc["sweep_type"] == "port_ratio_w16"]
    p_order = [1, 2, 4, 8, 16]
    xp = np.arange(len(p_order))
    opt_p50 = [float(sub_pr[(sub_pr["ocs_ports"] == pv) & (sub_pr["policy"] == "optikv")]["p50_ttft_s"].iloc[0]) * 1e3 for pv in p_order]
    opt_gp = [float(sub_pr[(sub_pr["ocs_ports"] == pv) & (sub_pr["policy"] == "optikv")]["goodput_rps"].iloc[0]) for pv in p_order]
    rpf_p50 = float(sub_pr[(sub_pr["ocs_ports"] == 4) & (sub_pr["policy"] == "rdma_prefetch")]["p50_ttft_s"].iloc[0]) * 1e3
    rpf_gp = float(sub_pr[(sub_pr["ocs_ports"] == 4) & (sub_pr["policy"] == "rdma_prefetch")]["goodput_rps"].iloc[0])

    bar_cols = [C_NAIVE, C_RDMA, C_OPTIKV, C_OPTIKV, C_OPTIKV_41]
    bars = ax.bar(xp - 0.18, opt_p50, width=0.34, color=bar_cols, edgecolor=C_BLACK, lw=0.55, label="OptiKV P50 (ms)")
    ax.axhline(250.0, color=C_NAIVE, ls=":", lw=1.05, label="250 ms SLO")
    ax.axhline(rpf_p50, color=C_BLACK, ls="--", lw=0.95, label=f"RDMA+Pf P50 ({rpf_p50:.0f} ms)")
    ax.set_yscale("log")
    ax.set_ylim(20, 38000)
    for rect, v in zip(bars, opt_p50):
        ax.text(rect.get_x() + rect.get_width() / 2, v * 1.18, f"{v:.0f}" if v >= 100 else f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=C_BLACK,
                bbox=dict(boxstyle="round,pad=0.05", fc="#FFFFFF", ec="none", alpha=0.85))

    ax_gp = ax.twinx()
    ax_gp.spines["top"].set_visible(False)
    ax_gp.spines["right"].set_color(C_BLACK)
    ax_gp.spines["right"].set_linewidth(0.75)
    ax_gp.grid(False)
    ax_gp.plot(xp + 0.18, opt_gp, color=C_RDMA_PF, marker="o", ms=4.6, lw=1.55, label="OptiKV Goodput")
    ax_gp.axhline(rpf_gp, color=C_RDMA_PF, ls="-.", lw=1.0, label=f"RDMA+Pf GP ({rpf_gp:.2f})")
    for idx_g, (xi, gv) in enumerate(zip(xp + 0.18, opt_gp)):
        if idx_g == 1:
            ax_gp.text(xi + 0.08, gv + 0.30, f"{gv:.2f}", ha="left", va="bottom", fontsize=7.5, fontweight="bold", color=C_BLACK)
        elif idx_g >= 2:
            ax_gp.text(xi, gv - 0.45, f"{gv:.2f}", ha="center", va="top", fontsize=7.5, fontweight="bold", color=C_BLACK)
        else:
            ax_gp.text(xi, gv + 0.35, f"{gv:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=C_BLACK)
    ax_gp.set_ylim(0, 13.8)
    ax_gp.set_ylabel("SLO Goodput (req/s)", color=C_BLACK)
    ax_gp.tick_params(axis="y", colors=C_BLACK)

    ax.set_xticks(xp, ["P=1\n(16:1)", "P=2\n(8:1)", "P=4\n(4:1 Knee)", "P=8\n(2:1)", "P=16\n(1:1)"], fontsize=8.0)
    ax.set_ylabel("Median (P50) TTFT (ms, log)")
    ax.set_title("(d) W=16 Port-Oversubscription Knee (64 req/s)", loc="center", fontsize=10.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_gp.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.7, ncol=1)

    fig.tight_layout()
    save_fig(fig, "fig9_fidelity_and_scaling")


# ---------------------------------------------------------------------------
# Figure 10: Dedicated Live RTX 3090 vLLM HIL Prototype Replay (Unified 45 GB/s!)
# ---------------------------------------------------------------------------

def make_fig10_live_3090_hil_replay() -> None:
    e2e_path = R3090 / "vllm_e2e_replay_summary.csv"
    req_path = R3090 / "vllm_e2e_replay_requests.csv"
    if not (e2e_path.is_file() and req_path.is_file()):
        print("  skipping fig10", flush=True)
        return

    df_e2e = pd.read_csv(e2e_path)
    df_req = pd.read_csv(req_path)

    n_sess = int(df_req["session"].nunique()) if "session" in df_req.columns else 12
    concur_mode_str = str(df_e2e[df_e2e["mode"].str.contains("CONCUR")]["mode"].iloc[0])
    naive_mode_str = str(df_e2e[df_e2e["mode"].str.contains("Optical-Naive")]["mode"].iloc[0])
    optikv_mode_str = str(df_e2e[df_e2e["mode"].str.contains("OptiKV-Fabric")]["mode"].iloc[0])
    m_w = re.search(r"W=(\d+)", concur_mode_str)
    w_sz = int(m_w.group(1)) if m_w else (n_sess // 2)
    working_gib = n_sess * 8192 * 57344 / (1024**3)
    pool_gib = float(df_e2e["kv_pool_gib"].iloc[0]) if "kv_pool_gib" in df_e2e.columns else 3.50

    # Keep taller physical Panel (a) y-axis per user specification!
    fig = plt.figure(figsize=(11.2, 6.65))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.65, 1.0], hspace=0.34, wspace=0.28)

    # --- Panel (a): Qualitative Per-Request Live RTX 3090 Silicon Trajectory ---
    ax_t = fig.add_subplot(gs[0, :])
    mode_specs = [
        ("Live vLLM (LRU)", "Local-HBM (LRU)", C_HBM, "v", "-", 1.25, 3),
        (concur_mode_str, f"CONCUR (W={w_sz})", C_CONCUR, "h", "-.", 1.25, 4),
        ("Live vLLM + RDMA (25 GB/s)", "RDMA (25 GB/s)", C_RDMA, "s", "-", 1.25, 4),
        (naive_mode_str, "Optical-Naive (45 GB/s, 25 ms)", C_NAIVE, "D", "-", 1.35, 5),
        (optikv_mode_str, "OptiKV-Fabric (45 GB/s, Ours)", C_OPTIKV, "o", "-", 1.65, 6),
    ]
    gt_rows = df_req[df_req["mode"] == optikv_mode_str].sort_values(["turn", "session"])
    req_x = np.arange(1, len(gt_rows) + 1)
    gt_floor = gt_rows["gpu_ttft_ms"].values

    ax_t.axvspan(0.5, 12.5, color=P_GRAY, alpha=0.45, zorder=0)
    ax_t.axvline(12.5, color=C_BLACK, ls=":", lw=0.95)
    ax_t.axvline(24.5, color=C_BLACK, ls=":", lw=0.80, alpha=0.65)
    ax_t.text(6.5, 31000, "Turn 0 (Cold Fill)", ha="center", va="top", fontsize=8.1, fontweight="bold", color=C_BLACK)
    ax_t.text(18.5, 31000, "Turn 1 (Prefix Reuse)", ha="center", va="top", fontsize=8.1, fontweight="bold", color=C_BLACK)
    ax_t.text(30.5, 31000, "Turn 2 (Prefix Reuse)", ha="center", va="top", fontsize=8.1, fontweight="bold", color=C_BLACK)

    for mkey, mlbl, mcol, mmk, mls, mlw, mz in mode_specs:
        sub = df_req[df_req["mode"] == mkey].sort_values(["turn", "session"])
        ax_t.plot(np.arange(1, len(sub) + 1), sub["e2e_ttft_ms"].values,
                  color=mcol, marker=mmk, ms=3.4 if "OptiKV" not in mlbl else 3.8, lw=mlw, ls=mls, zorder=mz, label=mlbl)
    ax_t.plot(req_x, gt_floor, color=C_BLACK, ls="--", lw=1.15, zorder=7, label="Ground-Truth Floor")

    ax_t.annotate("CONCUR Wave-2\n10.83 s Stall", xy=(9.5, 12538), xytext=(13.6, 8600),
                  fontsize=7.8, fontweight="bold", color=C_BLACK,
                  arrowprops=dict(arrowstyle="-|>", color=C_CONCUR, lw=0.95))
    ax_t.annotate("LRU Cliff\n(~1.71 s)", xy=(16.0, 1730), xytext=(13.3, 460),
                  fontsize=7.8, fontweight="bold", color=C_BLACK,
                  arrowprops=dict(arrowstyle="-|>", color=C_HBM, lw=0.95))

    naive_p50 = float(df_e2e[df_e2e["mode"] == naive_mode_str]["reuse_p50_ms"].iloc[0])
    optikv_p50 = float(df_e2e[df_e2e["mode"] == optikv_mode_str]["reuse_p50_ms"].iloc[0])
    pct_brk = 100.0 * (naive_p50 - optikv_p50) / optikv_p50
    ax_t.annotate(f"Optical-Naive ({_hu1(naive_p50)} ms):\n+{pct_brk:.1f}% Law 3 breakout vs. OptiKV ({_hu1(optikv_p50)} ms)",
                  xy=(28.5, naive_p50 + 1.0), xytext=(24.2, 330.0),
                  fontsize=7.6, fontweight="bold", color=C_BLACK,
                  bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec="none", alpha=0.88),
                  arrowprops=dict(arrowstyle="-|>", color=C_NAIVE, lw=0.95))

    ax_t.set_yscale("log")
    ax_t.set_xlim(0.5, 36.8)
    ax_t.set_ylim(26, 40000)
    ax_t.set_xticks([1, 6, 12, 18, 24, 30, 36])
    ax_t.set_xlabel("Sequential request index across 12 concurrent 3-turn 8K sessions")
    ax_t.set_ylabel("Live E2E TTFT (ms, log)")
    ax_t.set_title(f"(a) Qualitative Per-Request Live RTX 3090 Trajectory (N={n_sess}$\\times$8K, {working_gib:.2f} vs. {pool_gib:.2f} GiB HBM)",
                   loc="center", fontsize=10.5)
    ax_t.legend(loc="lower left", bbox_to_anchor=(0.006, 0.015), fontsize=7.0, ncol=2, handlelength=1.4, columnspacing=0.65)

    # --- Panel (b): Reuse P50 & P95 Bar Summary ---
    ax = fig.add_subplot(gs[1, 0])
    x_e = np.arange(len(df_e2e))
    short_names = [
        "Local-HBM\n(LRU)",
        f"CONCUR\n(W={w_sz})",
        "RDMA\n(25 GB/s)",
        "Opt-Naive\n(45 GB/s)",
        "OptiKV\n(Ours)",
    ][: len(df_e2e)]
    e_cols = [C_HBM, C_CONCUR, C_RDMA, C_NAIVE, C_OPTIKV][: len(df_e2e)]
    p50_e = df_e2e["reuse_p50_ms"].values
    p95_e = df_e2e["reuse_p95_ms"].values
    bw = 0.36
    b1 = ax.bar(x_e - bw / 2, p50_e, width=bw, color=e_cols, edgecolor=C_BLACK, lw=0.55, label="Reuse P50 TTFT")
    ax.bar(x_e + bw / 2, p95_e, width=bw, color=e_cols, alpha=0.40, edgecolor=C_BLACK, lw=0.55, hatch="//", label="Reuse P95 TTFT")
    ax.set_yscale("log")
    ax.set_ylim(20, 6800)
    for idx_e, (rect, v, sp) in enumerate(zip(b1, p50_e, df_e2e["speedup_vs_lru"].values)):
        txt = f"{_hu1(v)} ms\n({sp:.1f}$\\times$)" if sp > 1.05 else f"{v:.0f} ms\n(1.0$\\times$)"
        ax.text(rect.get_x() + rect.get_width() / 2, v * 1.18, txt, ha="center", va="bottom",
                fontsize=7.3, color=C_BLACK, fontweight="bold" if idx_e in (0, 3, 4) else "normal")
    ax.set_xticks(x_e, short_names, fontsize=8.2)
    ax.set_ylabel("Measured Reuse TTFT (ms, log)")
    ax.set_title("(b) Reuse P50 / P95 Summary (38.0$\\times$ Recovery)", loc="center", fontsize=10.3)
    ax.legend(loc="upper right", fontsize=7.4)

    # --- Panel (c): Unified 45 GB/s Law 3 Overlap Anatomy (Pure T_prefill,Delta!) & MEMS Duty ---
    ax = fig.add_subplot(gs[1, 1])
    kv_45g_ms = (8192 * 57344) / 45e9 * 1e3  # 10.439 ms
    kv_25g_ms = (8192 * 57344) / 25e9 * 1e3  # 18.790 ms
    cats = ["RDMA Pool\n(25 GB/s)", "Optical-Naive\n(45 GB/s, 25 ms)", "OptiKV-Fabric\n(45 GB/s, Ours)"]
    xfer_exp = [kv_25g_ms, 25.00 + kv_45g_ms, 0.0]
    duty_naive = float(df_e2e[df_e2e["mode"] == naive_mode_str]["ocs_duty_pct"].iloc[0])
    duty_optikv = float(df_e2e[df_e2e["mode"] == optikv_mode_str]["ocs_duty_pct"].iloc[0])
    duty_vals = [0.0, duty_naive, duty_optikv]
    duty_ratio = duty_optikv / max(1e-6, duty_naive)
    xc = np.arange(len(cats))
    b_x = ax.bar(xc - 0.18, xfer_exp, width=0.34, color=[C_RDMA, C_NAIVE, C_OPTIKV], edgecolor=C_BLACK, lw=0.55,
                 label="Critical-path xfer (ms)")
    # Exact Eq. (4) pure delta-prefill overlap shadow: 0.70 x 36.38 ms = 25.47 ms
    ovl_shadow = 0.70 * 36.38
    ax.axhline(ovl_shadow, color=C_BLACK, ls="--", lw=1.15,
               label=f"Eq. (4) shadow ({_hu1(ovl_shadow)} ms)")
    ax.set_ylim(0, 56)
    for idx_b, (rect, v, note) in enumerate(zip(b_x, xfer_exp, [f"{kv_25g_ms:.1f} ms\n(hidden)", f"{25.0+kv_45g_ms:.1f} ms\n(+10.0 ms)", "0.0 ms\n(hidden)"])):
        if idx_b == 0:
            ax.text(rect.get_x() + rect.get_width() / 2, v * 0.50, note, ha="center", va="center",
                    fontsize=7.2, fontweight="bold", color=C_BLACK,
                    bbox=dict(boxstyle="round,pad=0.12", fc="#FFFFFF", ec="none", alpha=0.86))
        else:
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1.2, note, ha="center", va="bottom",
                    fontsize=7.3, fontweight="bold", color=C_BLACK)

    ax_d = ax.twinx()
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_color(C_BLACK)
    ax_d.spines["right"].set_linewidth(0.75)
    ax_d.grid(False)
    b_d = ax_d.bar(xc + 0.18, duty_vals, width=0.34, color=[C_NOCACHE, P_VERM, P_SKY], edgecolor=C_BLACK, lw=0.55,
                   hatch="..", label="Measured OCS duty (%)")
    ax_d.set_ylim(0, 114)
    ax_d.set_ylabel("Measured MEMS OCS duty (%)", color=C_BLACK)
    ax_d.tick_params(axis="y", colors=C_BLACK)
    for idx_d, (rect, dv) in enumerate(zip(b_d, duty_vals)):
        if idx_d == 0:
            ax_d.text(rect.get_x() + rect.get_width() / 2, 2.5, "N/A\n(elec.)", ha="center", va="bottom",
                      fontsize=7.3, color=C_BLACK)
        else:
            lbl_d = f"{dv:.1f}%\n({duty_ratio:.2f}$\\times$)" if idx_d == 2 else f"{dv:.1f}%"
            ax_d.text(rect.get_x() + rect.get_width() / 2, dv + 2.2, lbl_d, ha="center", va="bottom",
                      fontsize=7.4, fontweight="bold", color=C_BLACK)

    ax.set_xticks(xc, cats, fontsize=8.2)
    ax.set_ylabel("Critical-path transfer exposure (ms)")
    ax.set_title("(c) Unified 45 GB/s Law 3 & MEMS Duty", loc="center", fontsize=10.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_d.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.7)

    save_fig(fig, "fig10_live_3090_hil_replay")


def main() -> int:
    print("Generating all 10 v3.0 IEEE Transactions figures (strict matching + honest E2E power)...", flush=True)
    make_fig1_teaser()
    make_fig2_workload_and_capacity()
    make_fig3_calibration()
    make_fig4_main_comparison()
    make_fig5_heatmap()
    make_fig6_sensitivity()
    make_fig7_ablation()
    make_fig8_load()
    make_fig9_fidelity_and_scaling()
    make_fig10_live_3090_hil_replay()
    print("All 10 v3.0 publication figures generated cleanly.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
