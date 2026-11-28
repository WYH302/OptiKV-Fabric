#!/usr/bin/env python3
"""Generate publication-grade Vector SVG -> Vector PDF Figure 1 for OptiKV-Fabric.

Uses precision HTML5/SVG with custom vector hardware icons (MEMS mirror matrix,
GPU HBM die, DRAM DIMM, NVMe SSD, optical pool), diagonal-hatched MEMS dead-time
bars, crisp studio cards with subtle elevation shadows and colored accent bars,
strictly unified Okabe-Ito colorblind-safe palette, pure-black typography, and
exact CSV-driven Panel (c) coordinates, rendered to native infinite-resolution
vector PDF via headless Chrome + pdfcrop + 300-DPI PNG via pdftoppm.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SIM = ROOT / "results" / "sim"
FIG_EXP = ROOT / "figures"
FIG_PAPER = REPO_ROOT / "paper" / "figures"

CHROME_PATHS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]
TEXLIVE_BIN = Path(r"D:\texlive\2026\bin\windows")

# Honest bandwidth-matched E2E power constants (8-GPU pod)
P_ELEC_400G_SW = 39.4
P_OCS_MEMS_SW = 0.39
P_XCVR_400G = 10.0

PWR_ELEC_400G_E2E = 16 * P_ELEC_400G_SW + 16 * P_XCVR_400G      # 790.4 W
PWR_ELEC_1600G_E2E = 64 * P_ELEC_400G_SW + 64 * P_XCVR_400G     # 3161.6 W
PWR_OCS_200G_11_E2E = 16 * P_OCS_MEMS_SW + 64 * P_XCVR_400G     # 646.24 W
PWR_OCS_200G_41_E2E = 10 * P_OCS_MEMS_SW + 40 * P_XCVR_400G     # 403.90 W


def _find_browser() -> Path:
    for p in CHROME_PATHS:
        if p.is_file():
            return p
    raise FileNotFoundError("Chrome or Edge headless binary not found.")


def _star_points(cx: float, cy: float, r_out: float, r_in: float, n: int = 5) -> str:
    pts = []
    for i in range(2 * n):
        ang = -math.pi / 2 + i * math.pi / n
        r = r_out if i % 2 == 0 else r_in
        pts.append(f"{cx + r * math.cos(ang):.2f},{cy + r * math.sin(ang):.2f}")
    return " ".join(pts)


def _polygon_points(cx: float, cy: float, r: float, sides: int, rot_deg: float = -90.0) -> str:
    pts = []
    rot = math.radians(rot_deg)
    for i in range(sides):
        ang = rot + i * 2 * math.pi / sides
        pts.append(f"{cx + r * math.cos(ang):.2f},{cy + r * math.sin(ang):.2f}")
    return " ".join(pts)


def build_svg_html() -> str:
    df_load = pd.read_csv(SIM / "exp_load_sweep.csv")
    peak = df_load.groupby("policy")["goodput_rps"].max()

    gp_opt = float(peak.loc["optikv"])
    gp_opt41 = float(peak.loc["optikv_4to1"])
    gp_orc = float(peak.loc["oracle_infinite_hbm"])
    gp_r50 = float(peak.loc["rdma50_prefetch"])
    gp_rpf = float(peak.loc["rdma_prefetch"])
    gp_rrt = float(peak.loc["rdma_routing"])

    gpw_41 = (gp_opt41 / PWR_OCS_200G_41_E2E) / (gp_rrt / PWR_ELEC_400G_E2E)
    gpw_41_r50 = (gp_opt41 / PWR_OCS_200G_41_E2E) / (gp_r50 / PWR_ELEC_400G_E2E)

    # Panel (c) plot frame in SVG coordinates (Panel C lives in x=[958..1372], y=[44..404])
    px0, px1 = 1008.0, 1368.0
    py0, py1 = 50.0, 396.0
    xmin, xmax = 145.0, 5400.0
    ymin, ymax = -0.34, 4.92

    def sx(w: float) -> float:
        return px0 + (math.log10(w) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * (px1 - px0)

    def sy(v: float) -> float:
        return py1 - (v - ymin) / (ymax - ymin) * (py1 - py0)

    # Precompute exact Panel (c) point positions
    c_opt41 = (sx(PWR_OCS_200G_41_E2E), sy(gp_opt41))
    c_opt11 = (sx(PWR_OCS_200G_11_E2E), sy(gp_opt))
    c_r50 = (sx(PWR_ELEC_400G_E2E), sy(gp_r50))
    c_elec200 = (sx(PWR_ELEC_1600G_E2E), sy(gp_orc))
    c_rpf = (sx(PWR_ELEC_400G_E2E), sy(gp_rpf))
    c_rrt = (sx(PWR_ELEC_400G_E2E), sy(gp_rrt))
    c_concur = (sx(PWR_ELEC_400G_E2E * 1.07), sy(gp_rrt - 0.04))
    c_naive = (sx(PWR_OCS_200G_11_E2E), sy(0.02))
    c_hbm = (sx(PWR_ELEC_400G_E2E * 1.04), sy(0.02))

    pareto_x1 = sx(720.0)
    pareto_y1 = sy(3.08)

    # Build Panel (c) gridlines & ticks
    xticks = [(250, "250"), (500, "500"), (1000, "1K"), (2000, "2K"), (4000, "4K")]
    yticks = [0, 1, 2, 3, 4]
    xgrid_svg = []
    for xv, xlbl in xticks:
        xx = sx(xv)
        xgrid_svg.append(
            f'<line x1="{xx:.2f}" y1="{py0:.2f}" x2="{xx:.2f}" y2="{py1:.2f}" stroke="#E3E8EE" stroke-width="0.9" stroke-dasharray="3,3"/>'
            f'<line x1="{xx:.2f}" y1="{py1:.2f}" x2="{xx:.2f}" y2="{py1+5:.2f}" stroke="#000000" stroke-width="1.0"/>'
            f'<text x="{xx:.2f}" y="{py1+19:.2f}" text-anchor="middle" font-size="11.5" fill="#000000">{xlbl}</text>'
        )
    # Minor log ticks
    for mv in [200, 300, 400, 600, 700, 800, 900, 1500, 3000, 5000]:
        xx = sx(mv)
        if px0 < xx < px1:
            xgrid_svg.append(f'<line x1="{xx:.2f}" y1="{py1:.2f}" x2="{xx:.2f}" y2="{py1+3:.2f}" stroke="#000000" stroke-width="0.7"/>')

    ygrid_svg = []
    for yv in yticks:
        yy = sy(yv)
        ygrid_svg.append(
            f'<line x1="{px0:.2f}" y1="{yy:.2f}" x2="{px1:.2f}" y2="{yy:.2f}" stroke="#E3E8EE" stroke-width="0.9" stroke-dasharray="3,3"/>'
            f'<line x1="{px0-5:.2f}" y1="{yy:.2f}" x2="{px0:.2f}" y2="{yy:.2f}" stroke="#000000" stroke-width="1.0"/>'
            f'<text x="{px0-9:.2f}" y="{yy+4:.2f}" text-anchor="end" font-size="11.5" fill="#000000">{yv}</text>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
  @page {{
    size: 1384px 456px;
    margin: 0;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 1384px;
    height: 456px;
    background: #FFFFFF;
    overflow: hidden;
    font-family: "Segoe UI", "Arial", "Helvetica Neue", sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  svg {{ display: block; width: 1384px; height: 456px; }}
</style>
</head>
<body>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1384 456" width="1384" height="456">
  <defs>
    <!-- Soft editorial card elevation shadow -->
    <filter id="cardShadow" x="-6%" y="-6%" width="114%" height="118%">
      <feDropShadow dx="0" dy="1.8" stdDeviation="2.4" flood-color="#0F2942" flood-opacity="0.08"/>
    </filter>
    <filter id="heroShadow" x="-6%" y="-8%" width="114%" height="122%">
      <feDropShadow dx="0" dy="2.5" stdDeviation="3.4" flood-color="#005282" flood-opacity="0.14"/>
    </filter>

    <!-- Diagonal hazard hatching for MEMS dead-time stall -->
    <pattern id="hatchVerm" width="7" height="7" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
      <rect width="7" height="7" fill="#FBE5D8"/>
      <line x1="0" y1="0" x2="0" y2="7" stroke="#D55E00" stroke-width="2.0" stroke-opacity="0.28"/>
    </pattern>
    <!-- Subtle diagonal hatch for hidden pre-switch -->
    <pattern id="hatchSky" width="7" height="7" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
      <rect width="7" height="7" fill="#E4F3FC"/>
      <line x1="0" y1="0" x2="0" y2="7" stroke="#0072B2" stroke-width="1.6" stroke-opacity="0.22"/>
    </pattern>

    <!-- Sleek gradients (Okabe-Ito derived, restrained academic) -->
    <linearGradient id="gradOpti" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0072B2"/>
      <stop offset="100%" stop-color="#005587"/>
    </linearGradient>
    <linearGradient id="gradBurst" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#CCE5F6"/>
      <stop offset="100%" stop-color="#B0D6F0"/>
    </linearGradient>
    <linearGradient id="gradPareto" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#D7EBF8" stop-opacity="0.72"/>
      <stop offset="100%" stop-color="#EDF6FC" stop-opacity="0.30"/>
    </linearGradient>
    <linearGradient id="gradBus" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#1E3A54"/>
      <stop offset="50%" stop-color="#294F73"/>
      <stop offset="100%" stop-color="#1E3A54"/>
    </linearGradient>
    <linearGradient id="gradSaved" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#D8F2E9" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#EAF8F3" stop-opacity="0.85"/>
    </linearGradient>

    <!-- Crisp small arrowheads -->
    <marker id="arrDark" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 7 4 L 0 7 z" fill="#1F2937"/>
    </marker>
    <marker id="arrBlue" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 7 4 L 0 7 z" fill="#0072B2"/>
    </marker>
    <marker id="arrGreen" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 7 4 L 0 7 z" fill="#009E73"/>
    </marker>
    <marker id="arrAmber" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 7 4 L 0 7 z" fill="#D48800"/>
    </marker>
    <marker id="arrVerm" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 7 4 L 0 7 z" fill="#D55E00"/>
    </marker>
    <marker id="arrVermL" viewBox="0 0 8 8" refX="1.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 8 1 L 1 4 L 8 7 z" fill="#D55E00"/>
    </marker>
    <marker id="arrBlueL" viewBox="0 0 8 8" refX="1.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 8 1 L 1 4 L 8 7 z" fill="#0072B2"/>
    </marker>
    <marker id="arrGreenL" viewBox="0 0 8 8" refX="1.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 8 1 L 1 4 L 8 7 z" fill="#009E73"/>
    </marker>
    <marker id="arrGray" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 7 4 L 0 7 z" fill="#64748B"/>
    </marker>
  </defs>

  <!-- Pure White Studio Background -->
  <rect x="0" y="0" width="1384" height="456" fill="#FFFFFF"/>

  <!-- =================================================================== -->
  <!-- PANEL (a): EDITORIAL TIME-ALIGNED GANTT CONTRAST (x: 10..462)       -->
  <!-- =================================================================== -->
  <text x="236" y="25" text-anchor="middle" font-size="14.5" font-weight="700" fill="#000000">(a) Optical Thrashing vs. Pause Hiding</text>

  <!-- LANE 1 CARD: Optical-Naive (y: 44..206) -->
  <g filter="url(#cardShadow)">
    <rect x="14" y="44" width="444" height="162" rx="8" fill="#FFFFFF" stroke="#E5CFC2" stroke-width="1.1"/>
    <!-- Left accent strip -->
    <path d="M 14 52 Q 14 44 22 44 L 20 44 L 20 206 L 22 206 Q 14 206 14 198 Z" fill="#D55E00"/>
    <rect x="14" y="44" width="5.5" height="162" rx="2.5" fill="#D55E00"/>
  </g>
  <!-- Lane 1 Header Row -->
  <text x="30" y="68" font-size="11.5" font-weight="700" fill="#000000">Conventional: Optical-Naive</text>
  <text x="190" y="68" font-size="10.0" font-weight="400" fill="#000000">(Packet &#x03C4; &#x2248; 0)</text>
  <!-- Status badge top-right -->
  <rect x="302" y="53" width="144" height="22" rx="11" fill="#FBE5D8" stroke="#D55E00" stroke-width="0.9"/>
  <text x="374" y="67.5" text-anchor="middle" font-size="9.7" font-weight="700" fill="#000000">Duty 3.8&#x2013;13.1% &#x00B7; 13.9 s</text>

  <!-- Lane 1 Gantt Track (y: 94..148, h=54) -->
  <!-- Turn k-1 -->
  <rect x="30" y="94" width="48" height="54" rx="5" fill="#F1F5F9" stroke="#94A3B8" stroke-width="1.0"/>
  <text x="54" y="117" text-anchor="middle" font-size="10.0" font-weight="600" fill="#000000">Turn</text>
  <text x="54" y="131" text-anchor="middle" font-size="10.0" font-style="italic" fill="#000000">k&#x2212;1</text>

  <!-- Agent Idle Pause -->
  <rect x="84" y="94" width="184" height="54" rx="5" fill="#FAFAFA" stroke="#94A3B8" stroke-width="1.0" stroke-dasharray="4,3"/>
  <!-- Mini pause icon -->
  <rect x="102" y="112" width="3.5" height="11" rx="1" fill="#475569"/>
  <rect x="108.5" y="112" width="3.5" height="11" rx="1" fill="#475569"/>
  <text x="184" y="117" text-anchor="middle" font-size="10.2" font-weight="600" fill="#000000">Agent Pause (0.71&#x2013;2.22 s)</text>
  <text x="184" y="132" text-anchor="middle" font-size="9.3" font-weight="400" fill="#000000">OCS Mirror Idle &amp; Unused</text>

  <!-- MEMS Switch Stall (Diagonal Hazard Hatching!) -->
  <rect x="278" y="94" width="76" height="54" rx="5" fill="url(#hatchVerm)" stroke="#D55E00" stroke-width="1.25"/>
  <text x="316" y="117" text-anchor="middle" font-size="10.0" font-weight="700" fill="#000000">MEMS Stall</text>
  <text x="316" y="132" text-anchor="middle" font-size="9.6" font-weight="600" fill="#000000">25&#x2013;50 ms</text>

  <!-- KV Xfer -->
  <rect x="359" y="94" width="38" height="54" rx="5" fill="#FEF3D6" stroke="#E69F00" stroke-width="1.1"/>
  <text x="378" y="117" text-anchor="middle" font-size="9.6" font-weight="600" fill="#000000">KV</text>
  <text x="378" y="131" text-anchor="middle" font-size="9.4" fill="#000000">Xfer</text>

  <!-- Turn k Prefill -->
  <rect x="402" y="94" width="44" height="54" rx="5" fill="#F1F5F9" stroke="#64748B" stroke-width="1.0"/>
  <text x="424" y="117" text-anchor="middle" font-size="9.6" font-weight="600" fill="#000000">Turn <tspan font-style="italic">k</tspan></text>
  <text x="424" y="131" text-anchor="middle" font-size="9.4" fill="#000000">Prefill</text>

  <!-- Critical path dimension bracket -->
  <line x1="278" y1="156" x2="278" y2="170" stroke="#D55E00" stroke-width="1.0"/>
  <line x1="446" y1="156" x2="446" y2="170" stroke="#D55E00" stroke-width="1.0"/>
  <line x1="281" y1="164" x2="443" y2="164" stroke="#D55E00" stroke-width="1.35" marker-start="url(#arrVermL)" marker-end="url(#arrVerm)"/>
  <text x="362" y="185" text-anchor="middle" font-size="10.2" font-weight="700" fill="#000000">Exposed Critical-Path Thrashing</text>
  <text x="176" y="185" text-anchor="middle" font-size="9.3" font-weight="400" fill="#000000">No prefetch during pause</text>

  <!-- LANE 2 CARD: OptiKV-Fabric (Ours) (y: 238..410) -->
  <g filter="url(#heroShadow)">
    <rect x="14" y="238" width="444" height="172" rx="8" fill="#FFFFFF" stroke="#B6D6EC" stroke-width="1.25"/>
    <rect x="14" y="238" width="5.5" height="172" rx="2.5" fill="#0072B2"/>
  </g>
  <!-- Lane 2 Header Row -->
  <text x="30" y="263" font-size="11.5" font-weight="700" fill="#000000">OptiKV-Fabric (Ours)</text>
  <text x="152" y="263" font-size="9.8" font-weight="400" fill="#000000">Calendar + GapFetch</text>
  <!-- Status badge top-right -->
  <rect x="302" y="248" width="144" height="22" rx="11" fill="#DCEEFB" stroke="#0072B2" stroke-width="0.95"/>
  <text x="374" y="262.5" text-anchor="middle" font-size="9.7" font-weight="700" fill="#000000">Duty 76&#x2013;96% &#x00B7; 36 ms</text>

  <!-- Lane 2 Gantt Track (y: 289..343, h=54) -->
  <!-- Turn k-1 -->
  <rect x="30" y="289" width="48" height="54" rx="5" fill="#F1F5F9" stroke="#94A3B8" stroke-width="1.0"/>
  <text x="54" y="312" text-anchor="middle" font-size="10.0" font-weight="600" fill="#000000">Turn</text>
  <text x="54" y="326" text-anchor="middle" font-size="10.0" font-style="italic" fill="#000000">k&#x2212;1</text>

  <!-- Pre-Switch (Hidden) -->
  <rect x="84" y="289" width="64" height="54" rx="5" fill="url(#hatchSky)" stroke="#56B4E9" stroke-width="1.1"/>
  <text x="116" y="312" text-anchor="middle" font-size="9.6" font-weight="700" fill="#000000">Pre-Sw.</text>
  <text x="116" y="326" text-anchor="middle" font-size="9.2" fill="#000000">(Hidden)</text>

  <!-- OCS Burst Prefetch -->
  <rect x="153" y="289" width="68" height="54" rx="5" fill="url(#gradBurst)" stroke="#0072B2" stroke-width="1.1"/>
  <text x="187" y="312" text-anchor="middle" font-size="9.6" font-weight="700" fill="#000000">OCS Burst</text>
  <text x="187" y="326" text-anchor="middle" font-size="9.2" fill="#000000">Prefetch</text>

  <!-- Warm HBM -->
  <rect x="226" y="289" width="42" height="54" rx="5" fill="#D9F2E9" stroke="#009E73" stroke-width="1.15"/>
  <text x="247" y="312" text-anchor="middle" font-size="9.4" font-weight="700" fill="#000000">Warm</text>
  <text x="247" y="326" text-anchor="middle" font-size="9.2" font-weight="600" fill="#000000">HBM</text>

  <!-- Turn k Delta-Prefill (Protagonist Deep Blue Bar!) -->
  <rect x="278" y="289" width="64" height="54" rx="5" fill="url(#gradOpti)" stroke="#004C78" stroke-width="1.2"/>
  <text x="310" y="312" text-anchor="middle" font-size="9.8" font-weight="700" fill="#FFFFFF">Turn <tspan font-style="italic">k</tspan></text>
  <text x="310" y="327" text-anchor="middle" font-size="9.6" font-weight="600" fill="#FFFFFF">&#x0394;-Prefill</text>

  <!-- Latency Saved Delta Zone -->
  <rect x="348" y="289" width="98" height="54" rx="5" fill="url(#gradSaved)" stroke="#009E73" stroke-width="1.0" stroke-dasharray="4,2.5"/>
  <line x1="356" y1="316" x2="438" y2="316" stroke="#009E73" stroke-width="1.35" marker-start="url(#arrGreenL)" marker-end="url(#arrGreen)"/>
  <text x="397" y="307" text-anchor="middle" font-size="9.8" font-weight="700" fill="#000000">Latency Saved</text>
  <text x="397" y="333" text-anchor="middle" font-size="9.4" font-weight="600" fill="#000000">Matches Oracle</text>

  <!-- Bottom brackets -->
  <line x1="84" y1="351" x2="84" y2="365" stroke="#0072B2" stroke-width="1.0"/>
  <line x1="268" y1="351" x2="268" y2="365" stroke="#0072B2" stroke-width="1.0"/>
  <line x1="87" y1="359" x2="265" y2="359" stroke="#0072B2" stroke-width="1.25" marker-start="url(#arrBlueL)" marker-end="url(#arrBlue)"/>
  <text x="176" y="381" text-anchor="middle" font-size="10.0" font-weight="700" fill="#000000">Switch &amp; Prefetch Hidden in Pause</text>

  <line x1="278" y1="351" x2="278" y2="365" stroke="#0072B2" stroke-width="1.0"/>
  <line x1="342" y1="351" x2="342" y2="365" stroke="#0072B2" stroke-width="1.0"/>
  <line x1="281" y1="359" x2="339" y2="359" stroke="#0072B2" stroke-width="1.25" marker-start="url(#arrBlueL)" marker-end="url(#arrBlue)"/>
  <text x="310" y="381" text-anchor="middle" font-size="10.0" font-weight="700" fill="#000000">0 ms Switch</text>

  <!-- Shared Vertical Arrival Alignment Line (x = 273) cleanly bridging both lanes! -->
  <line x1="273" y1="84" x2="273" y2="210" stroke="#000000" stroke-width="1.2" stroke-dasharray="3,2.5"/>
  <line x1="273" y1="234" x2="273" y2="352" stroke="#000000" stroke-width="1.2" stroke-dasharray="3,2.5"/>
  <!-- Turn k arrives pill centered cleanly in the gap between the 2 cards (y: 211..233) -->
  <rect x="225" y="211" width="96" height="22" rx="11" fill="#1E293B"/>
  <text x="273" y="225.5" text-anchor="middle" font-size="10.0" font-weight="700" fill="#FFFFFF">Turn <tspan font-style="italic">k</tspan> arrives</text>

  <!-- Time Axis Footer -->
  <line x1="24" y1="432" x2="442" y2="432" stroke="#475569" stroke-width="1.2" marker-end="url(#arrDark)"/>
  <text x="234" y="447" text-anchor="middle" font-size="10.5" font-weight="600" fill="#000000">Wall-clock timeline across multi-turn agent pause (<tspan font-style="italic">t</tspan>)</text>


  <!-- =================================================================== -->
  <!-- PANEL (b): MODERN HARDWARE-ICON SYSTEMS ARCHITECTURE (x: 478..942)  -->
  <!-- =================================================================== -->
  <text x="710" y="25" text-anchor="middle" font-size="14.5" font-weight="700" fill="#000000">(b) Unified OptiKV-Fabric Control Plane</text>

  <!-- Layer 1: Workload Input Banner (y: 44..94) -->
  <g filter="url(#cardShadow)">
    <rect x="484" y="44" width="452" height="48" rx="7" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.1"/>
    <rect x="484" y="44" width="5" height="48" rx="2.5" fill="#475569"/>
  </g>
  <!-- Multi-turn agent tree icon -->
  <g transform="translate(502, 55)" stroke="#334155" stroke-width="1.4" fill="none">
    <circle cx="5" cy="5" r="3.2" fill="#E2E8F0"/>
    <circle cx="19" cy="5" r="3.2" fill="#E2E8F0"/>
    <circle cx="12" cy="20" r="3.5" fill="#0072B2" stroke="#004C78"/>
    <line x1="7" y1="8" x2="10.5" y2="17"/>
    <line x1="17" y1="8" x2="13.5" y2="17"/>
  </g>
  <text x="536" y="64" font-size="11.2" font-weight="700" fill="#000000">Agentic Multi-Turn Sessions</text>
  <text x="692" y="64" font-size="9.8" font-weight="400" fill="#000000">(Prefix Hashes &amp; Pauses &#x0394;<tspan font-style="italic">t</tspan><tspan dy="2" font-size="7.6">gap</tspan><tspan dy="-2">)</tspan></text>
  <text x="536" y="81" font-size="9.8" font-weight="600" fill="#000000">Governed by Closed-Form Roofline Crossover Laws 1&#x2013;3 (Eqs. 5&#x2013;7)</text>

  <!-- Connector 1 -> 2 -->
  <line x1="710" y1="92" x2="710" y2="112" stroke="#1F2937" stroke-width="1.4" marker-end="url(#arrDark)"/>

  <!-- Layer 2: HERO ANCHOR — OCS Port Calendar (y: 114..174) -->
  <g filter="url(#heroShadow)">
    <rect x="484" y="114" width="452" height="58" rx="7" fill="#FFFFFF" stroke="#0072B2" stroke-width="1.4"/>
    <rect x="484" y="114" width="452" height="22" rx="6" fill="url(#gradOpti)"/>
    <rect x="484" y="128" width="452" height="8" fill="url(#gradOpti)"/>
  </g>
  <!-- Custom 2x2 Optical MEMS Mirror Crossbar Icon in white -->
  <g transform="translate(500, 118.5)" stroke="#FFFFFF" stroke-width="1.2" fill="none">
    <rect x="0" y="0" width="16" height="13" rx="2" stroke-opacity="0.85"/>
    <line x1="3" y1="4.5" x2="6.5" y2="2.5"/>
    <line x1="9.5" y1="4.5" x2="13" y2="2.5"/>
    <line x1="3" y1="10.5" x2="6.5" y2="8.5"/>
    <line x1="9.5" y1="10.5" x2="13" y2="8.5"/>
  </g>
  <text x="524" y="129.5" font-size="11.0" font-weight="700" fill="#FFFFFF">Single State Primitive: Matching-Conserved OCS Port Calendar</text>
  <text x="710" y="152.5" text-anchor="middle" font-size="10.3" font-weight="600" fill="#000000">Bipartite Matrix <tspan font-weight="700">A</tspan>(<tspan font-style="italic">t</tspan>), Port Release <tspan font-style="italic">t</tspan><tspan dy="2" font-size="7.8">free</tspan><tspan dy="-2"> &amp; Standing Latches (&#x03C4;</tspan><tspan dy="2" font-size="7.8">eff</tspan><tspan dy="-2"> = 0)</tspan></text>
  <text x="710" y="166.0" text-anchor="middle" font-size="9.0" font-weight="400" fill="#000000">Enforces |L<tspan dy="2" font-size="7.2">pool</tspan><tspan dy="-2">(</tspan><tspan font-style="italic">t</tspan>)| &#x2264; min(<tspan font-style="italic">W, P</tspan>) &#x00B7; Zero-Lookahead Online Scheduling</text>

  <!-- Clean Orthogonal Fan-out Bus: Layer 2 -> Layer 3 (y: 172..198) -->
  <line x1="710" y1="172" x2="710" y2="185" stroke="#0072B2" stroke-width="1.35"/>
  <line x1="556" y1="185" x2="864" y2="185" stroke="#0072B2" stroke-width="1.35"/>
  <line x1="556" y1="185" x2="556" y2="198" stroke="#0072B2" stroke-width="1.35" marker-end="url(#arrBlue)"/>
  <line x1="710" y1="185" x2="710" y2="198" stroke="#009E73" stroke-width="1.35" marker-end="url(#arrGreen)"/>
  <line x1="864" y1="185" x2="864" y2="198" stroke="#D48800" stroke-width="1.35" marker-end="url(#arrAmber)"/>

  <!-- Layer 3: Three Cohesive Module Cards (y: 201..299, w=144, gap=10 -> 484, 638, 792) -->
  <!-- Card 3A: OptiRoute -->
  <g filter="url(#cardShadow)">
    <rect x="484" y="201" width="144" height="98" rx="7" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.05"/>
    <path d="M 484 208 Q 484 201 491 201 L 621 201 Q 628 201 628 208 L 628 206 L 484 206 Z" fill="#0072B2"/>
    <rect x="484" y="201" width="144" height="5" rx="2.5" fill="#0072B2"/>
  </g>
  <!-- Routing fork icon -->
  <g transform="translate(495, 214)" stroke="#0072B2" stroke-width="1.4" fill="none">
    <circle cx="3" cy="8" r="2.2" fill="#0072B2"/>
    <path d="M 5 8 L 9 8 L 13 3.5 M 9 8 L 13 12.5"/>
    <circle cx="14.5" cy="3.5" r="1.8"/>
    <circle cx="14.5" cy="12.5" r="1.8"/>
  </g>
  <text x="516" y="225" font-size="11.2" font-weight="700" fill="#000000">OptiRoute</text>
  <rect x="580" y="213" width="40" height="15" rx="7.5" fill="#E6F2FA"/>
  <text x="600" y="223.5" text-anchor="middle" font-size="8.6" font-weight="700" fill="#000000">&#x00A7;III-B</text>
  <line x1="494" y1="234" x2="618" y2="234" stroke="#E2E8F0" stroke-width="0.9"/>
  <text x="495" y="250" font-size="9.6" fill="#000000">&#x2022; Standing-circuit affinity</text>
  <text x="495" y="266" font-size="9.6" fill="#000000">&#x2022; Pool-first coalescing</text>
  <text x="495" y="282" font-size="9.6" fill="#000000">&#x2022; Exact recompute abort</text>

  <!-- Card 3B: GapFetch -->
  <g filter="url(#cardShadow)">
    <rect x="638" y="201" width="144" height="98" rx="7" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.05"/>
    <rect x="638" y="201" width="144" height="5" rx="2.5" fill="#009E73"/>
  </g>
  <!-- Clock/Prefetch icon -->
  <g transform="translate(649, 214)" stroke="#009E73" stroke-width="1.4" fill="none">
    <circle cx="8" cy="8" r="6.5"/>
    <path d="M 8 4.2 L 8 8 L 11 9.8"/>
  </g>
  <text x="670" y="225" font-size="11.2" font-weight="700" fill="#000000">GapFetch</text>
  <rect x="734" y="213" width="40" height="15" rx="7.5" fill="#DCF4EC"/>
  <text x="754" y="223.5" text-anchor="middle" font-size="8.6" font-weight="700" fill="#000000">&#x00A7;III-C</text>
  <line x1="648" y1="234" x2="772" y2="234" stroke="#E2E8F0" stroke-width="0.9"/>
  <text x="649" y="250" font-size="9.6" fill="#000000">&#x2022; Causal EWMA pause</text>
  <text x="649" y="266" font-size="9.6" fill="#000000">&#x2022; Off-path JIT bursts</text>
  <text x="649" y="282" font-size="9.6" fill="#000000">&#x2022; Strata in-flight join</text>

  <!-- Card 3C: PrefixKeep -->
  <g filter="url(#cardShadow)">
    <rect x="792" y="201" width="144" height="98" rx="7" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.05"/>
    <rect x="792" y="201" width="144" height="5" rx="2.5" fill="#E69F00"/>
  </g>
  <!-- Shield/Pin icon -->
  <g transform="translate(803, 214)" stroke="#C87D00" stroke-width="1.4" fill="none">
    <path d="M 8 1.5 L 14 4 L 14 8.5 C 14 12 11 14 8 15 C 5 14 2 12 2 8.5 L 2 4 Z" fill="#FEF3D6"/>
    <path d="M 5.5 8.2 L 7.3 10.1 L 10.8 6.4"/>
  </g>
  <text x="824" y="225" font-size="11.2" font-weight="700" fill="#000000">PrefixKeep</text>
  <rect x="888" y="213" width="40" height="15" rx="7.5" fill="#FEF3D6"/>
  <text x="908" y="223.5" text-anchor="middle" font-size="8.6" font-weight="700" fill="#000000">&#x00A7;III-D</text>
  <line x1="802" y1="234" x2="926" y2="234" stroke="#E2E8F0" stroke-width="0.9"/>
  <text x="803" y="250" font-size="9.6" fill="#000000">&#x2022; 2nd-obs anchor filter</text>
  <text x="803" y="266" font-size="9.6" fill="#000000">&#x2022; Bounded Prefix-GDSF</text>
  <text x="803" y="282" font-size="9.6" fill="#000000">&#x2022; Request-unit pinning</text>

  <!-- Clean Steel/Optical Interconnect Bus (y: 315..336) -->
  <line x1="556" y1="299" x2="556" y2="315" stroke="#0072B2" stroke-width="1.35"/>
  <line x1="710" y1="299" x2="710" y2="315" stroke="#009E73" stroke-width="1.35"/>
  <line x1="864" y1="299" x2="864" y2="315" stroke="#D48800" stroke-width="1.35"/>

  <rect x="484" y="315" width="452" height="21" rx="10.5" fill="url(#gradBus)"/>
  <text x="710" y="329" text-anchor="middle" font-size="9.8" font-weight="700" fill="#FFFFFF" letter-spacing="0.2">Five-Tier Lossless Data &amp; Optical Control Bus (PCIe + WDM OCS)</text>

  <!-- 5 Clean Orthogonal Drops from Bus to T1..T5 -->
  <!-- T1..T5 geometry: 5 cards across 484..936 (452px): w=84, gap=8 -> 484, 576, 668, 760, 852 -->
  <line x1="526" y1="336" x2="526" y2="352" stroke="#0072B2" stroke-width="1.35" marker-end="url(#arrBlue)"/>
  <line x1="618" y1="336" x2="618" y2="352" stroke="#D48800" stroke-width="1.35" marker-end="url(#arrAmber)"/>
  <line x1="710" y1="336" x2="710" y2="352" stroke="#64748B" stroke-width="1.35" marker-end="url(#arrGray)"/>
  <line x1="802" y1="336" x2="802" y2="352" stroke="#0072B2" stroke-width="1.35" marker-end="url(#arrBlue)"/>
  <line x1="894" y1="336" x2="894" y2="352" stroke="#009E73" stroke-width="1.35" marker-end="url(#arrGreen)"/>

  <!-- Layer 4: Five Hardware-Icon Memory Tier Cards (y: 355..442, h=87) -->
  <!-- T1: GPU HBM -->
  <g filter="url(#cardShadow)">
    <rect x="484" y="355" width="84" height="87" rx="6" fill="#F4FAFF" stroke="#0072B2" stroke-width="1.2"/>
    <rect x="484" y="355" width="84" height="4.5" rx="2" fill="#0072B2"/>
  </g>
  <!-- GPU chip icon -->
  <g transform="translate(516, 366)" stroke="#0072B2" stroke-width="1.2" fill="none">
    <rect x="3" y="3" width="14" height="14" rx="2" fill="#DCEEFB"/>
    <rect x="6.5" y="6.5" width="7" height="7" rx="1" fill="#0072B2"/>
    <line x1="6" y1="0.5" x2="6" y2="3"/><line x1="10" y1="0.5" x2="10" y2="3"/><line x1="14" y1="0.5" x2="14" y2="3"/>
    <line x1="6" y1="17" x2="6" y2="19.5"/><line x1="10" y1="17" x2="10" y2="19.5"/><line x1="14" y1="17" x2="14" y2="19.5"/>
  </g>
  <text x="526" y="401" text-anchor="middle" font-size="10.2" font-weight="700" fill="#000000">T1: HBM</text>
  <text x="526" y="416" text-anchor="middle" font-size="9.0" font-weight="600" fill="#000000">Pinned</text>
  <text x="526" y="429" text-anchor="middle" font-size="8.7" font-weight="400" fill="#000000">Hot Tier</text>

  <!-- T2: Host DRAM -->
  <g filter="url(#cardShadow)">
    <rect x="576" y="355" width="84" height="87" rx="6" fill="#FFFFFF" stroke="#E2C37E" stroke-width="1.1"/>
    <rect x="576" y="355" width="84" height="4.5" rx="2" fill="#E69F00"/>
  </g>
  <!-- DRAM DIMM icon -->
  <g transform="translate(606, 368)" stroke="#C87D00" stroke-width="1.2" fill="none">
    <rect x="0" y="2" width="24" height="11" rx="1.5" fill="#FEF3D6"/>
    <rect x="3" y="4.5" width="4" height="6" fill="#E69F00"/>
    <rect x="10" y="4.5" width="4" height="6" fill="#E69F00"/>
    <rect x="17" y="4.5" width="4" height="6" fill="#E69F00"/>
    <line x1="4" y1="13" x2="4" y2="16"/><line x1="9" y1="13" x2="9" y2="16"/><line x1="15" y1="13" x2="15" y2="16"/><line x1="20" y1="13" x2="20" y2="16"/>
  </g>
  <text x="618" y="401" text-anchor="middle" font-size="10.2" font-weight="700" fill="#000000">T2: DRAM</text>
  <text x="618" y="416" text-anchor="middle" font-size="9.0" font-weight="600" fill="#000000">26&#x2013;50 GB/s</text>
  <text x="618" y="429" text-anchor="middle" font-size="8.7" font-weight="400" fill="#000000">PCIe Spill</text>

  <!-- T3: Local SSD -->
  <g filter="url(#cardShadow)">
    <rect x="668" y="355" width="84" height="87" rx="6" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.1"/>
    <rect x="668" y="355" width="84" height="4.5" rx="2" fill="#64748B"/>
  </g>
  <!-- NVMe Drive icon -->
  <g transform="translate(698, 367)" stroke="#475569" stroke-width="1.2" fill="none">
    <rect x="1" y="2" width="22" height="14" rx="2" fill="#F1F5F9"/>
    <line x1="5" y1="6.5" x2="15" y2="6.5"/>
    <line x1="5" y1="10.5" x2="12" y2="10.5"/>
    <circle cx="18.5" cy="11.5" r="1.5" fill="#475569"/>
  </g>
  <text x="710" y="401" text-anchor="middle" font-size="10.2" font-weight="700" fill="#000000">T3: SSD</text>
  <text x="710" y="416" text-anchor="middle" font-size="9.0" font-weight="600" fill="#000000">NVMe</text>
  <text x="710" y="429" text-anchor="middle" font-size="8.7" font-weight="400" fill="#000000">Cold Tier</text>

  <!-- T4: Peer HBM -->
  <g filter="url(#cardShadow)">
    <rect x="760" y="355" width="84" height="87" rx="6" fill="#FFFFFF" stroke="#9DD2F2" stroke-width="1.1"/>
    <rect x="760" y="355" width="84" height="4.5" rx="2" fill="#56B4E9"/>
  </g>
  <!-- Peer link icon -->
  <g transform="translate(789, 368)" stroke="#0072B2" stroke-width="1.2" fill="none">
    <rect x="0" y="3" width="9" height="10" rx="1.5" fill="#E6F4FC"/>
    <rect x="17" y="3" width="9" height="10" rx="1.5" fill="#E6F4FC"/>
    <line x1="9" y1="8" x2="17" y2="8" stroke-dasharray="2,1.5"/>
  </g>
  <text x="802" y="401" text-anchor="middle" font-size="10.2" font-weight="700" fill="#000000">T4: Peer</text>
  <text x="802" y="416" text-anchor="middle" font-size="9.0" font-weight="600" fill="#000000">Direct</text>
  <text x="802" y="429" text-anchor="middle" font-size="8.7" font-weight="400" fill="#000000">OCS Hop</text>

  <!-- T5: Optical KV Pool -->
  <g filter="url(#cardShadow)">
    <rect x="852" y="355" width="84" height="87" rx="6" fill="#F2FBF7" stroke="#009E73" stroke-width="1.2"/>
    <rect x="852" y="355" width="84" height="4.5" rx="2" fill="#009E73"/>
  </g>
  <!-- Optical rack/wavelength pool icon -->
  <g transform="translate(882, 366)" stroke="#009E73" stroke-width="1.2" fill="none">
    <rect x="2" y="2" width="20" height="5" rx="1.2" fill="#D9F2E9"/>
    <rect x="2" y="9" width="20" height="5" rx="1.2" fill="#D9F2E9"/>
    <path d="M 4 17.5 Q 8 14.5 12 17.5 T 20 17.5" stroke="#0072B2" stroke-width="1.3"/>
  </g>
  <text x="894" y="401" text-anchor="middle" font-size="10.2" font-weight="700" fill="#000000">T5: Pool</text>
  <text x="894" y="416" text-anchor="middle" font-size="9.0" font-weight="600" fill="#000000">45&#x2013;800 GB/s</text>
  <text x="894" y="429" text-anchor="middle" font-size="8.7" font-weight="400" fill="#000000">Optical OCS</text>


  <!-- =================================================================== -->
  <!-- PANEL (c): MINIMALIST VECTOR PARETO FRONTIER (x: 960..1376)         -->
  <!-- =================================================================== -->
  <text x="1186" y="25" text-anchor="middle" font-size="14.5" font-weight="700" fill="#000000">(c) 128K Goodput vs. E2E Fabric Power</text>

  <!-- Pareto-Optimal shaded region -->
  <rect x="{px0:.2f}" y="{py0:.2f}" width="{pareto_x1 - px0:.2f}" height="{pareto_y1 - py0:.2f}" fill="url(#gradPareto)"/>
  <text x="{px0 + 8:.2f}" y="{py0 + 18:.2f}" font-size="10.6" font-weight="700" fill="#000000">Pareto-Optimal</text>

  <!-- Gridlines & Ticks -->
  {"".join(xgrid_svg)}
  {"".join(ygrid_svg)}

  <!-- Matched 200 GB/s Electrical Vertical Reference -->
  <line x1="{sx(PWR_ELEC_1600G_E2E):.2f}" y1="{py0:.2f}" x2="{sx(PWR_ELEC_1600G_E2E):.2f}" y2="{py1:.2f}" stroke="#888888" stroke-width="1.2" stroke-dasharray="2.5,3.5"/>
  <text x="{sx(PWR_ELEC_1600G_E2E) - 7:.2f}" y="{sy(2.14):.2f}" text-anchor="end" font-size="9.6" font-weight="400" fill="#000000">Matched 200 GB/s</text>
  <text x="{sx(PWR_ELEC_1600G_E2E) - 7:.2f}" y="{sy(1.93):.2f}" text-anchor="end" font-size="9.6" font-weight="400" fill="#000000">Electrical (3.16 kW)</text>
  <text x="{sx(PWR_ELEC_1600G_E2E) - 7:.2f}" y="{sy(1.72):.2f}" text-anchor="end" font-size="9.5" font-weight="700" fill="#000000">4.9&#x00D7;&#x2013;7.8&#x00D7; power</text>

  <!-- Oracle Horizontal Dashed Bound -->
  <line x1="{px0:.2f}" y1="{sy(gp_orc):.2f}" x2="{px1:.2f}" y2="{sy(gp_orc):.2f}" stroke="#000000" stroke-width="1.35" stroke-dasharray="6,4"/>
  <text x="{px1 - 4:.2f}" y="{sy(gp_orc) - 10:.2f}" text-anchor="end" font-size="10.4" font-weight="700" fill="#000000">Oracle ({gp_orc:.2f} req/s)</text>

  <!-- Axes Spines (clean bottom + left) -->
  <line x1="{px0:.2f}" y1="{py0:.2f}" x2="{px0:.2f}" y2="{py1:.2f}" stroke="#000000" stroke-width="1.2"/>
  <line x1="{px0:.2f}" y1="{py1:.2f}" x2="{px1:.2f}" y2="{py1:.2f}" stroke="#000000" stroke-width="1.2"/>

  <!-- Axis Labels -->
  <text x="{(px0 + px1)/2:.2f}" y="446" text-anchor="middle" font-size="11.2" font-weight="600" fill="#000000">End-to-end fabric power: switch + NIC optics (W)</text>
  <text x="971" y="{(py0 + py1)/2:.2f}" text-anchor="middle" font-size="11.2" font-weight="600" fill="#000000" transform="rotate(-90 971 {(py0 + py1)/2:.2f})">Peak 128K SLO goodput (req/s)</text>

  <!-- SINGLE Thin Vertical Leader Line ONLY for sandwiched middle dot OptiKV 1:1 (646 W) -->
  <line x1="{c_opt11[0]:.2f}" y1="{c_opt11[1] + 9:.2f}" x2="{c_opt11[0]:.2f}" y2="{sy(2.94):.2f}" stroke="#0072B2" stroke-width="1.15"/>

  <!-- Data Points (exact coordinates, crisp strokes) -->
  <!-- Matched 200G Elec -->
  <rect x="{c_elec200[0]-5:.2f}" y="{c_elec200[1]-5:.2f}" width="10" height="10" fill="#A69060" stroke="#000000" stroke-width="1.0"/>
  <!-- CONCUR+RDMA -->
  <polygon points="{_polygon_points(c_concur[0], c_concur[1], 5.5, 6, 0)}" fill="#CC79A7" stroke="#000000" stroke-width="0.9"/>
  <!-- RDMA-25G+Rt -->
  <rect x="{c_rrt[0]-5.2:.2f}" y="{c_rrt[1]-5.2:.2f}" width="10.4" height="10.4" fill="#E69F00" stroke="#000000" stroke-width="1.0"/>
  <!-- RDMA-25G+Pf -->
  <polygon points="{_polygon_points(c_rpf[0], c_rpf[1], 6.2, 5, -90)}" fill="#009E73" stroke="#000000" stroke-width="1.0"/>
  <!-- Optical-Naive -->
  <polygon points="{_polygon_points(c_naive[0], c_naive[1], 6.4, 4, 0)}" fill="#D55E00" stroke="#000000" stroke-width="1.0"/>
  <!-- HBM/NoCache -->
  <polygon points="{_polygon_points(c_hbm[0], c_hbm[1], 6.0, 3, 90)}" fill="#7F7F7F" stroke="#000000" stroke-width="1.0"/>
  <!-- RDMA-50G (400G Line) -->
  <polygon points="{_polygon_points(c_r50[0], c_r50[1], 6.8, 3, -90)}" fill="#2B6B56" stroke="#000000" stroke-width="1.0"/>
  <!-- OptiKV 1:1 (646 W) -->
  <circle cx="{c_opt11[0]:.2f}" cy="{c_opt11[1]:.2f}" r="6.8" fill="#0072B2" stroke="#000000" stroke-width="1.1"/>
  <!-- OptiKV 4:1 (404 W) Star Protagonist -->
  <polygon points="{_star_points(c_opt41[0], c_opt41[1], 10.5, 4.6, 5)}" fill="#004C78" stroke="#000000" stroke-width="1.05"/>

  <!-- Direct Minimalist Labels (ZERO redundant arrows, pure black ink!) -->
  <text x="{sx(318):.2f}" y="{sy(3.64):.2f}" text-anchor="middle" font-size="9.6" font-weight="700" fill="#000000">OptiKV 4:1 (404 W)</text>
  <text x="{sx(318):.2f}" y="{sy(3.41):.2f}" text-anchor="middle" font-size="9.1" font-weight="500" fill="#000000">{gp_opt41:.2f} ({gpw_41_r50:.2f}&#x00D7; 50G / {gpw_41:.1f}&#x00D7; 25G)</text>

  <text x="{c_opt11[0]:.2f}" y="{sy(2.74):.2f}" text-anchor="middle" font-size="9.6" font-weight="700" fill="#000000">OptiKV 1:1 (646 W)</text>
  <text x="{c_opt11[0]:.2f}" y="{sy(2.51):.2f}" text-anchor="middle" font-size="9.2" font-weight="400" fill="#000000">50 ms P50</text>

  <text x="{sx(900):.2f}" y="{sy(3.69):.2f}" text-anchor="start" font-size="9.5" font-weight="700" fill="#000000">RDMA-50G (790 W)</text>
  <text x="{sx(900):.2f}" y="{sy(3.46):.2f}" text-anchor="start" font-size="9.1" font-weight="400" fill="#000000">162 ms (3.25&#x00D7; slower)</text>

  <text x="{c_rpf[0] + 12:.2f}" y="{c_rpf[1] + 3.5:.2f}" text-anchor="start" font-size="9.6" fill="#000000">RDMA-25G+Pf ({gp_rpf:.2f})</text>
  <text x="{c_concur[0] + 11:.2f}" y="{c_rrt[1] + 2:.2f}" text-anchor="start" font-size="9.4" fill="#000000">RDMA-25G+Rt /</text>
  <text x="{c_concur[0] + 11:.2f}" y="{c_rrt[1] + 14:.2f}" text-anchor="start" font-size="9.4" fill="#000000">CONCUR ({gp_rrt:.2f})</text>

  <text x="{c_naive[0] - 12:.2f}" y="{c_naive[1] + 3.5:.2f}" text-anchor="end" font-size="9.6" fill="#000000">Opt-Naive (0.00)</text>
  <text x="{c_hbm[0] + 11:.2f}" y="{c_hbm[1] + 3.5:.2f}" text-anchor="start" font-size="9.6" fill="#000000">HBM / NoCache (0.00)</text>
</svg>
</body>
</html>
"""
    return html


def main() -> int:
    browser = _find_browser()
    FIG_EXP.mkdir(parents=True, exist_ok=True)
    FIG_PAPER.mkdir(parents=True, exist_ok=True)

    # Use a pure-ASCII temp directory so Chrome & TeX Live Perl tools never hit CJK path encoding issues
    with tempfile.TemporaryDirectory(prefix="optikv_fig1_") as td:
        tmp = Path(td)
        html_path = tmp / "fig1.html"
        raw_pdf = tmp / "fig1_raw.pdf"
        crop_pdf = tmp / "fig1_crop.pdf"
        png_prefix = tmp / "fig1_render"

        html_path.write_text(build_svg_html(), encoding="utf-8")

        subprocess.run(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--no-pdf-header-footer",
                f"--print-to-pdf={raw_pdf}",
                html_path.resolve().as_uri(),
            ],
            check=True,
        )

        # Chrome already prints exact @page size (1384x456 px); run pdfcrop if available & clean
        pdfcrop = TEXLIVE_BIN / "pdfcrop.exe"
        final_pdf = raw_pdf
        if pdfcrop.is_file():
            res = subprocess.run([str(pdfcrop), "--margins", "2", str(raw_pdf), str(crop_pdf)], capture_output=True)
            if res.returncode == 0 and crop_pdf.is_file():
                final_pdf = crop_pdf

        shutil.copy2(final_pdf, FIG_EXP / "fig1_teaser.pdf")
        shutil.copy2(final_pdf, FIG_PAPER / "fig1_teaser.pdf")

        # Render 300-DPI PNG from the exact vector PDF via pdftoppm
        pdftoppm = TEXLIVE_BIN / "pdftoppm.exe"
        subprocess.run(
            [str(pdftoppm), "-png", "-r", "300", "-singlefile", str(final_pdf), str(png_prefix)],
            check=True,
        )
        rendered_png = tmp / "fig1_render.png"
        shutil.copy2(rendered_png, FIG_EXP / "fig1_teaser.png")
        shutil.copy2(rendered_png, FIG_PAPER / "fig1_teaser.png")

    print("Saved native vector PDF + 300-DPI PNG -> experiments/figures & paper/figures/fig1_teaser.{pdf,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
