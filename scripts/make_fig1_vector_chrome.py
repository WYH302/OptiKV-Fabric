#!/usr/bin/env python3
"""Generate publication-grade native Vector SVG -> Vector PDF figures via headless Chrome:
  1. fig1_teaser.{pdf,png}       : Spacious 2-panel Introduction Teaser
                                   (a) Optical Thrashing vs. Pause Hiding (aligned Gantt)
                                   (b) 128K Goodput vs. E2E Fabric Power (minimalist Pareto)
  2. fig_arch_overview.{pdf,png} : Dedicated full-width standalone Overall Architecture Diagram
                                   for Section III (OptiKV-Fabric Control Plane & 5-Tier Hierarchy)

All text is pure black (#000000) on light surfaces (white only on dark accent strips),
using Okabe-Ito colorblind-safe accents, custom vector hardware icons, and exact CSV-driven data.
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


def _render_html_to_pdf_png(browser: Path, html: str, stem: str) -> None:
    FIG_EXP.mkdir(parents=True, exist_ok=True)
    FIG_PAPER.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"optikv_{stem}_") as td:
        tmp = Path(td)
        html_path = tmp / f"{stem}.html"
        raw_pdf = tmp / f"{stem}_raw.pdf"
        crop_pdf = tmp / f"{stem}_crop.pdf"
        png_prefix = tmp / f"{stem}_render"

        html_path.write_text(html, encoding="utf-8")
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

        pdfcrop = TEXLIVE_BIN / "pdfcrop.exe"
        final_pdf = raw_pdf
        if pdfcrop.is_file():
            res = subprocess.run([str(pdfcrop), "--margins", "2", str(raw_pdf), str(crop_pdf)], capture_output=True)
            if res.returncode == 0 and crop_pdf.is_file():
                final_pdf = crop_pdf

        shutil.copy2(final_pdf, FIG_EXP / f"{stem}.pdf")
        shutil.copy2(final_pdf, FIG_PAPER / f"{stem}.pdf")

        pdftoppm = TEXLIVE_BIN / "pdftoppm.exe"
        subprocess.run(
            [str(pdftoppm), "-png", "-r", "300", "-singlefile", str(final_pdf), str(png_prefix)],
            check=True,
        )
        rendered_png = tmp / f"{stem}_render.png"
        shutil.copy2(rendered_png, FIG_EXP / f"{stem}.png")
        shutil.copy2(rendered_png, FIG_PAPER / f"{stem}.png")
    print(f"  saved native vector PDF + 300-DPI PNG -> {stem}.{{pdf,png}}", flush=True)


# ===========================================================================
# FIGURE 1: Spacious 2-Panel Introduction Teaser (1120 x 390 px)
#   (a) Optical Thrashing vs. Pause Hiding
#   (b) 128K Goodput vs. E2E Fabric Power
# ===========================================================================

def build_fig1_teaser_html() -> str:
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

    # Panel (b) Pareto plot frame: x=[640..1102], y=[44..340]
    px0, px1 = 640.0, 1102.0
    py0, py1 = 44.0, 336.0
    xmin, xmax = 145.0, 5400.0
    ymin, ymax = -0.34, 4.92

    def sx(w: float) -> float:
        return px0 + (math.log10(w) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * (px1 - px0)

    def sy(v: float) -> float:
        return py1 - (v - ymin) / (ymax - ymin) * (py1 - py0)

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
    pareto_y1 = sy(3.05)

    xticks = [(250, "250"), (500, "500"), (1000, "1K"), (2000, "2K"), (4000, "4K")]
    yticks = [0, 1, 2, 3, 4]
    xgrid_svg = []
    for xv, xlbl in xticks:
        xx = sx(xv)
        xgrid_svg.append(
            f'<line x1="{xx:.2f}" y1="{py0:.2f}" x2="{xx:.2f}" y2="{py1:.2f}" stroke="#E3E8EE" stroke-width="0.9" stroke-dasharray="3,3"/>'
            f'<line x1="{xx:.2f}" y1="{py1:.2f}" x2="{xx:.2f}" y2="{py1+5:.2f}" stroke="#000000" stroke-width="1.0"/>'
            f'<text x="{xx:.2f}" y="{py1+18:.2f}" text-anchor="middle" font-size="11.0" fill="#000000">{xlbl}</text>'
        )
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
            f'<text x="{px0-9:.2f}" y="{yy+4:.2f}" text-anchor="end" font-size="11.0" fill="#000000">{yv}</text>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
  @page {{ size: 1120px 390px; margin: 0; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 1120px; height: 390px; background: #FFFFFF; overflow: hidden;
    font-family: "Segoe UI", "Arial", "Helvetica Neue", sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  svg {{ display: block; width: 1120px; height: 390px; }}
</style>
</head>
<body>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 390" width="1120" height="390">
  <defs>
    <filter id="cardShadow" x="-5%" y="-6%" width="112%" height="118%">
      <feDropShadow dx="0" dy="1.8" stdDeviation="2.4" flood-color="#0F2942" flood-opacity="0.08"/>
    </filter>
    <filter id="heroShadow" x="-5%" y="-6%" width="112%" height="118%">
      <feDropShadow dx="0" dy="2.2" stdDeviation="3.0" flood-color="#005282" flood-opacity="0.12"/>
    </filter>
    <pattern id="hatchVerm" width="7" height="7" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
      <rect width="7" height="7" fill="#FBE5D8"/>
      <line x1="0" y1="0" x2="0" y2="7" stroke="#D55E00" stroke-width="2.0" stroke-opacity="0.28"/>
    </pattern>
    <pattern id="hatchSky" width="7" height="7" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
      <rect width="7" height="7" fill="#E4F3FC"/>
      <line x1="0" y1="0" x2="0" y2="7" stroke="#0072B2" stroke-width="1.6" stroke-opacity="0.22"/>
    </pattern>
    <linearGradient id="gradOpti" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0072B2"/><stop offset="100%" stop-color="#005587"/>
    </linearGradient>
    <linearGradient id="gradBurst" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#CCE5F6"/><stop offset="100%" stop-color="#B0D6F0"/>
    </linearGradient>
    <linearGradient id="gradPareto" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#D7EBF8" stop-opacity="0.72"/>
      <stop offset="100%" stop-color="#EDF6FC" stop-opacity="0.30"/>
    </linearGradient>
    <linearGradient id="gradSaved" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#D8F2E9" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#EAF8F3" stop-opacity="0.85"/>
    </linearGradient>
    <marker id="arrDark" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 7 4 L 0 7 z" fill="#1F2937"/>
    </marker>
    <marker id="arrBlue" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 7 4 L 0 7 z" fill="#0072B2"/>
    </marker>
    <marker id="arrBlueL" viewBox="0 0 8 8" refX="1.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 8 1 L 1 4 L 8 7 z" fill="#0072B2"/>
    </marker>
    <marker id="arrGreen" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 7 4 L 0 7 z" fill="#009E73"/>
    </marker>
    <marker id="arrGreenL" viewBox="0 0 8 8" refX="1.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 8 1 L 1 4 L 8 7 z" fill="#009E73"/>
    </marker>
    <marker id="arrVerm" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 7 4 L 0 7 z" fill="#D55E00"/>
    </marker>
    <marker id="arrVermL" viewBox="0 0 8 8" refX="1.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 8 1 L 1 4 L 8 7 z" fill="#D55E00"/>
    </marker>
  </defs>

  <rect x="0" y="0" width="1120" height="390" fill="#FFFFFF"/>

  <!-- =================================================================== -->
  <!-- PANEL (a): SPACIOUS TIME-ALIGNED GANTT CONTRAST (x: 14..566)        -->
  <!-- =================================================================== -->
  <text x="290" y="24" text-anchor="middle" font-size="14.2" font-weight="700" fill="#000000">(a) Optical Thrashing vs. Pause Hiding</text>

  <!-- LANE 1 CARD: Optical-Naive (y: 40..176, h=136) -->
  <g filter="url(#cardShadow)">
    <rect x="14" y="40" width="552" height="136" rx="8" fill="#FFFFFF" stroke="#E5CFC2" stroke-width="1.1"/>
    <rect x="14" y="40" width="6" height="136" rx="2.5" fill="#D55E00"/>
  </g>
  <text x="32" y="63" font-size="11.6" font-weight="700" fill="#000000">Conventional: Optical-Naive</text>
  <text x="196" y="63" font-size="10.2" font-weight="400" fill="#000000">(Packet-Switched &#x03C4; &#x2248; 0)</text>
  <rect x="394" y="48" width="158" height="22" rx="11" fill="#FBE5D8" stroke="#D55E00" stroke-width="0.9"/>
  <text x="473" y="62.5" text-anchor="middle" font-size="9.8" font-weight="700" fill="#000000">Duty 3.8&#x2013;13.1% &#x00B7; 13.9 s</text>

  <!-- Lane 1 Gantt Track (y: 84..128, h=44) -->
  <rect x="32" y="84" width="54" height="44" rx="5" fill="#F1F5F9" stroke="#94A3B8" stroke-width="1.0"/>
  <text x="59" y="103" text-anchor="middle" font-size="10.0" font-weight="600" fill="#000000">Turn</text>
  <text x="59" y="117" text-anchor="middle" font-size="10.0" font-style="italic" fill="#000000">k&#x2212;1</text>

  <rect x="92" y="84" width="234" height="44" rx="5" fill="#FAFAFA" stroke="#94A3B8" stroke-width="1.0" stroke-dasharray="4,3"/>
  <rect x="114" y="99" width="3.5" height="12" rx="1" fill="#475569"/>
  <rect x="120.5" y="99" width="3.5" height="12" rx="1" fill="#475569"/>
  <text x="214" y="103" text-anchor="middle" font-size="10.4" font-weight="600" fill="#000000">Agent Pause (0.71&#x2013;2.22 s)</text>
  <text x="214" y="118" text-anchor="middle" font-size="9.4" font-weight="400" fill="#000000">OCS Mirror Idle &amp; Unused</text>

  <rect x="338" y="84" width="96" height="44" rx="5" fill="url(#hatchVerm)" stroke="#D55E00" stroke-width="1.25"/>
  <text x="386" y="103" text-anchor="middle" font-size="10.2" font-weight="700" fill="#000000">MEMS Stall</text>
  <text x="386" y="118" text-anchor="middle" font-size="9.6" font-weight="600" fill="#000000">25&#x2013;50 ms</text>

  <rect x="440" y="84" width="50" height="44" rx="5" fill="#FEF3D6" stroke="#E69F00" stroke-width="1.1"/>
  <text x="465" y="103" text-anchor="middle" font-size="9.8" font-weight="600" fill="#000000">KV</text>
  <text x="465" y="117" text-anchor="middle" font-size="9.4" fill="#000000">Xfer</text>

  <rect x="496" y="84" width="56" height="44" rx="5" fill="#F1F5F9" stroke="#64748B" stroke-width="1.0"/>
  <text x="524" y="103" text-anchor="middle" font-size="9.8" font-weight="600" fill="#000000">Turn <tspan font-style="italic">k</tspan></text>
  <text x="524" y="117" text-anchor="middle" font-size="9.4" fill="#000000">Prefill</text>

  <line x1="338" y1="135" x2="338" y2="147" stroke="#D55E00" stroke-width="1.0"/>
  <line x1="552" y1="135" x2="552" y2="147" stroke="#D55E00" stroke-width="1.0"/>
  <line x1="341" y1="142" x2="549" y2="142" stroke="#D55E00" stroke-width="1.35" marker-start="url(#arrVermL)" marker-end="url(#arrVerm)"/>
  <text x="445" y="162" text-anchor="middle" font-size="10.2" font-weight="700" fill="#000000">Exposed Critical-Path Thrashing</text>
  <text x="209" y="162" text-anchor="middle" font-size="9.4" font-weight="400" fill="#000000">No prefetch during pause</text>

  <!-- LANE 2 CARD: OptiKV-Fabric (Ours) (y: 208..348, h=140) -->
  <g filter="url(#heroShadow)">
    <rect x="14" y="208" width="552" height="140" rx="8" fill="#FFFFFF" stroke="#B6D6EC" stroke-width="1.25"/>
    <rect x="14" y="208" width="6" height="140" rx="2.5" fill="#0072B2"/>
  </g>
  <text x="32" y="232" font-size="11.6" font-weight="700" fill="#000000">OptiKV-Fabric (Ours)</text>
  <text x="158" y="232" font-size="10.0" font-weight="400" fill="#000000">OCS Port Calendar + GapFetch</text>
  <rect x="394" y="217" width="158" height="22" rx="11" fill="#DCEEFB" stroke="#0072B2" stroke-width="0.95"/>
  <text x="473" y="231.5" text-anchor="middle" font-size="9.8" font-weight="700" fill="#000000">Duty 76&#x2013;96% &#x00B7; 36 ms</text>

  <!-- Lane 2 Gantt Track (y: 253..297, h=44) -->
  <rect x="32" y="253" width="54" height="44" rx="5" fill="#F1F5F9" stroke="#94A3B8" stroke-width="1.0"/>
  <text x="59" y="272" text-anchor="middle" font-size="10.0" font-weight="600" fill="#000000">Turn</text>
  <text x="59" y="286" text-anchor="middle" font-size="10.0" font-style="italic" fill="#000000">k&#x2212;1</text>

  <rect x="92" y="253" width="80" height="44" rx="5" fill="url(#hatchSky)" stroke="#56B4E9" stroke-width="1.1"/>
  <text x="132" y="272" text-anchor="middle" font-size="9.8" font-weight="700" fill="#000000">Pre-Switch</text>
  <text x="132" y="286" text-anchor="middle" font-size="9.2" fill="#000000">(Hidden)</text>

  <rect x="178" y="253" width="88" height="44" rx="5" fill="url(#gradBurst)" stroke="#0072B2" stroke-width="1.1"/>
  <text x="222" y="272" text-anchor="middle" font-size="9.8" font-weight="700" fill="#000000">OCS Burst</text>
  <text x="222" y="286" text-anchor="middle" font-size="9.2" fill="#000000">Prefetch</text>

  <rect x="272" y="253" width="54" height="44" rx="5" fill="#D9F2E9" stroke="#009E73" stroke-width="1.15"/>
  <text x="299" y="272" text-anchor="middle" font-size="9.6" font-weight="700" fill="#000000">Warm</text>
  <text x="299" y="286" text-anchor="middle" font-size="9.4" font-weight="600" fill="#000000">HBM</text>

  <rect x="338" y="253" width="82" height="44" rx="5" fill="url(#gradOpti)" stroke="#004C78" stroke-width="1.2"/>
  <text x="379" y="272" text-anchor="middle" font-size="10.0" font-weight="700" fill="#FFFFFF">Turn <tspan font-style="italic">k</tspan></text>
  <text x="379" y="287" text-anchor="middle" font-size="9.6" font-weight="600" fill="#FFFFFF">&#x0394;-Prefill</text>

  <rect x="426" y="253" width="126" height="44" rx="5" fill="url(#gradSaved)" stroke="#009E73" stroke-width="1.0" stroke-dasharray="4,2.5"/>
  <line x1="436" y1="275" x2="542" y2="275" stroke="#009E73" stroke-width="1.35" marker-start="url(#arrGreenL)" marker-end="url(#arrGreen)"/>
  <text x="489" y="267" text-anchor="middle" font-size="10.0" font-weight="700" fill="#000000">Latency Saved</text>
  <text x="489" y="290" text-anchor="middle" font-size="9.4" font-weight="600" fill="#000000">Matches Oracle (36 ms)</text>

  <!-- Bottom brackets -->
  <line x1="92" y1="304" x2="92" y2="316" stroke="#0072B2" stroke-width="1.0"/>
  <line x1="326" y1="304" x2="326" y2="316" stroke="#0072B2" stroke-width="1.0"/>
  <line x1="95" y1="311" x2="323" y2="311" stroke="#0072B2" stroke-width="1.25" marker-start="url(#arrBlueL)" marker-end="url(#arrBlue)"/>
  <text x="209" y="331" text-anchor="middle" font-size="10.2" font-weight="700" fill="#000000">Switch &amp; Prefetch Hidden in Pause</text>

  <line x1="338" y1="304" x2="338" y2="316" stroke="#0072B2" stroke-width="1.0"/>
  <line x1="420" y1="304" x2="420" y2="316" stroke="#0072B2" stroke-width="1.0"/>
  <line x1="341" y1="311" x2="417" y2="311" stroke="#0072B2" stroke-width="1.25" marker-start="url(#arrBlueL)" marker-end="url(#arrBlue)"/>
  <text x="379" y="331" text-anchor="middle" font-size="10.2" font-weight="700" fill="#000000">0 ms Switch</text>

  <!-- Shared Vertical Arrival Alignment Guide at x = 332 -->
  <line x1="332" y1="74" x2="332" y2="180" stroke="#000000" stroke-width="1.2" stroke-dasharray="3,2.5"/>
  <line x1="332" y1="204" x2="332" y2="306" stroke="#000000" stroke-width="1.2" stroke-dasharray="3,2.5"/>
  <rect x="282" y="181" width="100" height="22" rx="11" fill="#1E293B"/>
  <text x="332" y="195.5" text-anchor="middle" font-size="10.0" font-weight="700" fill="#FFFFFF">Turn <tspan font-style="italic">k</tspan> arrives</text>

  <!-- Time Axis Footer -->
  <line x1="26" y1="368" x2="554" y2="368" stroke="#475569" stroke-width="1.2" marker-end="url(#arrDark)"/>
  <text x="290" y="383" text-anchor="middle" font-size="10.6" font-weight="600" fill="#000000">Wall-clock timeline across multi-turn agent pause (<tspan font-style="italic">t</tspan>)</text>


  <!-- =================================================================== -->
  <!-- PANEL (b): SPACIOUS MINIMALIST PARETO FRONTIER (x: 590..1110)       -->
  <!-- =================================================================== -->
  <text x="871" y="24" text-anchor="middle" font-size="14.2" font-weight="700" fill="#000000">(b) 128K Goodput vs. E2E Fabric Power</text>

  <rect x="{px0:.2f}" y="{py0:.2f}" width="{pareto_x1 - px0:.2f}" height="{pareto_y1 - py0:.2f}" fill="url(#gradPareto)"/>
  <text x="{px0 + 9:.2f}" y="{py0 + 18:.2f}" font-size="10.8" font-weight="700" fill="#000000">Pareto-Optimal</text>

  {"".join(xgrid_svg)}
  {"".join(ygrid_svg)}

  <line x1="{sx(PWR_ELEC_1600G_E2E):.2f}" y1="{py0:.2f}" x2="{sx(PWR_ELEC_1600G_E2E):.2f}" y2="{py1:.2f}" stroke="#888888" stroke-width="1.2" stroke-dasharray="2.5,3.5"/>
  <text x="{sx(PWR_ELEC_1600G_E2E) - 8:.2f}" y="{sy(2.16):.2f}" text-anchor="end" font-size="10.0" font-weight="400" fill="#000000">Matched 200 GB/s</text>
  <text x="{sx(PWR_ELEC_1600G_E2E) - 8:.2f}" y="{sy(1.93):.2f}" text-anchor="end" font-size="10.0" font-weight="400" fill="#000000">Electrical (3.16 kW)</text>
  <text x="{sx(PWR_ELEC_1600G_E2E) - 8:.2f}" y="{sy(1.70):.2f}" text-anchor="end" font-size="9.8" font-weight="700" fill="#000000">4.9&#x00D7;&#x2013;7.8&#x00D7; power</text>

  <line x1="{px0:.2f}" y1="{sy(gp_orc):.2f}" x2="{px1:.2f}" y2="{sy(gp_orc):.2f}" stroke="#000000" stroke-width="1.35" stroke-dasharray="6,4"/>
  <text x="{px1 - 4:.2f}" y="{sy(gp_orc) - 10:.2f}" text-anchor="end" font-size="10.6" font-weight="700" fill="#000000">Oracle ({gp_orc:.2f} req/s)</text>

  <line x1="{px0:.2f}" y1="{py0:.2f}" x2="{px0:.2f}" y2="{py1:.2f}" stroke="#000000" stroke-width="1.2"/>
  <line x1="{px0:.2f}" y1="{py1:.2f}" x2="{px1:.2f}" y2="{py1:.2f}" stroke="#000000" stroke-width="1.2"/>

  <text x="{(px0 + px1)/2:.2f}" y="382" text-anchor="middle" font-size="11.2" font-weight="600" fill="#000000">End-to-end fabric power: switch + NIC optics (W)</text>
  <text x="603" y="{(py0 + py1)/2:.2f}" text-anchor="middle" font-size="11.2" font-weight="600" fill="#000000" transform="rotate(-90 603 {(py0 + py1)/2:.2f})">Peak 128K SLO goodput (req/s)</text>

  <!-- Single thin straight vertical leader ONLY for sandwiched OptiKV 1:1 (646 W) -->
  <line x1="{c_opt11[0]:.2f}" y1="{c_opt11[1] + 9:.2f}" x2="{c_opt11[0]:.2f}" y2="{sy(2.90):.2f}" stroke="#0072B2" stroke-width="1.15"/>

  <rect x="{c_elec200[0]-5:.2f}" y="{c_elec200[1]-5:.2f}" width="10" height="10" fill="#A69060" stroke="#000000" stroke-width="1.0"/>
  <polygon points="{_polygon_points(c_concur[0], c_concur[1], 5.6, 6, 0)}" fill="#CC79A7" stroke="#000000" stroke-width="0.9"/>
  <rect x="{c_rrt[0]-5.4:.2f}" y="{c_rrt[1]-5.4:.2f}" width="10.8" height="10.8" fill="#E69F00" stroke="#000000" stroke-width="1.0"/>
  <polygon points="{_polygon_points(c_rpf[0], c_rpf[1], 6.4, 5, -90)}" fill="#009E73" stroke="#000000" stroke-width="1.0"/>
  <polygon points="{_polygon_points(c_naive[0], c_naive[1], 6.6, 4, 0)}" fill="#D55E00" stroke="#000000" stroke-width="1.0"/>
  <polygon points="{_polygon_points(c_hbm[0], c_hbm[1], 6.2, 3, 90)}" fill="#7F7F7F" stroke="#000000" stroke-width="1.0"/>
  <polygon points="{_polygon_points(c_r50[0], c_r50[1], 7.0, 3, -90)}" fill="#2B6B56" stroke="#000000" stroke-width="1.0"/>
  <circle cx="{c_opt11[0]:.2f}" cy="{c_opt11[1]:.2f}" r="7.0" fill="#0072B2" stroke="#000000" stroke-width="1.1"/>
  <polygon points="{_star_points(c_opt41[0], c_opt41[1], 11.0, 4.8, 5)}" fill="#004C78" stroke="#000000" stroke-width="1.05"/>

  <!-- Direct Minimalist Labels (ZERO redundant arrows, pure black ink!) -->
  <text x="{sx(340):.2f}" y="{sy(3.60):.2f}" text-anchor="middle" font-size="10.0" font-weight="700" fill="#000000">OptiKV 4:1 (404 W)</text>
  <text x="{sx(340):.2f}" y="{sy(3.34):.2f}" text-anchor="middle" font-size="9.5" font-weight="500" fill="#000000">{gp_opt41:.2f} ({gpw_41_r50:.2f}&#x00D7; 50G / {gpw_41:.1f}&#x00D7; 25G)</text>

  <text x="{c_opt11[0]:.2f}" y="{sy(2.68):.2f}" text-anchor="middle" font-size="10.0" font-weight="700" fill="#000000">OptiKV 1:1 (646 W)</text>
  <text x="{c_opt11[0]:.2f}" y="{sy(2.42):.2f}" text-anchor="middle" font-size="9.5" font-weight="400" fill="#000000">50 ms P50</text>

  <text x="{sx(895):.2f}" y="{sy(3.64):.2f}" text-anchor="start" font-size="9.9" font-weight="700" fill="#000000">RDMA-50G (790 W)</text>
  <text x="{sx(895):.2f}" y="{sy(3.38):.2f}" text-anchor="start" font-size="9.5" font-weight="400" fill="#000000">162 ms (3.25&#x00D7; slower)</text>

  <text x="{c_rpf[0] + 13:.2f}" y="{c_rpf[1] + 3.5:.2f}" text-anchor="start" font-size="9.9" fill="#000000">RDMA-25G+Pf ({gp_rpf:.2f})</text>
  <text x="{c_concur[0] + 12:.2f}" y="{c_rrt[1] + 3.5:.2f}" text-anchor="start" font-size="9.8" fill="#000000">RDMA-25G+Rt / CONCUR ({gp_rrt:.2f})</text>

  <text x="{c_naive[0] - 13:.2f}" y="{c_naive[1] + 3.5:.2f}" text-anchor="end" font-size="9.9" fill="#000000">Opt-Naive (0.00)</text>
  <text x="{c_hbm[0] + 12:.2f}" y="{c_hbm[1] + 3.5:.2f}" text-anchor="start" font-size="9.9" fill="#000000">HBM / NoCache (0.00)</text>
</svg>
</body>
</html>
"""


# ===========================================================================
# STANDALONE OVERALL ARCHITECTURE FIGURE (Section III, 1140 x 342 px)
#   Full-width wide-landscape modern systems stack with custom hardware icons!
# ===========================================================================

def build_arch_overview_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
  @page { size: 1140px 342px; margin: 0; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 1140px; height: 342px; background: #FFFFFF; overflow: hidden;
    font-family: "Segoe UI", "Arial", "Helvetica Neue", sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  svg { display: block; width: 1140px; height: 342px; }
</style>
</head>
<body>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1140 342" width="1140" height="342">
  <defs>
    <filter id="cardShadow" x="-4%" y="-6%" width="108%" height="118%">
      <feDropShadow dx="0" dy="1.8" stdDeviation="2.4" flood-color="#0F2942" flood-opacity="0.08"/>
    </filter>
    <filter id="heroShadow" x="-3%" y="-8%" width="106%" height="122%">
      <feDropShadow dx="0" dy="2.5" stdDeviation="3.4" flood-color="#005282" flood-opacity="0.14"/>
    </filter>
    <linearGradient id="gradOpti" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0072B2"/><stop offset="100%" stop-color="#005587"/>
    </linearGradient>
    <linearGradient id="gradBus" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#1E3A54"/><stop offset="50%" stop-color="#294F73"/><stop offset="100%" stop-color="#1E3A54"/>
    </linearGradient>
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
    <marker id="arrGray" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 7 4 L 0 7 z" fill="#64748B"/>
    </marker>
  </defs>

  <rect x="0" y="0" width="1140" height="342" fill="#FFFFFF"/>

  <!-- =================================================================== -->
  <!-- LAYER 1: WORKLOAD INPUT & ROOFLINE LAWS BANNER (y: 10..48, h=38)    -->
  <!-- =================================================================== -->
  <g filter="url(#cardShadow)">
    <rect x="14" y="10" width="1112" height="38" rx="7" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.1"/>
    <rect x="14" y="10" width="6" height="38" rx="2.5" fill="#475569"/>
  </g>
  <!-- Multi-turn session tree icon -->
  <g transform="translate(34, 18)" stroke="#334155" stroke-width="1.4" fill="none">
    <circle cx="5" cy="4.5" r="3" fill="#E2E8F0"/>
    <circle cx="19" cy="4.5" r="3" fill="#E2E8F0"/>
    <circle cx="12" cy="17.5" r="3.2" fill="#0072B2" stroke="#004C78"/>
    <line x1="7" y1="7" x2="10.5" y2="14.5"/>
    <line x1="17" y1="7" x2="13.5" y2="14.5"/>
  </g>
  <text x="70" y="33" font-size="11.8" font-weight="700" fill="#000000">Agentic Multi-Turn Sessions</text>
  <text x="234" y="33" font-size="10.6" font-weight="400" fill="#000000">(Immutable Prefix Block Hashes <tspan font-style="italic">B</tspan><tspan dy="2" font-size="8.0">q</tspan><tspan dy="-2"> &amp; Inter-Turn Pauses &#x0394;</tspan><tspan font-style="italic">t</tspan><tspan dy="2" font-size="8.0">gap</tspan><tspan dy="-2"> = 0.71&#x2013;2.22 s)</tspan></text>
  <!-- Right pill for Laws 1-3 -->
  <rect x="756" y="17" width="356" height="24" rx="12" fill="#E6F2FA" stroke="#0072B2" stroke-width="0.85"/>
  <text x="934" y="32.5" text-anchor="middle" font-size="10.2" font-weight="700" fill="#000000">Governed by Roofline Crossover Laws 1&#x2013;3 (Eqs. 5&#x2013;7)</text>

  <!-- Connector 1 -> 2 -->
  <line x1="570" y1="48" x2="570" y2="66" stroke="#1F2937" stroke-width="1.4" marker-end="url(#arrDark)"/>

  <!-- =================================================================== -->
  <!-- LAYER 2: HERO ANCHOR — MATCHING-CONSERVED OCS PORT CALENDAR         -->
  <!--          (y: 68..122, h=54)                                         -->
  <!-- =================================================================== -->
  <g filter="url(#heroShadow)">
    <rect x="14" y="68" width="1112" height="54" rx="7" fill="#FFFFFF" stroke="#0072B2" stroke-width="1.4"/>
    <rect x="14" y="68" width="1112" height="23" rx="6" fill="url(#gradOpti)"/>
    <rect x="14" y="83" width="1112" height="8" fill="url(#gradOpti)"/>
  </g>
  <!-- Custom 2x2 Optical MEMS Crossbar Icon -->
  <g transform="translate(32, 72.5)" stroke="#FFFFFF" stroke-width="1.25" fill="none">
    <rect x="0" y="0" width="17" height="14" rx="2" stroke-opacity="0.9"/>
    <line x1="3" y1="5" x2="7" y2="2.5"/><line x1="10" y1="5" x2="14" y2="2.5"/>
    <line x1="3" y1="11.5" x2="7" y2="9"/><line x1="10" y1="11.5" x2="14" y2="9"/>
  </g>
  <text x="58" y="84" font-size="11.6" font-weight="700" fill="#FFFFFF">Single Shared State Primitive: Matching-Conserved OCS Port Calendar (K)</text>
  <text x="1110" y="84" text-anchor="end" font-size="10.0" font-weight="600" fill="#E6F4FC">Theorem 1 &amp; Proposition 1 Guarantees</text>

  <!-- 3 clean horizontal spec chips inside hero anchor -->
  <text x="198" y="109" text-anchor="middle" font-size="10.5" font-weight="600" fill="#000000">Bipartite Matching <tspan font-weight="700">A</tspan>(<tspan font-style="italic">t</tspan>) &#x2208; {0,1}<tspan dy="-3" font-size="8.0">P&#x00D7;W</tspan><tspan dy="3"> &#x00B7; |L</tspan><tspan dy="2" font-size="7.8">pool</tspan><tspan dy="-2">| &#x2264; min(</tspan><tspan font-style="italic">W, P</tspan>)</text>
  <line x1="398" y1="97" x2="398" y2="115" stroke="#CBD5E1" stroke-width="1.1"/>
  <text x="584" y="109" text-anchor="middle" font-size="10.5" font-weight="600" fill="#000000">Port / NIC Release Vectors (<tspan font-weight="700">t</tspan><tspan dy="2" font-size="7.8">free</tspan><tspan dy="-2">, </tspan><tspan font-weight="700">t</tspan><tspan dy="2" font-size="7.8">wfree</tspan><tspan dy="-2">) &amp; Standing Latch (&#x03C4;</tspan><tspan dy="2" font-size="7.8">eff</tspan><tspan dy="-2"> = 0)</tspan></text>
  <line x1="772" y1="97" x2="772" y2="115" stroke="#CBD5E1" stroke-width="1.1"/>
  <text x="948" y="109" text-anchor="middle" font-size="10.5" font-weight="600" fill="#000000">Zero-Lookahead Online &#x00B7; &lt;16 MiB RAM &#x00B7; O(<tspan font-style="italic">W</tspan>(<tspan font-style="italic">L</tspan><tspan dy="2" font-size="7.8">q</tspan><tspan dy="-2">+</tspan><tspan font-style="italic">P</tspan>))</text>

  <!-- Clean Orthogonal Fan-out Bus: Layer 2 -> Layer 3 (y: 122..144) -->
  <line x1="570" y1="122" x2="570" y2="133" stroke="#0072B2" stroke-width="1.4"/>
  <line x1="194" y1="133" x2="946" y2="133" stroke="#0072B2" stroke-width="1.4"/>
  <line x1="194" y1="133" x2="194" y2="144" stroke="#0072B2" stroke-width="1.4" marker-end="url(#arrBlue)"/>
  <line x1="570" y1="133" x2="570" y2="144" stroke="#009E73" stroke-width="1.4" marker-end="url(#arrGreen)"/>
  <line x1="946" y1="133" x2="946" y2="144" stroke="#D48800" stroke-width="1.4" marker-end="url(#arrAmber)"/>

  <!-- =================================================================== -->
  <!-- LAYER 3: THREE SPACIOUS CONTROL-PLANE MODULES (y: 147..229, h=82)   -->
  <!--          3 cards across 14..1126 (1112px): w=360, gap=16 -> 14,390,766 -->
  <!-- =================================================================== -->
  <!-- Card 3A: OptiRoute -->
  <g filter="url(#cardShadow)">
    <rect x="14" y="147" width="360" height="82" rx="7" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.05"/>
    <rect x="14" y="147" width="360" height="5" rx="2.5" fill="#0072B2"/>
  </g>
  <g transform="translate(28, 159)" stroke="#0072B2" stroke-width="1.5" fill="none">
    <circle cx="3" cy="8" r="2.3" fill="#0072B2"/>
    <path d="M 5.5 8 L 10 8 L 14.5 3 M 10 8 L 14.5 13"/>
    <circle cx="16.5" cy="3" r="1.9"/><circle cx="16.5" cy="13" r="1.9"/>
  </g>
  <text x="54" y="171" font-size="12.0" font-weight="700" fill="#000000">OptiRoute</text>
  <text x="120" y="171" font-size="10.0" font-weight="400" fill="#000000">Co-Routing &amp; Exact Abort</text>
  <rect x="302" y="158" width="60" height="17" rx="8.5" fill="#E6F2FA"/>
  <text x="332" y="170" text-anchor="middle" font-size="9.0" font-weight="700" fill="#000000">Alg. 1 &#x00B7; &#x00A7;III-B</text>
  <line x1="26" y1="181" x2="362" y2="181" stroke="#E2E8F0" stroke-width="0.9"/>
  <text x="28" y="198" font-size="10.0" fill="#000000">&#x2022; Matching-aware affinity to latched workers (&#x03C4;<tspan dy="2" font-size="7.6">eff</tspan><tspan dy="-2"> = 0)</tspan></text>
  <text x="28" y="216" font-size="10.0" fill="#000000">&#x2022; Pool-first coalescing, DRAM co-steering &amp; Law 1 abort</text>

  <!-- Card 3B: GapFetch -->
  <g filter="url(#cardShadow)">
    <rect x="390" y="147" width="360" height="82" rx="7" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.05"/>
    <rect x="390" y="147" width="360" height="5" rx="2.5" fill="#009E73"/>
  </g>
  <g transform="translate(404, 159)" stroke="#009E73" stroke-width="1.5" fill="none">
    <circle cx="8" cy="8" r="6.8"/><path d="M 8 4 L 8 8 L 11.2 10"/>
  </g>
  <text x="428" y="171" font-size="12.0" font-weight="700" fill="#000000">GapFetch</text>
  <text x="490" y="171" font-size="10.0" font-weight="400" fill="#000000">Causal Pause Prefetching</text>
  <rect x="678" y="158" width="60" height="17" rx="8.5" fill="#DCF4EC"/>
  <text x="708" y="170" text-anchor="middle" font-size="9.0" font-weight="700" fill="#000000">Alg. 2 &#x00B7; &#x00A7;III-B</text>
  <line x1="402" y1="181" x2="738" y2="181" stroke="#E2E8F0" stroke-width="0.9"/>
  <text x="404" y="198" font-size="10.0" fill="#000000">&#x2022; Strictly causal EWMA pause predictor (no lookahead)</text>
  <text x="404" y="216" font-size="10.0" fill="#000000">&#x2022; Congestion-guarded JIT bursts &amp; Strata in-flight join</text>

  <!-- Card 3C: PrefixKeep -->
  <g filter="url(#cardShadow)">
    <rect x="766" y="147" width="360" height="82" rx="7" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.05"/>
    <rect x="766" y="147" width="360" height="5" rx="2.5" fill="#E69F00"/>
  </g>
  <g transform="translate(780, 159)" stroke="#C87D00" stroke-width="1.5" fill="none">
    <path d="M 8 1.5 L 14.5 4 L 14.5 8.8 C 14.5 12.2 11.2 14.4 8 15.4 C 4.8 14.4 1.5 12.2 1.5 8.8 L 1.5 4 Z" fill="#FEF3D6"/>
    <path d="M 5.4 8.4 L 7.3 10.3 L 11.0 6.5"/>
  </g>
  <text x="804" y="171" font-size="12.0" font-weight="700" fill="#000000">PrefixKeep</text>
  <text x="874" y="171" font-size="10.0" font-weight="400" fill="#000000">Lossless Multi-Tier Retention</text>
  <rect x="1054" y="158" width="60" height="17" rx="8.5" fill="#FEF3D6"/>
  <text x="1084" y="170" text-anchor="middle" font-size="9.0" font-weight="700" fill="#000000">Alg. 2 &#x00B7; &#x00A7;III-B</text>
  <line x1="778" y1="181" x2="1114" y2="181" stroke="#E2E8F0" stroke-width="0.9"/>
  <text x="780" y="198" font-size="10.0" fill="#000000">&#x2022; 2nd-observation anchor filter &amp; Bounded Prefix-GDSF</text>
  <text x="780" y="216" font-size="10.0" fill="#000000">&#x2022; Request-unit HBM pinning &amp; 100% lossless 5-tier spill</text>

  <!-- =================================================================== -->
  <!-- LAYER 4: STEEL-NAVY DATA & OPTICAL CONTROL BUS (y: 243..265)        -->
  <!-- =================================================================== -->
  <line x1="194" y1="229" x2="194" y2="243" stroke="#0072B2" stroke-width="1.4"/>
  <line x1="570" y1="229" x2="570" y2="243" stroke="#009E73" stroke-width="1.4"/>
  <line x1="946" y1="229" x2="946" y2="243" stroke="#D48800" stroke-width="1.4"/>

  <rect x="14" y="243" width="1112" height="22" rx="11" fill="url(#gradBus)"/>
  <text x="570" y="257.5" text-anchor="middle" font-size="10.4" font-weight="700" fill="#FFFFFF" letter-spacing="0.25">Five-Tier Lossless Data &amp; Reconfigurable Optical Control Bus (PCIe Gen4/5 + 45&#x2013;800 GB/s WDM MEMS/SiPh OCS)</text>

  <!-- 5 Clean Orthogonal Drops into T1..T5 -->
  <!-- 5 cards across 14..1126 (1112px): w=212, gap=13 -> 14, 239, 464, 689, 914; centers: 120, 345, 570, 795, 1020 -->
  <line x1="120" y1="265" x2="120" y2="279" stroke="#0072B2" stroke-width="1.4" marker-end="url(#arrBlue)"/>
  <line x1="345" y1="265" x2="345" y2="279" stroke="#D48800" stroke-width="1.4" marker-end="url(#arrAmber)"/>
  <line x1="570" y1="265" x2="570" y2="279" stroke="#64748B" stroke-width="1.4" marker-end="url(#arrGray)"/>
  <line x1="795" y1="265" x2="795" y2="279" stroke="#0072B2" stroke-width="1.4" marker-end="url(#arrBlue)"/>
  <line x1="1020" y1="265" x2="1020" y2="279" stroke="#009E73" stroke-width="1.4" marker-end="url(#arrGreen)"/>

  <!-- =================================================================== -->
  <!-- LAYER 5: FIVE SPACIOUS HORIZONTAL HARDWARE TIER CARDS (y: 282..334) -->
  <!-- =================================================================== -->
  <!-- T1: GPU HBM -->
  <g filter="url(#cardShadow)">
    <rect x="14" y="282" width="212" height="52" rx="6" fill="#F4FAFF" stroke="#0072B2" stroke-width="1.2"/>
    <rect x="14" y="282" width="5" height="52" rx="2.5" fill="#0072B2"/>
  </g>
  <g transform="translate(30, 297)" stroke="#0072B2" stroke-width="1.3" fill="none">
    <rect x="3" y="3" width="16" height="16" rx="2" fill="#DCEEFB"/>
    <rect x="7" y="7" width="8" height="8" rx="1" fill="#0072B2"/>
    <line x1="7" y1="0" x2="7" y2="3"/><line x1="11" y1="0" x2="11" y2="3"/><line x1="15" y1="0" x2="15" y2="3"/>
    <line x1="7" y1="19" x2="7" y2="22"/><line x1="11" y1="19" x2="11" y2="22"/><line x1="15" y1="19" x2="15" y2="22"/>
  </g>
  <text x="64" y="303" font-size="11.0" font-weight="700" fill="#000000">T1: GPU HBM</text>
  <text x="64" y="319" font-size="9.6" font-weight="400" fill="#000000">Pinned Hot Tier (0.9&#x2013;3.35 TB/s)</text>

  <!-- T2: Host DRAM -->
  <g filter="url(#cardShadow)">
    <rect x="239" y="282" width="212" height="52" rx="6" fill="#FFFFFF" stroke="#E2C37E" stroke-width="1.1"/>
    <rect x="239" y="282" width="5" height="52" rx="2.5" fill="#E69F00"/>
  </g>
  <g transform="translate(254, 299)" stroke="#C87D00" stroke-width="1.3" fill="none">
    <rect x="0" y="2" width="26" height="12" rx="1.5" fill="#FEF3D6"/>
    <rect x="3.5" y="4.8" width="4.5" height="6.4" fill="#E69F00"/>
    <rect x="10.8" y="4.8" width="4.5" height="6.4" fill="#E69F00"/>
    <rect x="18" y="4.8" width="4.5" height="6.4" fill="#E69F00"/>
    <line x1="5" y1="14" x2="5" y2="17.5"/><line x1="11" y1="14" x2="11" y2="17.5"/><line x1="17" y1="14" x2="17" y2="17.5"/><line x1="22" y1="14" x2="22" y2="17.5"/>
  </g>
  <text x="290" y="303" font-size="11.0" font-weight="700" fill="#000000">T2: Host DRAM</text>
  <text x="290" y="319" font-size="9.6" font-weight="400" fill="#000000">26&#x2013;50 GB/s PCIe Spill Tier</text>

  <!-- T3: Local NVMe SSD -->
  <g filter="url(#cardShadow)">
    <rect x="464" y="282" width="212" height="52" rx="6" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.1"/>
    <rect x="464" y="282" width="5" height="52" rx="2.5" fill="#64748B"/>
  </g>
  <g transform="translate(480, 298)" stroke="#475569" stroke-width="1.3" fill="none">
    <rect x="1" y="2" width="24" height="15" rx="2" fill="#F1F5F9"/>
    <line x1="5" y1="7" x2="16" y2="7"/><line x1="5" y1="11.5" x2="13" y2="11.5"/>
    <circle cx="20" cy="12" r="1.6" fill="#475569"/>
  </g>
  <text x="515" y="303" font-size="11.0" font-weight="700" fill="#000000">T3: Local SSD</text>
  <text x="515" y="319" font-size="9.6" font-weight="400" fill="#000000">NVMe Cold Archive (3 GB/s)</text>

  <!-- T4: Peer Remote HBM -->
  <g filter="url(#cardShadow)">
    <rect x="689" y="282" width="212" height="52" rx="6" fill="#FFFFFF" stroke="#9DD2F2" stroke-width="1.1"/>
    <rect x="689" y="282" width="5" height="52" rx="2.5" fill="#56B4E9"/>
  </g>
  <g transform="translate(703, 299)" stroke="#0072B2" stroke-width="1.3" fill="none">
    <rect x="0" y="3" width="10" height="11" rx="1.5" fill="#E6F4FC"/>
    <rect x="18" y="3" width="10" height="11" rx="1.5" fill="#E6F4FC"/>
    <line x1="10" y1="8.5" x2="18" y2="8.5" stroke-dasharray="2,1.5"/>
  </g>
  <text x="740" y="303" font-size="11.0" font-weight="700" fill="#000000">T4: Peer HBM</text>
  <text x="740" y="319" font-size="9.6" font-weight="400" fill="#000000">Direct Peer-to-Peer OCS Hop</text>

  <!-- T5: Disaggregated Optical KV Pool -->
  <g filter="url(#cardShadow)">
    <rect x="914" y="282" width="212" height="52" rx="6" fill="#F2FBF7" stroke="#009E73" stroke-width="1.2"/>
    <rect x="914" y="282" width="5" height="52" rx="2.5" fill="#009E73"/>
  </g>
  <g transform="translate(929, 297)" stroke="#009E73" stroke-width="1.3" fill="none">
    <rect x="2" y="2" width="22" height="5.5" rx="1.2" fill="#D9F2E9"/>
    <rect x="2" y="9.5" width="22" height="5.5" rx="1.2" fill="#D9F2E9"/>
    <path d="M 4 18.5 Q 8.5 15.2 13 18.5 T 22 18.5" stroke="#0072B2" stroke-width="1.4"/>
  </g>
  <text x="964" y="303" font-size="11.0" font-weight="700" fill="#000000">T5: Optical Pool</text>
  <text x="964" y="319" font-size="9.6" font-weight="400" fill="#000000">45&#x2013;800 GB/s Shared OCS Pool</text>
</svg>
</body>
</html>
"""


def main() -> int:
    browser = _find_browser()
    _render_html_to_pdf_png(browser, build_fig1_teaser_html(), "fig1_teaser")
    _render_html_to_pdf_png(browser, build_arch_overview_html(), "fig_arch_overview")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
