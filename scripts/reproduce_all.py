#!/usr/bin/env python3
"""One-command end-to-end reproducibility pipeline for OptiKV-Fabric.

Runs:
1. Physical conservation & <10% MAPE hardware calibration pytest gate.
2. Master parallel simulation suite (Suites 1-6).
3. Machine-verified LaTeX table & number export.
4. All 8 publication-grade figures (PDF + 300-DPI PNG).
5. LaTeX paper build (pdflatex -> bibtex -> pdflatex -> pdflatex).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=str(cwd))
    if res.returncode != 0:
        raise SystemExit(f"Step failed ({res.returncode}): {' '.join(cmd)}")


def main() -> int:
    run([PY, "-m", "pytest", "tests/", "-v"])
    run([PY, "scripts/run_full_suite.py"])
    run([PY, "scripts/run_suites_3_to_6_refined.py"])
    run([PY, "scripts/export_paper_tables.py"])
    run([PY, "scripts/make_publication_figures.py"])
    if shutil.which("pdflatex") and shutil.which("bibtex"):
        paper_dir = ROOT / "paper"
        run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=paper_dir)
        run(["bibtex", "main"], cwd=paper_dir)
        run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=paper_dir)
        run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=paper_dir)
    run([PY, "scripts/check_assets.py"])
    print("\n=== FULL PIPELINE REPRODUCED CLEANLY ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
