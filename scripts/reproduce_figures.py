"""Check a result CSV against the simulator schema.

Figure drawing belongs to the figure agent. This stub does not invent TTFT
numbers and does not plot provisional roofline output as a measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optikv.metrics import SUMMARY_COLUMNS  # noqa: E402


def validate(path: Path) -> int:
    import polars as pl

    frame = pl.read_csv(path)
    missing = [column for column in SUMMARY_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    if "calibration_status" in frame.columns and (frame["calibration_status"] == "provisional").any():
        print("NOTE: calibration_status=provisional means the 3090 roofline, not a measured TTFT.")
    print(f"validated {frame.height} rows from {path}")
    return frame.height


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an OptiKV-Fabric result CSV")
    parser.add_argument("--csv", default=None, help="Summary CSV produced by scripts/run_simulations.py")
    args = parser.parse_args()
    if args.csv is None:
        print("summary columns:")
        print(",".join(SUMMARY_COLUMNS))
        print("Pass --csv to validate a file. This stub does not draw paper figures.")
        return
    validate(Path(args.csv))


if __name__ == "__main__":
    main()
