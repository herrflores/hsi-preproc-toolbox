"""
End-to-end reproducible example.

Demonstrates the full preprocessing pipeline on a single UAV VNIR
hyperspectral acquisition using the HAIP BlackBird V2 reference datasets.

Usage
-----
    python examples/run_pipeline.py \\
        --raw       data/flight_20251010.hdr \\
        --dark      data/Calibration/dark_frame.hdr \\
        --panel-ref data/Calibration/calibration_panel_HAIP.csv \\
        --panel-roi data/Calibration/calibration_panel_polygon.csv \\
        --out       outputs/flight_20251010_reflectance.hdr

The script writes both the reflectance-calibrated ENVI cube and a JSON
processing log (with SHA-256 input hashes and QC diagnostics) alongside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hsi_preproc_toolbox import Pipeline, PipelineConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw", required=True, type=Path, help="Raw datacube .hdr")
    p.add_argument("--dark", required=True, type=Path, help="Dark reference .hdr")
    p.add_argument(
        "--panel-ref",
        required=True,
        type=Path,
        help="CSV with columns 'wavelength' and 'reflectance' (auto-detects % or fraction)",
    )
    p.add_argument(
        "--panel-roi",
        required=True,
        type=Path,
        help="Panel ROI mask (.npy boolean) or polygon CSV (.csv with columns x, y)",
    )
    p.add_argument("--out", required=True, type=Path, help="Output reflectance .hdr path")
    p.add_argument("--no-smoothing", action="store_true", help="Disable Savitzky–Golay smoothing")
    p.add_argument("--window", type=int, default=11, help="Savitzky–Golay window length (bands)")
    p.add_argument("--polyorder", type=int, default=3, help="Savitzky–Golay polynomial order")
    p.add_argument(
        "--neg-policy",
        choices=["clip", "flag", "none"],
        default="clip",
        help="Policy for negative reflectance values",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = PipelineConfig(
        dark_frame_path=args.dark,
        panel_reference_path=args.panel_ref,
        panel_roi_path=args.panel_roi,
        apply_smoothing=not args.no_smoothing,
        savgol_window=args.window,
        savgol_polyorder=args.polyorder,
        negative_reflectance_policy=args.neg_policy,
    )

    pipeline = Pipeline(config)
    result = pipeline.run(args.raw)

    header_path = result.save(args.out)

    print(f"Wrote reflectance cube: {header_path}")
    print(f"Wrote processing log:   {header_path.with_suffix('.log.json')}")
    print("\nQC summary:")
    print(json.dumps(result.qc_report.as_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
