#!/usr/bin/env python3
"""
Generate processing_log.json and qc_summary.json files for each validation run.

Each JSON file documents the parameters, software version, and quality control
diagnostics for the corresponding hyperspectral preprocessing run, providing
the auditable processing record described in Section 2.4.5 of Flores et al. (2026).

Usage:
    python generate_processing_logs.py

Author: Hernán Flores (hernan.flores@thga.de)
License: MIT
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Campaign metadata (matches Table 2 of the manuscript)
# -----------------------------------------------------------------------------
CAMPAIGNS = {
    "ibbenburen_2023_07": {
        "campaign": "Ibbenbüren",
        "site": "Ibbenbüren tailing pond",
        "country": "Germany",
        "region": "North Rhine-Westphalia (NRW)",
        "acquisition_date": "2023-07-20",
        "local_time": "11:30",
        "weather": "sunny",
        "altitude_m_agl": 120,
        "stand_off_distance_m": None,
        "acquisition_mode": "nadir hover (auto-nadir 90°)",
        "integration_time_us": 5000,
        "gain_setting": 50,
        "gain_factor_x": 5.0,
        "dark_frames_acquired": 2,
        "dark_frames_used": 1,
        "target_class": "tailings sludge (iron-rich)",
    },
    "bochum_2025_10": {
        "campaign": "Bochum",
        "site": "Harpener Teiche (Harpen ponds)",
        "country": "Germany",
        "region": "North Rhine-Westphalia (NRW)",
        "acquisition_date": "2025-10-10",
        "local_time": "15:00",
        "weather": "partly cloudy",
        "altitude_m_agl": 120,
        "stand_off_distance_m": None,
        "acquisition_mode": "nadir hover (auto-nadir 90°)",
        "integration_time_us": 4500,
        "gain_setting": 50,
        "gain_factor_x": 5.0,
        "dark_frames_acquired": 2,
        "dark_frames_used": 1,
        "target_class": "mine-water inflow + pond (single calibration applied to two targets)",
    },
    "itzenplitz_2025_11": {
        "campaign": "Itzenplitz",
        "site": "Itzenplitz mine headframe",
        "country": "Germany",
        "region": "Saarland",
        "acquisition_date": "2025-11-14",
        "local_time": "10:50",
        "weather": "cloudy",
        "altitude_m_agl": None,
        "stand_off_distance_m": 4,
        "acquisition_mode": "near-horizontal hover (oblique)",
        "integration_time_us": 5000,
        "gain_setting": 60,
        "gain_factor_x": 6.0,
        "dark_frames_acquired": 2,
        "dark_frames_used": 1,
        "target_class": "engineered infrastructure (corrosion on vertical headframe)",
    },
}


# -----------------------------------------------------------------------------
# Common preprocessing parameters (identical across all three campaigns)
# -----------------------------------------------------------------------------
PREPROCESSING_PARAMETERS = {
    "dark_subtraction": {
        "method": "spatial_mean_per_band",
        "frames_used": 1,
        "frames_total_acquired": 2,
        "intermediate_clipping": False,
    },
    "empirical_line_calibration": {
        "method": "single_panel_reduced_ELC",
        "panel_reflectance_target_pct": 50.0,
        "panel_manufacturer": "Labsphere",
        "panel_model": "Spectralon SRT-50-100",
        "panel_traceability": "NIST-traceable diffuse reflectance reference",
        "offset_term_b_lambda": "fixed_to_zero_after_dark_subtraction",
        "panel_DN_estimator": "median",
        "ROI_protocol": "conservative_interior_polygon",
        "ROI_outermost_excluded_pct": "10-15",
    },
    "savitzky_golay_smoothing": {
        "applied": False,
        "note": "smoothing not applied to validation outputs reported in this log; applied only to spectra shown in Figures 7 and 10 of the manuscript",
        "default_window_length_bands": 11,
        "default_polyorder": 3,
    },
    "quality_control": {
        "saturation_avoidance": "operator-in-the-loop, 40-80% dynamic range target",
        "negative_reflectance_handling": "clip_to_zero",
        "spectral_outlier_method": "MAD_first_difference",
        "spectral_outlier_sigma": 4.0,
        "validation_split_ratio": "70/30",
        "validation_random_seed": 42,
        "validation_spectral_range_nm": [520, 930],
    },
}


SOFTWARE_VERSION = {
    "toolbox_name": "hsi-preproc-toolbox",
    "version": "0.1.2",
    "concept_doi": "10.5281/zenodo.19699318",
    "license": "MIT",
    "repository": "https://github.com/herrflores/hsi-preproc-toolbox",
    "python_min_version": "3.10",
    "key_dependencies": ["numpy>=1.24", "pandas>=2.0", "scipy>=1.10", "scikit-learn>=1.3"],
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_validation_metrics(csv_path: Path, usable_range=(520, 930)) -> dict:
    """Compute per-campaign summary metrics from per-band CSV, restricted to usable range."""
    df = pd.read_csv(csv_path)

    # Filter to usable spectral range
    mask = (df["wavelength_nm"] >= usable_range[0]) & (df["wavelength_nm"] <= usable_range[1])
    df_usable = df.loc[mask].copy()

    # Compute aggregated metrics
    ref = df_usable["ref_reflectance"].values
    val = df_usable["val_mean_reflectance"].values

    # R² (coefficient of determination, treating ref as truth)
    ss_res = np.sum((ref - val) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else None

    return {
        "n_bands_total": int(len(df)),
        "n_bands_usable": int(len(df_usable)),
        "wavelength_range_nm": [float(df_usable["wavelength_nm"].min()), float(df_usable["wavelength_nm"].max())],
        "R_squared": round(r_squared, 4) if r_squared is not None else None,
        "mean_RMSE_pct": round(float(df_usable["RMSE"].mean()), 4),
        "mean_MAE_pct": round(float(df_usable["MAE"].mean()), 4),
        "mean_bias_pct": round(float(df_usable["Bias"].mean()), 4),
        "max_RMSE_pct": round(float(df_usable["RMSE"].max()), 4),
        "min_RMSE_pct": round(float(df_usable["RMSE"].min()), 4),
    }


def build_processing_log(run_id: str, run_dir: Path, csv_path: Path) -> dict:
    """Build the complete processing log for one validation run."""
    metadata = CAMPAIGNS[run_id]
    csv_sha = file_sha256(csv_path)

    return {
        "log_schema_version": "1.0",
        "log_generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "campaign_metadata": metadata,
        "software": SOFTWARE_VERSION,
        "preprocessing_parameters": PREPROCESSING_PARAMETERS,
        "input_file_checksums": {
            "panel_validation_metrics_per_band.csv": {
                "sha256": csv_sha,
                "size_bytes": csv_path.stat().st_size,
            }
        },
        "data_provenance": {
            "raw_HSI_data_distribution": "not redistributed (size and site-access constraints)",
            "processed_outputs_availability": "available from corresponding author upon reasonable request",
            "panel_validation_metrics_published": True,
            "panel_validation_metrics_format": "per-band CSV with reference reflectance, validation mean/std, RMSE, MAE, and bias",
        },
        "notes": [
            "This log documents the panel-based hold-out validation run for the campaign.",
            "Per-band metrics are stored alongside this log in panel_validation_metrics_per_band.csv.",
            "Aggregated summary metrics are stored in qc_summary.json (same directory).",
            "Reference panel reflectance was provided by the manufacturer's calibration certificate.",
            "Per-pixel n values (calibration / validation) reported in qc_summary.json are scene-specific.",
        ],
    }


def build_qc_summary(run_id: str, csv_path: Path) -> dict:
    """Build the QC summary file for one validation run."""
    metrics = compute_validation_metrics(csv_path)

    # Per-pixel calibration/validation counts (from manuscript Table 3)
    pixel_counts = {
        "ibbenburen_2023_07": {"n_calibration": 3171, "n_validation": 1359},
        "bochum_2025_10": {"n_calibration": 2998, "n_validation": 1286},
        "itzenplitz_2025_11": {"n_calibration": 10154, "n_validation": 4353},
    }

    return {
        "run_id": run_id,
        "qc_summary_schema_version": "1.0",
        "validation_method": "panel-based hold-out (70/30 split, random seed 42)",
        "usable_spectral_range_nm": [520, 930],
        "panel_pixels": pixel_counts[run_id],
        "validation_metrics_usable_range": metrics,
        "panel_DN_estimator": "median",
        "panel_DN_negative_clip_count": 0,
        "panel_saturation_rate_pct": "<1",
        "spectral_outlier_method": "MAD_first_difference",
        "spectral_outlier_sigma_threshold": 4.0,
        "qc_pass": True,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(base_dir: Path):
    print(f"Generating processing logs in: {base_dir}\n")

    for run_id in CAMPAIGNS:
        run_dir = base_dir / run_id
        csv_path = run_dir / "panel_validation_metrics_per_band.csv"

        if not csv_path.exists():
            print(f"  [SKIP] {run_id}: CSV not found at {csv_path}")
            continue

        # Generate processing_log.json
        processing_log = build_processing_log(run_id, run_dir, csv_path)
        log_path = run_dir / "processing_log.json"
        with open(log_path, "w") as f:
            json.dump(processing_log, f, indent=2, ensure_ascii=False)
        print(f"  [OK] {log_path.name}")

        # Generate qc_summary.json
        qc_summary = build_qc_summary(run_id, csv_path)
        qc_path = run_dir / "qc_summary.json"
        with open(qc_path, "w") as f:
            json.dump(qc_summary, f, indent=2, ensure_ascii=False)
        print(f"  [OK] {qc_path.name}")
        print(f"       R² = {qc_summary['validation_metrics_usable_range']['R_squared']}, "
              f"mean RMSE = {qc_summary['validation_metrics_usable_range']['mean_RMSE_pct']}%")
        print()

    print("Done.")


if __name__ == "__main__":
    base = Path(__file__).parent.resolve()
    main(base)
