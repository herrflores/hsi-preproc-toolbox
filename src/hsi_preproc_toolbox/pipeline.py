"""
End-to-end preprocessing pipeline orchestration.

Composes the individual preprocessing stages — dark current subtraction,
empirical line calibration, optional spectral smoothing, and integrated
quality control — into a single reproducible workflow. Each run produces
a JSON-formatted processing log documenting configuration, input file
hashes, environment, and QC diagnostics for traceability across
monitoring campaigns.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from . import __version__
from .dark_correction import estimate_dark_signal, subtract_dark
from .elc import empirical_line_calibration
from .io import (
    Datacube,
    load_datacube,
    load_panel_mask,
    load_reference_panel_spectrum,
    save_datacube,
)
from .qc import QCReport, run_quality_control
from .smoothing import DEFAULT_POLYORDER, DEFAULT_WINDOW_LENGTH, savgol_smooth


@dataclass
class PipelineConfig:
    """
    Configuration for a full preprocessing run. All parameters are logged
    to the processing report for reproducibility.
    """

    # Inputs
    dark_frame_path: str | Path
    panel_reference_path: str | Path   # CSV with wavelength, reflectance
    panel_roi_path: str | Path         # .npy mask or .csv polygon vertices

    # Smoothing
    apply_smoothing: bool = True
    savgol_window: int = DEFAULT_WINDOW_LENGTH   # 11
    savgol_polyorder: int = DEFAULT_POLYORDER    # 3

    # QC — sensor
    sensor_max_dn: float = 1023.0  # HAIP BlackBird V2 is 10-bit
    saturation_dn_fraction: float = 0.98
    saturation_rate_threshold: float = 0.01

    # QC — validation
    calibration_fraction: float = 0.7
    holdout_seed: int = 42
    usable_range_nm: tuple[float, float] = (520.0, 930.0)
    run_holdout: bool = True

    # QC — output policy
    negative_reflectance_policy: Literal["clip", "flag", "none"] = "clip"
    spectral_spike_sigma: float = 4.0

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("dark_frame_path", "panel_reference_path", "panel_roi_path"):
            d[key] = str(d[key])
        return d


@dataclass
class PipelineResult:
    """Output of a pipeline run."""

    reflectance: Datacube
    qc_report: QCReport
    processing_log: dict = field(default_factory=dict)

    def save(self, output_path: str | Path, *, write_log: bool = True) -> Path:
        """
        Save the reflectance cube (ENVI .hdr + .img) and the processing log
        (JSON). Returns the cube header path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        header_path = save_datacube(self.reflectance, output_path)
        if write_log:
            log_path = output_path.with_suffix(".log.json")
            log_path.write_text(json.dumps(self.processing_log, indent=2, default=_json_default))
        return header_path


def _json_default(obj):
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class Pipeline:
    """
    End-to-end preprocessing pipeline orchestrator.

    Example
    -------
    >>> from hsi_preproc_toolbox import Pipeline, PipelineConfig
    >>> config = PipelineConfig(
    ...     dark_frame_path="Calibration/dark_frame.hdr",
    ...     panel_reference_path="Calibration/calibration_panel_HAIP.csv",
    ...     panel_roi_path="Calibration/calibration_panel_polygon.csv",
    ... )
    >>> pipeline = Pipeline(config)
    >>> result = pipeline.run("flight_20251010.hdr")
    >>> result.save("outputs/flight_20251010_reflectance.hdr")
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self, raw_datacube_path: str | Path) -> PipelineResult:
        raw_datacube_path = Path(raw_datacube_path)
        started_at = datetime.now(timezone.utc).isoformat()

        # 1. Load inputs
        raw_cube = load_datacube(raw_datacube_path)
        dark_cube = load_datacube(self.config.dark_frame_path)
        ref_wl, ref_refl = load_reference_panel_spectrum(
            self.config.panel_reference_path, output_units="fraction"
        )
        panel_mask = load_panel_mask(self.config.panel_roi_path, raw_cube.spatial_shape)

        # 2. Dark subtraction (no clipping here)
        dark_stats = estimate_dark_signal(dark_cube)
        dark_corrected = subtract_dark(raw_cube, dark_stats)

        # 3. ELC (no clipping here)
        elc_result = empirical_line_calibration(
            dark_corrected,
            panel_roi_mask=panel_mask,
            reference_wavelengths=ref_wl,
            reference_reflectance=ref_refl,
        )
        refl_cube = elc_result.reflectance

        # 4. Optional smoothing
        if self.config.apply_smoothing:
            refl_cube = savgol_smooth(
                refl_cube,
                window_length=self.config.savgol_window,
                polyorder=self.config.savgol_polyorder,
            )

        # 5. QC (first and only place where clipping may occur)
        panel_dn_dc = dark_corrected.data[panel_mask]
        refl_cube, qc_report = run_quality_control(
            raw_cube=raw_cube,
            reflectance_cube=refl_cube,
            panel_roi_mask=panel_mask,
            reference_reflectance_on_sensor_wavelengths=elc_result.reference_reflectance,
            panel_dn_dark_corrected=panel_dn_dc,
            sensor_max_dn=self.config.sensor_max_dn,
            saturation_dn_fraction=self.config.saturation_dn_fraction,
            saturation_rate_threshold=self.config.saturation_rate_threshold,
            negative_reflectance_policy=self.config.negative_reflectance_policy,
            spectral_spike_sigma=self.config.spectral_spike_sigma,
            calibration_fraction=self.config.calibration_fraction,
            holdout_seed=self.config.holdout_seed,
            usable_range_nm=self.config.usable_range_nm,
            run_holdout=self.config.run_holdout,
        )

        finished_at = datetime.now(timezone.utc).isoformat()

        processing_log = {
            "software": {
                "name": "hsi-preproc-toolbox",
                "version": __version__,
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "timestamps": {
                "started_utc": started_at,
                "finished_utc": finished_at,
            },
            "inputs": {
                "raw_datacube": {
                    "path": str(raw_datacube_path),
                    "sha256": _sha256(raw_datacube_path),
                    "shape": list(raw_cube.shape),
                    "wavelength_range_nm": [
                        float(raw_cube.wavelengths.min()),
                        float(raw_cube.wavelengths.max()),
                    ],
                },
                "dark_frame": {
                    "path": str(self.config.dark_frame_path),
                    "sha256": _sha256(self.config.dark_frame_path),
                },
                "panel_reference": {
                    "path": str(self.config.panel_reference_path),
                    "sha256": _sha256(self.config.panel_reference_path),
                },
                "panel_roi": {
                    "path": str(self.config.panel_roi_path),
                    "n_pixels": int(panel_mask.sum()),
                },
            },
            "config": self.config.to_dict(),
            "diagnostics": {
                "dark_signal_mean_dn_overall": dark_stats.mean_overall,
                "dark_signal_std_dn_overall": dark_stats.std_overall,
                "dark_frame_n_pixels": dark_stats.n_pixels,
                "elc_panel_median_dn": elc_result.panel_dn_median.tolist(),
                "elc_gain_per_band": [
                    float(x) if np.isfinite(x) else None for x in elc_result.gain_per_band
                ],
                "elc_n_panel_pixels": elc_result.n_panel_pixels,
            },
            "qc": qc_report.as_dict(),
        }

        return PipelineResult(
            reflectance=refl_cube,
            qc_report=qc_report,
            processing_log=processing_log,
        )
