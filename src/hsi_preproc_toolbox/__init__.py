"""
hsi-preproc-toolbox
===================

Reproducible radiometric preprocessing for UAV VNIR hyperspectral imagery
in post-mining environments.

Public API
----------
- :class:`Pipeline`                    — end-to-end orchestrator
- :class:`PipelineConfig`              — configuration dataclass
- :class:`Datacube`                    — hyperspectral cube container
- :func:`load_datacube`                — ENVI cube loader
- :func:`save_datacube`                — ENVI cube writer
- :func:`load_reference_panel_spectrum`— reference panel CSV loader
- :func:`load_panel_mask`              — panel ROI loader (.npy / .csv)
- :func:`subtract_dark`                — per-band dark subtraction
- :func:`empirical_line_calibration`   — single-panel ELC (median)
- :func:`savgol_smooth`                — optional spectral smoothing
- :func:`run_quality_control`          — integrated QC + validation
- :func:`holdout_validation`           — panel hold-out metrics

See the accompanying paper for methodological details:

    Flores, H. (2026). Standardizing UAV Hyperspectral Data for Monitoring
    Post-Mining Environments: A Reproducible Preprocessing Framework.
    Green and Smart Mining Engineering.
"""

__version__ = "0.1.0"
__author__ = "Hernán Flores"
__license__ = "MIT"

from .pipeline import Pipeline, PipelineConfig, PipelineResult
from .batch import batch_process_folder, BatchReport
from .io import (
    Datacube,
    load_datacube,
    save_datacube,
    load_reference_panel_spectrum,
    load_panel_mask,
)
from .dark_correction import subtract_dark, estimate_dark_signal, DarkSignalStatistics
from .elc import empirical_line_calibration, compute_gain_factors, ELCResult
from .smoothing import savgol_smooth
from .qc import (
    run_quality_control,
    holdout_validation,
    filter_negative_reflectance,
    detect_spectral_spikes,
    check_panel_saturation,
    QCReport,
    HoldoutMetrics,
)

__all__ = [
    "__version__",
    # Pipeline
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    # Batch
    "batch_process_folder",
    "BatchReport",
    # IO
    "Datacube",
    "load_datacube",
    "save_datacube",
    "load_reference_panel_spectrum",
    "load_panel_mask",
    # Stages
    "subtract_dark",
    "estimate_dark_signal",
    "DarkSignalStatistics",
    "empirical_line_calibration",
    "compute_gain_factors",
    "ELCResult",
    "savgol_smooth",
    # QC
    "run_quality_control",
    "holdout_validation",
    "filter_negative_reflectance",
    "detect_spectral_spikes",
    "check_panel_saturation",
    "QCReport",
    "HoldoutMetrics",
]
