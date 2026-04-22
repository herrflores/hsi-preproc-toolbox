"""
Integrated quality control and validation (Section 2.3.5 of the paper).

Implements the explicit QC stage embedded within the preprocessing chain:

1. Panel saturation screening (pre-calibration, on raw DN)
2. Panel-based hold-out validation (post-calibration)
3. Negative reflectance policy enforcement (configurable)
4. Spectral outlier / spike detection (diagnostic)

All reflectance values are expected and returned as **fractions in [0, 1]**.
Validation metrics are reported in the fraction scale; percent-scale
equivalents can be derived by multiplying by 100.

Default parameters match the reference implementation used for the paper
datasets:
- calibration fraction = 0.7 (→ hold-out = 0.3), random seed = 42
- usable metric range = 520–930 nm
- saturation DN threshold = 98% of sensor max (sensor-dependent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .io import Datacube


# ---------------------------------------------------------------------------
# Defaults (reference implementation for HAIP BlackBird V2, 10-bit)
# ---------------------------------------------------------------------------

DEFAULT_SENSOR_MAX_DN = 1023  # 10-bit; override per sensor
DEFAULT_SATURATION_DN_FRACTION = 0.98  # → 1002 DN for 10-bit
DEFAULT_SATURATION_RATE_THRESHOLD = 0.01  # 1% of panel ROI
DEFAULT_NEGATIVE_REFLECTANCE_POLICY: Literal["clip", "flag", "none"] = "clip"
DEFAULT_SPECTRAL_SPIKE_SIGMA = 4.0
DEFAULT_CALIBRATION_FRACTION = 0.7
DEFAULT_HOLDOUT_SEED = 42
DEFAULT_USABLE_RANGE_NM = (520.0, 930.0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class HoldoutMetrics:
    """Per-band and summary metrics for panel hold-out validation."""

    wavelengths_nm: np.ndarray
    ref_reflectance: np.ndarray      # fraction
    val_mean_reflectance: np.ndarray  # fraction
    val_std_reflectance: np.ndarray   # fraction
    rmse_per_band: np.ndarray         # fraction
    mae_per_band: np.ndarray
    bias_per_band: np.ndarray
    r2: float
    rmse_mean: float                  # fraction
    mae_mean: float
    usable_range_nm: tuple[float, float]
    n_calibration_pixels: int
    n_validation_pixels: int

    def to_summary_dict(self) -> dict:
        return {
            "r2": float(self.r2),
            "rmse_mean_fraction": float(self.rmse_mean),
            "rmse_mean_percent": float(self.rmse_mean * 100.0),
            "mae_mean_fraction": float(self.mae_mean),
            "usable_range_nm": list(self.usable_range_nm),
            "n_calibration_pixels": int(self.n_calibration_pixels),
            "n_validation_pixels": int(self.n_validation_pixels),
        }


@dataclass
class QCReport:
    """Quality-control diagnostics produced by :func:`run_quality_control`."""

    panel_saturation_rate: float
    panel_saturation_passed: bool
    negative_reflectance_fraction: float
    negative_reflectance_policy_applied: str
    spectral_outlier_fraction: float
    holdout: HoldoutMetrics | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "panel_saturation_rate": float(self.panel_saturation_rate),
            "panel_saturation_passed": bool(self.panel_saturation_passed),
            "negative_reflectance_fraction": float(self.negative_reflectance_fraction),
            "negative_reflectance_policy_applied": self.negative_reflectance_policy_applied,
            "spectral_outlier_fraction": float(self.spectral_outlier_fraction),
            "holdout": self.holdout.to_summary_dict() if self.holdout else None,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Individual QC checks
# ---------------------------------------------------------------------------


def check_panel_saturation(
    raw_cube: Datacube,
    panel_roi_mask: np.ndarray,
    *,
    saturation_dn_threshold: float,
) -> float:
    """
    Fraction of panel ROI pixels with at least one saturated band.

    Parameters
    ----------
    raw_cube : Datacube
        Raw cube (uncorrected DN).
    panel_roi_mask : np.ndarray
    saturation_dn_threshold : float
        Absolute DN threshold above which a band is considered saturated.
    """
    panel_pixels = raw_cube.data[panel_roi_mask]
    if panel_pixels.size == 0:
        return 0.0
    saturated = (panel_pixels >= saturation_dn_threshold).any(axis=1)
    return float(saturated.mean())


def filter_negative_reflectance(
    cube: Datacube,
    policy: Literal["clip", "flag", "none"] = DEFAULT_NEGATIVE_REFLECTANCE_POLICY,
) -> tuple[Datacube, float]:
    """
    Handle negative reflectance values in the final calibrated cube.

    Parameters
    ----------
    cube : Datacube
    policy : {'clip', 'flag', 'none'}
        - 'clip' : negative values set to 0 (default).
        - 'flag' : values preserved; caller handles.
        - 'none' : no action.

    Returns
    -------
    Datacube
        Cube after policy.
    float
        Fraction of pixels that had at least one negative band (pre-policy).
    """
    data = cube.data
    neg_any_band = (data < 0).any(axis=2)
    neg_fraction = float(neg_any_band.mean())

    if policy == "clip":
        new_data = np.clip(data, 0.0, None).astype(np.float32)
    elif policy in ("flag", "none"):
        new_data = data
    else:
        raise ValueError(f"Unknown policy: {policy!r}")

    new_metadata = dict(cube.metadata)
    new_metadata["preprocessing_qc_negative_reflectance"] = {
        "policy": policy,
        "fraction_before_policy": neg_fraction,
    }

    return (
        Datacube(
            data=new_data,
            wavelengths=cube.wavelengths,
            metadata=new_metadata,
            source_path=cube.source_path,
        ),
        neg_fraction,
    )


def detect_spectral_spikes(
    cube: Datacube, *, sigma: float = DEFAULT_SPECTRAL_SPIKE_SIGMA
) -> np.ndarray:
    """
    Detect per-pixel spectral spikes via MAD-based thresholding of first
    spectral differences.

    Returns a boolean mask (rows, cols).
    """
    diff = np.diff(cube.data, axis=-1)
    abs_diff = np.abs(diff)

    med = np.median(abs_diff, axis=(0, 1), keepdims=True)
    mad = np.median(np.abs(abs_diff - med), axis=(0, 1), keepdims=True)
    mad_scaled = 1.4826 * mad
    threshold = sigma * mad_scaled

    # Avoid all-zero thresholds producing spurious flags
    threshold = np.where(threshold > 0, threshold, np.inf)
    per_pixel_flag = (abs_diff > threshold).any(axis=-1)
    return per_pixel_flag


def holdout_validation(
    panel_pixels_dn_dark_corrected: np.ndarray,
    reference_reflectance_fraction: np.ndarray,
    *,
    wavelengths: np.ndarray,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    usable_range_nm: tuple[float, float] = DEFAULT_USABLE_RANGE_NM,
    random_seed: int = DEFAULT_HOLDOUT_SEED,
) -> HoldoutMetrics:
    """
    Panel-based hold-out validation of the ELC calibration.

    Panel pixels are randomly split into a calibration subset (fraction
    ``calibration_fraction``) and a validation subset. Gain factors derived
    from the calibration subset are applied to the validation pixels, and
    per-band metrics are computed against the reference spectrum.

    Metrics are reported globally (R², mean RMSE, mean MAE) over a usable
    wavelength range that excludes edge bands with reduced SNR, and
    per-band for all bands in the sensor range.

    Parameters
    ----------
    panel_pixels_dn_dark_corrected : np.ndarray
        Dark-corrected DN of panel pixels, shape (n_pixels, n_bands).
    reference_reflectance_fraction : np.ndarray
        Reference reflectance (fraction, [0, 1]) at sensor wavelengths,
        shape (n_bands,).
    wavelengths : np.ndarray
        Sensor band wavelengths (nm), shape (n_bands,).
    calibration_fraction : float
        Fraction of panel pixels reserved for calibration.
    usable_range_nm : tuple[float, float]
        Bounds for reporting summary metrics.
    random_seed : int

    Returns
    -------
    HoldoutMetrics
    """
    if panel_pixels_dn_dark_corrected.ndim != 2:
        raise ValueError("panel_pixels must be 2D (n_pixels, n_bands).")
    if not (0 < calibration_fraction < 1):
        raise ValueError("calibration_fraction must be strictly between 0 and 1.")

    # Drop any non-finite rows
    finite_rows = np.isfinite(panel_pixels_dn_dark_corrected).all(axis=1)
    panel = panel_pixels_dn_dark_corrected[finite_rows]

    rng = np.random.default_rng(random_seed)
    n_total = panel.shape[0]
    if n_total < 10:
        raise ValueError(f"Too few finite panel pixels for hold-out: {n_total}.")

    idx = np.arange(n_total)
    rng.shuffle(idx)
    n_cal = int(calibration_fraction * n_total)
    cal_idx = idx[:n_cal]
    val_idx = idx[n_cal:]

    panel_cal = panel[cal_idx]
    panel_val = panel[val_idx]

    dn_cal_median = np.median(panel_cal, axis=0)
    dn_cal_median = np.where(dn_cal_median >= 1e-12, dn_cal_median, np.nan)

    gain = reference_reflectance_fraction / dn_cal_median
    R_val = panel_val * gain[None, :]

    R_val_mean = np.nanmean(R_val, axis=0)
    R_val_std = np.nanstd(R_val, axis=0)

    err = R_val - reference_reflectance_fraction[None, :]
    rmse_per_band = np.sqrt(np.nanmean(err**2, axis=0))
    mae_per_band = np.nanmean(np.abs(err), axis=0)
    bias_per_band = np.nanmean(err, axis=0)

    # Usable-range summary
    lo, hi = usable_range_nm
    usable = (wavelengths >= lo) & (wavelengths <= hi)

    x = reference_reflectance_fraction[usable]
    y = R_val_mean[usable]
    ss_res = np.nansum((x - y) ** 2)
    ref_variance = np.nanvar(x)
    # Guard: R² is undefined when reference has ~zero variance (e.g. flat
    # synthetic spectra or very narrow spectral windows). In that case we
    # return NaN rather than a spurious value dominated by numerical noise.
    if ref_variance < 1e-10:
        r2 = float("nan")
    else:
        ss_tot = np.nansum((x - np.nanmean(x)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot)
    rmse_mean = float(np.nanmean(rmse_per_band[usable]))
    mae_mean = float(np.nanmean(mae_per_band[usable]))

    return HoldoutMetrics(
        wavelengths_nm=wavelengths,
        ref_reflectance=reference_reflectance_fraction,
        val_mean_reflectance=R_val_mean,
        val_std_reflectance=R_val_std,
        rmse_per_band=rmse_per_band,
        mae_per_band=mae_per_band,
        bias_per_band=bias_per_band,
        r2=r2,
        rmse_mean=rmse_mean,
        mae_mean=mae_mean,
        usable_range_nm=usable_range_nm,
        n_calibration_pixels=int(panel_cal.shape[0]),
        n_validation_pixels=int(panel_val.shape[0]),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_quality_control(
    *,
    raw_cube: Datacube,
    reflectance_cube: Datacube,
    panel_roi_mask: np.ndarray,
    reference_reflectance_on_sensor_wavelengths: np.ndarray,  # fraction
    panel_dn_dark_corrected: np.ndarray,
    sensor_max_dn: float = DEFAULT_SENSOR_MAX_DN,
    saturation_dn_fraction: float = DEFAULT_SATURATION_DN_FRACTION,
    saturation_rate_threshold: float = DEFAULT_SATURATION_RATE_THRESHOLD,
    negative_reflectance_policy: Literal[
        "clip", "flag", "none"
    ] = DEFAULT_NEGATIVE_REFLECTANCE_POLICY,
    spectral_spike_sigma: float = DEFAULT_SPECTRAL_SPIKE_SIGMA,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    holdout_seed: int = DEFAULT_HOLDOUT_SEED,
    usable_range_nm: tuple[float, float] = DEFAULT_USABLE_RANGE_NM,
    run_holdout: bool = True,
) -> tuple[Datacube, QCReport]:
    """
    Run the full QC stage on a reflectance-calibrated cube.

    Returns the (possibly modified) reflectance cube and a QC report.
    """
    notes: list[str] = []

    sat_dn_threshold = saturation_dn_fraction * sensor_max_dn
    sat_rate = check_panel_saturation(
        raw_cube, panel_roi_mask, saturation_dn_threshold=sat_dn_threshold
    )
    sat_passed = sat_rate <= saturation_rate_threshold
    if not sat_passed:
        notes.append(
            f"Panel saturation rate {sat_rate:.3%} exceeds threshold "
            f"{saturation_rate_threshold:.1%}; calibration may be biased."
        )

    refl_cube, neg_fraction = filter_negative_reflectance(
        reflectance_cube, policy=negative_reflectance_policy
    )

    spike_mask = detect_spectral_spikes(refl_cube, sigma=spectral_spike_sigma)
    spike_fraction = float(spike_mask.mean())

    holdout_metrics: HoldoutMetrics | None = None
    if run_holdout:
        try:
            holdout_metrics = holdout_validation(
                panel_dn_dark_corrected,
                reference_reflectance_on_sensor_wavelengths,
                wavelengths=raw_cube.wavelengths,
                calibration_fraction=calibration_fraction,
                usable_range_nm=usable_range_nm,
                random_seed=holdout_seed,
            )
        except Exception as exc:  # pragma: no cover
            notes.append(f"Hold-out validation skipped: {exc}")

    report = QCReport(
        panel_saturation_rate=sat_rate,
        panel_saturation_passed=sat_passed,
        negative_reflectance_fraction=neg_fraction,
        negative_reflectance_policy_applied=negative_reflectance_policy,
        spectral_outlier_fraction=spike_fraction,
        holdout=holdout_metrics,
        notes=notes,
    )

    # Inject QC summary into metadata
    new_metadata = dict(refl_cube.metadata)
    new_metadata["preprocessing_qc"] = report.as_dict()
    refl_cube = Datacube(
        data=refl_cube.data,
        wavelengths=refl_cube.wavelengths,
        metadata=new_metadata,
        source_path=refl_cube.source_path,
    )

    return refl_cube, report
