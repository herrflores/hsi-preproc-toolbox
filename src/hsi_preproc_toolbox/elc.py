"""
Empirical line calibration (Section 2.3.3 of the paper).

Single-panel reduced ELC that converts dark-corrected digital numbers (DN)
to surface reflectance using a NIST-traceable reference panel, with the
additive term fixed to zero after dark subtraction.

General ELC formulation:

    R(λ) = a(λ) · DN_dc(λ) + b(λ)

For this reduced implementation, b(λ) = 0, and:

    a(λ) = R_ref(λ) / DN_panel(λ)

where ``DN_panel(λ)`` is the **per-band median** of the panel ROI after
dark correction. The median is preferred over the mean because it is more
robust to outliers from edge mixing, shadowed panel borders, and
occasional hot pixels.

Output reflectance is returned as a fraction in [0, 1]. Values outside this
range are **not clipped** at this stage; the QC stage is responsible for
enforcing physical plausibility explicitly.

References
----------
Smith, G.M., Milton, E.J. (1999). The use of the empirical line method to
calibrate remotely sensed data to reflectance. IJRS, 20, 2653–2662.

Wang, C., Myint, S.W. (2015). A simplified empirical line method of
radiometric calibration for small UAS-based remote sensing. IEEE JSTARS,
8, 1876–1885.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import Datacube


@dataclass
class ELCResult:
    """
    Output of empirical line calibration.

    Attributes
    ----------
    reflectance : Datacube
        Reflectance-calibrated cube (fraction, [0, 1]); no clipping applied.
    gain_per_band : np.ndarray
        Wavelength-dependent gain factors a(λ).
    panel_dn_median : np.ndarray
        Per-band median DN within the panel ROI (after dark correction).
    reference_reflectance : np.ndarray
        Reference panel reflectance (fraction) interpolated to sensor
        wavelengths.
    n_panel_pixels : int
        Number of panel pixels used.
    """

    reflectance: Datacube
    gain_per_band: np.ndarray
    panel_dn_median: np.ndarray
    reference_reflectance: np.ndarray
    n_panel_pixels: int


def compute_gain_factors(
    panel_dn_values: np.ndarray,
    reference_reflectance: np.ndarray,
    *,
    min_panel_dn: float = 1e-6,
) -> np.ndarray:
    """
    Compute per-band gain factors a(λ) = R_ref(λ) / DN_panel(λ)
    using the **median** of panel DN.

    Parameters
    ----------
    panel_dn_values : np.ndarray
        Per-band dark-corrected DN of the panel, shape (pixels, bands) or
        (bands,).
    reference_reflectance : np.ndarray
        Reference reflectance in [0, 1] at sensor wavelengths.
    min_panel_dn : float
        Minimum DN accepted for a valid gain. Bands whose median falls
        below this threshold return NaN in the gain vector; downstream QC
        flags the affected bands.

    Returns
    -------
    np.ndarray
        Per-band gain, shape (bands,).
    """
    if panel_dn_values.ndim == 2:
        panel_median = np.median(panel_dn_values, axis=0)
    elif panel_dn_values.ndim == 1:
        panel_median = panel_dn_values
    else:
        raise ValueError(
            f"panel_dn_values must be 1D or 2D; got shape {panel_dn_values.shape}."
        )

    if panel_median.shape != reference_reflectance.shape:
        raise ValueError(
            f"Shape mismatch: panel_median {panel_median.shape} vs "
            f"reference_reflectance {reference_reflectance.shape}."
        )

    # Guard against division by near-zero
    safe_dn = np.where(panel_median >= min_panel_dn, panel_median, np.nan)
    gain = reference_reflectance / safe_dn
    return gain


def empirical_line_calibration(
    cube: Datacube,
    panel_roi_mask: np.ndarray,
    reference_wavelengths: np.ndarray,
    reference_reflectance: np.ndarray,
) -> ELCResult:
    """
    Apply single-panel ELC to a dark-corrected hyperspectral cube.

    Parameters
    ----------
    cube : Datacube
        Dark-corrected cube (output of :func:`subtract_dark`).
    panel_roi_mask : np.ndarray
        Boolean mask (rows, cols) selecting panel pixels.
    reference_wavelengths : np.ndarray
        Wavelengths (nm) of the reference spectrum.
    reference_reflectance : np.ndarray
        Reference reflectance in [0, 1] at ``reference_wavelengths``.

    Returns
    -------
    ELCResult

    Raises
    ------
    ValueError
        If shapes are inconsistent or the ROI is empty.
    """
    if panel_roi_mask.shape != cube.spatial_shape:
        raise ValueError(
            f"Panel ROI shape {panel_roi_mask.shape} does not match cube "
            f"spatial shape {cube.spatial_shape}."
        )

    # Interpolate reference spectrum onto sensor wavelengths
    ref_interp = np.interp(cube.wavelengths, reference_wavelengths, reference_reflectance)

    panel_pixels = cube.data[panel_roi_mask]  # (n_panel, bands)
    n_panel = panel_pixels.shape[0]
    if n_panel == 0:
        raise ValueError("Panel ROI mask selected zero pixels.")

    panel_median = np.median(panel_pixels, axis=0)
    gain = compute_gain_factors(panel_median, ref_interp)

    reflectance_data = cube.data.astype(np.float32) * gain.astype(np.float32)

    new_metadata = dict(cube.metadata)
    new_metadata["preprocessing_elc"] = {
        "applied": True,
        "mode": "single_panel_reduced_elc",
        "offset_term": 0.0,
        "panel_statistic": "median",
        "output_units": "fraction",
        "n_panel_pixels": int(n_panel),
        "clipping_applied": False,
    }

    refl_cube = Datacube(
        data=reflectance_data,
        wavelengths=cube.wavelengths,
        metadata=new_metadata,
        source_path=cube.source_path,
    )

    return ELCResult(
        reflectance=refl_cube,
        gain_per_band=gain,
        panel_dn_median=panel_median,
        reference_reflectance=ref_interp,
        n_panel_pixels=n_panel,
    )
