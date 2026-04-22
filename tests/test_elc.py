"""Smoke tests for empirical line calibration (single-panel, median-based)."""

import numpy as np
import pytest

from hsi_preproc_toolbox.elc import compute_gain_factors, empirical_line_calibration
from hsi_preproc_toolbox.io import Datacube


def test_compute_gain_factors_basic():
    # Panel DN median = 500, reference reflectance = 0.5 (fraction) → gain = 0.001
    panel_dn = np.full(100, 500.0)
    ref_refl = np.full(100, 0.5)

    gain = compute_gain_factors(panel_dn, ref_refl)
    np.testing.assert_allclose(gain, 0.001, atol=1e-9)


def test_compute_gain_factors_uses_median_for_2d():
    # If 2D input is provided, function should reduce with MEDIAN (not mean)
    # Construct panel with an outlier pixel that would shift the mean but not median
    n_pix, n_bands = 100, 50
    panel = np.full((n_pix, n_bands), 500.0)
    panel[0, :] = 1e6  # outlier pixel
    ref_refl = np.full(n_bands, 0.5)

    gain = compute_gain_factors(panel, ref_refl)
    np.testing.assert_allclose(gain, 0.001, atol=1e-9)


def test_elc_recovers_reflectance_in_fraction():
    n_bands = 100
    wavelengths = np.linspace(502, 997, n_bands)

    # Simulate: DN_dc = reflectance * k_inv, k_inv = 1000
    k_inv = 1000.0
    cube_data = np.zeros((20, 20, n_bands), dtype=np.float32)
    cube_data[:10, :10, :] = 0.5 * k_inv   # panel region
    cube_data[10:, 10:, :] = 0.2 * k_inv   # target region

    cube = Datacube(data=cube_data, wavelengths=wavelengths)
    mask = np.zeros((20, 20), dtype=bool)
    mask[:10, :10] = True

    ref_refl = np.full(n_bands, 0.5)  # fraction
    result = empirical_line_calibration(cube, mask, wavelengths, ref_refl)

    np.testing.assert_allclose(result.reflectance.data[:10, :10, :].mean(), 0.5, atol=1e-4)
    np.testing.assert_allclose(result.reflectance.data[10:, 10:, :].mean(), 0.2, atol=1e-4)


def test_elc_empty_roi_raises():
    cube = Datacube(
        data=np.ones((10, 10, 50), dtype=np.float32),
        wavelengths=np.linspace(502, 997, 50),
    )
    empty_mask = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError, match="zero pixels"):
        empirical_line_calibration(cube, empty_mask, np.linspace(502, 997, 50), np.full(50, 0.5))


def test_elc_preserves_negative_values():
    """ELC must not clip; negative reflectance is QC's responsibility."""
    n_bands = 50
    wavelengths = np.linspace(502, 997, n_bands)

    # Add a region with negative dark-corrected DN (simulates post-dark underflow)
    cube_data = np.full((10, 10, n_bands), 500.0, dtype=np.float32)
    cube_data[5:, 5:, :] = -50.0

    cube = Datacube(data=cube_data, wavelengths=wavelengths)
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :5] = True

    result = empirical_line_calibration(cube, mask, wavelengths, np.full(n_bands, 0.5))

    assert (result.reflectance.data[5:, 5:, :] < 0).all()
