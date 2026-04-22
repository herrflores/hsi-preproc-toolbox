"""End-to-end smoke tests: synthetic dark + panel + target → reflectance cube."""

import numpy as np
import pytest

from hsi_preproc_toolbox.dark_correction import estimate_dark_signal, subtract_dark
from hsi_preproc_toolbox.elc import empirical_line_calibration
from hsi_preproc_toolbox.io import Datacube
from hsi_preproc_toolbox.qc import (
    detect_spectral_spikes,
    filter_negative_reflectance,
    holdout_validation,
    run_quality_control,
)
from hsi_preproc_toolbox.smoothing import savgol_smooth


def test_end_to_end_synthetic_pipeline():
    rng = np.random.default_rng(42)
    n_bands = 100
    wavelengths = np.linspace(502, 997, n_bands)
    ref_refl = np.full(n_bands, 0.5)

    # Simulated: raw = reflectance * k_inv + dark_offset + noise
    k_inv = 1000.0
    dark_offset = 22.0
    true_refl_map = np.full((40, 40, n_bands), 0.3, dtype=np.float32)
    true_refl_map[:20, :20, :] = 0.5  # panel area

    raw_data = (
        true_refl_map * k_inv + dark_offset
        + rng.normal(0, 0.5, size=true_refl_map.shape).astype(np.float32)
    )
    dark_data = (dark_offset + rng.normal(0, 0.3, size=(30, 30, n_bands))).astype(np.float32)

    raw_cube = Datacube(data=raw_data, wavelengths=wavelengths)
    dark_cube = Datacube(data=dark_data, wavelengths=wavelengths)
    panel_mask = np.zeros((40, 40), dtype=bool)
    panel_mask[:20, :20] = True

    # Stages
    dark_stats = estimate_dark_signal(dark_cube)
    dark_corrected = subtract_dark(raw_cube, dark_stats)
    elc_result = empirical_line_calibration(dark_corrected, panel_mask, wavelengths, ref_refl)
    refl_cube = savgol_smooth(elc_result.reflectance, window_length=11, polyorder=3)

    # Checks
    assert 0.48 < refl_cube.data[:20, :20, :].mean() < 0.52
    assert 0.28 < refl_cube.data[20:, 20:, :].mean() < 0.32


def test_holdout_validation_recovers_high_r2():
    rng = np.random.default_rng(42)
    n_bands = 100
    wavelengths = np.linspace(502, 997, n_bands)
    # Realistic non-flat reference: gentle spectral variation across VNIR,
    # similar in shape to NIST-traceable 50% Spectralon calibration data.
    ref_refl = 0.5 + 0.02 * np.sin(2 * np.pi * np.arange(n_bands) / n_bands)

    # Panel DN proportional to reflectance with small noise
    k_inv = 1000.0
    panel_dn = ref_refl[None, :] * k_inv + rng.normal(0, 1.0, size=(500, n_bands))

    hm = holdout_validation(
        panel_dn,
        ref_refl,
        wavelengths=wavelengths,
        calibration_fraction=0.7,
        random_seed=42,
    )
    assert hm.r2 > 0.5
    assert hm.rmse_mean < 0.01  # RMSE < 1% in fraction scale
    assert hm.n_calibration_pixels > 0
    assert hm.n_validation_pixels > 0


def test_filter_negative_reflectance_clip():
    cube = Datacube(
        data=np.array([[[-0.1, 0.2, 0.5], [0.1, -0.05, 0.3]]], dtype=np.float32),
        wavelengths=np.array([500.0, 700.0, 900.0]),
    )
    clipped, frac = filter_negative_reflectance(cube, policy="clip")
    assert (clipped.data >= 0).all()
    assert frac > 0


def test_filter_negative_reflectance_flag_preserves():
    cube = Datacube(
        data=np.array([[[-0.1, 0.2, 0.5]]], dtype=np.float32),
        wavelengths=np.array([500.0, 700.0, 900.0]),
    )
    flagged, frac = filter_negative_reflectance(cube, policy="flag")
    assert (flagged.data < 0).any()
    assert frac > 0


def test_qc_full_runs_without_error():
    rng = np.random.default_rng(0)
    n_bands = 50
    wavelengths = np.linspace(502, 997, n_bands)

    raw_cube = Datacube(
        data=(rng.random((30, 30, n_bands)) * 500 + 22).astype(np.float32),
        wavelengths=wavelengths,
    )
    refl_cube = Datacube(
        data=(rng.random((30, 30, n_bands)) * 0.5).astype(np.float32),
        wavelengths=wavelengths,
    )
    mask = np.zeros((30, 30), dtype=bool)
    mask[:10, :10] = True

    # Realistic non-flat reference and panel DN consistent with it
    ref_refl = 0.5 + 0.02 * np.sin(2 * np.pi * np.arange(n_bands) / n_bands)
    k_inv = 1000.0
    panel_dn_dc = ref_refl[None, :] * k_inv + rng.normal(0, 2.0, size=(100, n_bands))

    _, report = run_quality_control(
        raw_cube=raw_cube,
        reflectance_cube=refl_cube,
        panel_roi_mask=mask,
        reference_reflectance_on_sensor_wavelengths=ref_refl,
        panel_dn_dark_corrected=panel_dn_dc,
    )

    assert 0 <= report.panel_saturation_rate <= 1
    assert 0 <= report.negative_reflectance_fraction <= 1
    assert 0 <= report.spectral_outlier_fraction <= 1
    assert report.holdout is not None
    assert report.holdout.r2 > 0.5


def test_spectral_spike_detection_returns_mask():
    rng = np.random.default_rng(0)
    n_bands = 30
    cube_data = (rng.random((10, 10, n_bands)) * 0.5).astype(np.float32)
    # Inject a spike in one pixel
    cube_data[5, 5, 15] = 100.0

    cube = Datacube(data=cube_data, wavelengths=np.linspace(500, 1000, n_bands))
    mask = detect_spectral_spikes(cube, sigma=4.0)

    assert mask.shape == (10, 10)
    assert mask[5, 5]
