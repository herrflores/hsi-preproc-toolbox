"""Smoke tests for dark current subtraction."""

import numpy as np
import pytest

from hsi_preproc_toolbox.dark_correction import estimate_dark_signal, subtract_dark
from hsi_preproc_toolbox.io import Datacube


def make_cube(data: np.ndarray, wavelengths: np.ndarray | None = None) -> Datacube:
    if wavelengths is None:
        wavelengths = np.linspace(502, 997, data.shape[-1])
    return Datacube(data=data.astype(np.float32), wavelengths=wavelengths)


def test_estimate_dark_signal_basic():
    dark_data = np.full((10, 10, 100), 22.0, dtype=np.float32)
    stats = estimate_dark_signal(make_cube(dark_data))

    assert stats.mean_per_band.shape == (100,)
    np.testing.assert_allclose(stats.mean_per_band, 22.0, atol=1e-6)
    assert abs(stats.mean_overall - 22.0) < 1e-6
    assert stats.n_pixels == 100


def test_subtract_dark_removes_offset():
    dark_data = np.full((10, 10, 100), 22.0, dtype=np.float32)
    raw_data = np.full((10, 10, 100), 122.0, dtype=np.float32)

    stats = estimate_dark_signal(make_cube(dark_data))
    corrected = subtract_dark(make_cube(raw_data), stats)

    np.testing.assert_allclose(corrected.data, 100.0, atol=1e-5)


def test_subtract_dark_preserves_negatives():
    """Unlike the legacy notebook, subtract_dark MUST NOT clip to zero."""
    dark_data = np.full((5, 5, 10), 30.0, dtype=np.float32)
    raw_data = np.full((5, 5, 10), 10.0, dtype=np.float32)  # below dark level

    stats = estimate_dark_signal(make_cube(dark_data, np.linspace(502, 997, 10)))
    corrected = subtract_dark(make_cube(raw_data, np.linspace(502, 997, 10)), stats)

    assert (corrected.data < 0).all()
    np.testing.assert_allclose(corrected.data, -20.0, atol=1e-5)


def test_subtract_dark_band_mismatch_raises():
    dark_data = np.full((5, 5, 50), 22.0, dtype=np.float32)
    raw_data = np.full((5, 5, 100), 122.0, dtype=np.float32)

    stats = estimate_dark_signal(
        make_cube(dark_data, wavelengths=np.linspace(502, 997, 50))
    )
    with pytest.raises(ValueError, match="Band mismatch"):
        subtract_dark(make_cube(raw_data), stats)
