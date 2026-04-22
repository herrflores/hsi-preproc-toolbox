"""
Dark current subtraction (Section 2.3.2 of the paper).

Implements per-band subtraction of an additive sensor baseline estimated from
same-day dark reference frames acquired with the sensor optically shielded.

Design decision
---------------
Unlike some implementations that clip negative values to zero immediately
after dark correction, this module **preserves the full DN range** including
slightly negative values. Negative-value policy is applied only at the final
QC stage. This keeps the preprocessing chain scientifically transparent and
avoids hiding information that may indicate calibration issues.

References
----------
Letexier, D., Bourennane, S. (2008). Noise removal from hyperspectral images
by multidimensional filtering. IEEE Trans. Geosci. Remote Sensing, 46,
2061–2069.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import Datacube


@dataclass
class DarkSignalStatistics:
    """
    Summary statistics of the dark signal used for subtraction.

    Attributes
    ----------
    mean_per_band : np.ndarray
        Mean dark DN per band (length n_bands).
    std_per_band : np.ndarray
        Standard deviation per band.
    mean_overall : float
        Mean across bands in the usable range (diagnostic only).
    std_overall : float
        Std of per-band means (diagnostic only).
    n_pixels : int
        Number of spatial pixels in the dark frame used for the estimate.
    """

    mean_per_band: np.ndarray
    std_per_band: np.ndarray
    mean_overall: float
    std_overall: float
    n_pixels: int


def estimate_dark_signal(dark_cube: Datacube) -> DarkSignalStatistics:
    """
    Estimate per-band mean dark DN from a dark reference acquisition.

    Protocol (reference implementation)
    ------------------------------------
    - Dark reference acquired with the sensor optically shielded
    - Same integration time and gain as the survey acquisition
    - Same-day acquisition to avoid temperature drift
    - Mean computed per band across all spatial pixels

    Parameters
    ----------
    dark_cube : Datacube

    Returns
    -------
    DarkSignalStatistics
    """
    flat = dark_cube.data.reshape(-1, dark_cube.n_bands).astype(np.float64)
    mean_per_band = flat.mean(axis=0)
    std_per_band = flat.std(axis=0)

    return DarkSignalStatistics(
        mean_per_band=mean_per_band,
        std_per_band=std_per_band,
        mean_overall=float(mean_per_band.mean()),
        std_overall=float(mean_per_band.std()),
        n_pixels=flat.shape[0],
    )


def subtract_dark(raw_cube: Datacube, dark_stats: DarkSignalStatistics) -> Datacube:
    """
    Subtract the per-band mean dark signal from a raw hyperspectral cube.

    Implements:

        DN_dc(x, y, λ) = DN_raw(x, y, λ) − DN_dark(λ)

    The output is kept in float32 and **not clipped**. Downstream stages
    (ELC, QC) handle out-of-range values explicitly.

    Parameters
    ----------
    raw_cube : Datacube
    dark_stats : DarkSignalStatistics

    Returns
    -------
    Datacube
        Dark-corrected cube (float32).

    Raises
    ------
    ValueError
        If band counts between cubes differ.
    """
    if dark_stats.mean_per_band.shape[0] != raw_cube.n_bands:
        raise ValueError(
            f"Band mismatch: raw cube has {raw_cube.n_bands} bands, "
            f"dark reference has {dark_stats.mean_per_band.shape[0]}."
        )

    corrected = raw_cube.data.astype(np.float32) - dark_stats.mean_per_band.astype(np.float32)

    new_metadata = dict(raw_cube.metadata)
    new_metadata["preprocessing_dark_subtraction"] = {
        "applied": True,
        "mean_dark_dn_overall": dark_stats.mean_overall,
        "std_dark_dn_overall": dark_stats.std_overall,
        "n_pixels": dark_stats.n_pixels,
        "clipping_applied": False,
    }

    return Datacube(
        data=corrected,
        wavelengths=raw_cube.wavelengths,
        metadata=new_metadata,
        source_path=raw_cube.source_path,
    )
