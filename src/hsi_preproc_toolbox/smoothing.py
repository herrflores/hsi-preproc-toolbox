"""
Optional Savitzky–Golay spectral smoothing (Section 2.3.4 of the paper).

Applied along the spectral dimension of reflectance-calibrated cubes to
reduce high-frequency noise while preserving diagnostic absorption feature
shape.

Default parameters (window_length=11, polyorder=3) match the reference
implementation used for the paper datasets (HAIP BlackBird V2, 5 nm
spectral sampling). At 5 nm sampling, a window of 11 bands spans ~55 nm,
which is wider than diagnostic features such as the O₂ A-band (~760 nm,
<10 nm wide) but narrower than broader absorption domains (Fe charge
transfer, red edge, water absorption near 970 nm). Users working on
narrow diagnostic features should consider reducing the window length.

References
----------
Savitzky, A., Golay, M.J.E. (1964). Smoothing and differentiation of data
by simplified least squares procedures. Analytical Chemistry, 36, 1627–1639.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .io import Datacube


# Reference implementation parameters (matching the paper)
DEFAULT_WINDOW_LENGTH = 11
DEFAULT_POLYORDER = 3


def savgol_smooth(
    cube: Datacube,
    *,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    polyorder: int = DEFAULT_POLYORDER,
    mode: str = "nearest",
) -> Datacube:
    """
    Apply Savitzky–Golay smoothing along the spectral dimension.

    Parameters
    ----------
    cube : Datacube
        Reflectance-calibrated cube.
    window_length : int
        Filter window length in number of bands. Must be odd and strictly
        greater than ``polyorder``. Default: 11 bands (~55 nm at 5 nm
        spectral sampling).
    polyorder : int
        Polynomial order. Default: 3.
    mode : str
        Boundary handling passed to ``scipy.signal.savgol_filter``.

    Returns
    -------
    Datacube
        Smoothed cube.

    Raises
    ------
    ValueError
        If window_length is not odd, not greater than polyorder, or exceeds
        the number of bands.
    """
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd; got {window_length}.")
    if window_length <= polyorder:
        raise ValueError(
            f"window_length ({window_length}) must be greater than polyorder ({polyorder})."
        )
    if window_length > cube.n_bands:
        raise ValueError(
            f"window_length ({window_length}) exceeds number of bands ({cube.n_bands})."
        )

    smoothed = savgol_filter(
        cube.data, window_length=window_length, polyorder=polyorder, axis=-1, mode=mode
    ).astype(np.float32)

    new_metadata = dict(cube.metadata)
    new_metadata["preprocessing_smoothing"] = {
        "applied": True,
        "method": "savitzky_golay",
        "window_length_bands": int(window_length),
        "polyorder": int(polyorder),
        "boundary_mode": mode,
    }

    return Datacube(
        data=smoothed,
        wavelengths=cube.wavelengths,
        metadata=new_metadata,
        source_path=cube.source_path,
    )
