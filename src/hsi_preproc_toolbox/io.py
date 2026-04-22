"""
I/O utilities for UAV VNIR hyperspectral datacubes.

Thin wrappers around the ``spectral`` library for reading ENVI-compatible
hyperspectral cubes (.hdr + .img) and writing reflectance-calibrated outputs
with processing metadata.

Notes
-----
The reference implementation targets the HAIP BlackBird V2 (500–1000 nm,
100 bands, 10-bit, BIP interleave). If the ENVI header does not declare
``wavelength``, a linear fallback from 502 to 997 nm is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_WAVELENGTH_MIN_NM = 502.0
DEFAULT_WAVELENGTH_MAX_NM = 997.0
DEFAULT_N_BANDS = 100


@dataclass
class Datacube:
    """
    In-memory representation of a hyperspectral datacube.

    Attributes
    ----------
    data : np.ndarray
        3D array of shape (rows, cols, bands).
    wavelengths : np.ndarray
        1D array of band center wavelengths in nm.
    metadata : dict
        Metadata extracted from / written to the ENVI header.
    source_path : Optional[Path]
        Original header path, if loaded from disk.
    """

    data: np.ndarray
    wavelengths: np.ndarray
    metadata: dict = field(default_factory=dict)
    source_path: Optional[Path] = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    @property
    def n_bands(self) -> int:
        return self.data.shape[2]

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.data.shape[:2]


def _resolve_img_path(hdr_path: Path) -> Path:
    """Given a .hdr path, locate the companion binary file."""
    for suffix in (".img", ".bil", ".bsq", ".bip", ".raw"):
        candidate = hdr_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return hdr_path.with_suffix(".img")  # default


def _extract_wavelengths(img, *, n_bands: int) -> np.ndarray:
    """Extract wavelengths from an ENVI image, with fallback to linear grid."""
    if getattr(img, "bands", None) and img.bands.centers:
        return np.asarray(img.bands.centers, dtype=float)
    return np.linspace(DEFAULT_WAVELENGTH_MIN_NM, DEFAULT_WAVELENGTH_MAX_NM, n_bands)


def load_datacube(hdr_path: str | Path) -> Datacube:
    """
    Load an ENVI hyperspectral datacube.

    Parameters
    ----------
    hdr_path : str or Path
        Path to the ENVI header file (.hdr).

    Returns
    -------
    Datacube
    """
    import spectral.io.envi as envi

    hdr_path = Path(hdr_path)
    img_path = _resolve_img_path(hdr_path)
    img = envi.open(str(hdr_path), str(img_path))
    data = np.asarray(img.load()[:])
    wavelengths = _extract_wavelengths(img, n_bands=data.shape[2])

    return Datacube(
        data=data,
        wavelengths=wavelengths,
        metadata=dict(img.metadata),
        source_path=hdr_path,
    )


def save_datacube(
    cube: Datacube,
    hdr_path: str | Path,
    *,
    description: str = "Output of hsi-preproc-toolbox",
    interleave: str = "bip",
) -> Path:
    """
    Save a datacube to ENVI format (.hdr + .img), preserving wavelengths.

    Parameters
    ----------
    cube : Datacube
    hdr_path : str or Path
        Destination header path.
    description : str
        ENVI description string.
    interleave : str
        ENVI interleave mode. Default 'bip' to match HAIP BlackBird V2 output.

    Returns
    -------
    Path
        Written header path.
    """
    import spectral.io.envi as envi

    hdr_path = Path(hdr_path)
    hdr_path.parent.mkdir(parents=True, exist_ok=True)

    merged_metadata = dict(cube.metadata)
    merged_metadata.update({
        "description": description,
        "wavelength": list(cube.wavelengths.astype(float)),
        "wavelength units": "nm",
        "data type": 4,
        "interleave": interleave,
        "byte order": 0,
    })

    envi.save_image(
        str(hdr_path),
        cube.data.astype(np.float32),
        dtype="float32",
        interleave=interleave,
        metadata=merged_metadata,
        force=True,
    )
    return hdr_path


def load_reference_panel_spectrum(
    path: str | Path,
    *,
    output_units: str = "fraction",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the reference reflectance spectrum of the calibration panel.

    Expected CSV columns (case-insensitive): ``wavelength`` and
    ``reflectance``. Values may be stored as percent (0–100) or fraction
    (0–1); the function auto-detects and converts to ``output_units``.

    Parameters
    ----------
    path : str or Path
        Path to the calibration panel CSV.
    output_units : {'fraction', 'percent'}
        Desired output units.

    Returns
    -------
    wavelengths_nm : np.ndarray
    reflectance : np.ndarray
    """
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "wavelength" not in df.columns or "reflectance" not in df.columns:
        raise ValueError(
            "Reference CSV must contain columns 'wavelength' and 'reflectance' "
            f"(case-insensitive); got {list(df.columns)}."
        )

    wl = df["wavelength"].to_numpy(dtype=float)
    refl = df["reflectance"].to_numpy(dtype=float)

    input_is_percent = refl.max() > 1.5
    if output_units == "fraction":
        if input_is_percent:
            refl = refl / 100.0
    elif output_units == "percent":
        if not input_is_percent:
            refl = refl * 100.0
    else:
        raise ValueError(f"output_units must be 'fraction' or 'percent'; got {output_units!r}.")

    return wl, refl


def load_panel_mask(mask_path: str | Path, spatial_shape: tuple[int, int]) -> np.ndarray:
    """
    Load a panel ROI mask from disk.

    Supported formats:
    - .npy         : boolean / uint8 array matching ``spatial_shape``
    - .csv         : polygon vertices with columns 'x' and 'y', rasterized
                     against ``spatial_shape`` (matches the ``PolygonSelector``
                     workflow used in the reference notebook)

    Parameters
    ----------
    mask_path : str or Path
    spatial_shape : tuple
        (rows, cols) of the target cube.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``spatial_shape``.
    """
    import pandas as pd
    from matplotlib.path import Path as MplPath

    mask_path = Path(mask_path)

    if mask_path.suffix.lower() == ".npy":
        arr = np.load(mask_path)
        if arr.shape != spatial_shape:
            raise ValueError(f"Mask shape {arr.shape} does not match cube {spatial_shape}.")
        return arr.astype(bool)

    if mask_path.suffix.lower() == ".csv":
        df = pd.read_csv(mask_path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "x" not in df.columns or "y" not in df.columns:
            raise ValueError("Polygon CSV must contain columns 'x' and 'y'.")
        vertices = list(zip(df["x"].to_numpy(), df["y"].to_numpy()))
        ny, nx = spatial_shape
        Y, X = np.mgrid[:ny, :nx]
        points = np.vstack((X.flatten(), Y.flatten())).T
        mask = MplPath(vertices).contains_points(points).reshape(ny, nx)
        return mask

    raise ValueError(f"Unsupported mask format: {mask_path.suffix}")
