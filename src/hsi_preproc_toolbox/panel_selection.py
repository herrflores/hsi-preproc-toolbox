"""
Interactive panel ROI selection utilities.

This module provides a helper to interactively select the calibration panel
region on a displayed band of the calibration cube, analogous to the
``PolygonSelector``-based workflow in the reference notebook. The resulting
polygon vertices can be saved to CSV for reproducibility and subsequently
loaded with :func:`hsi_preproc_toolbox.load_panel_mask`.

Intended for interactive (Jupyter / GUI) use only. In headless pipelines,
pre-saved polygon CSVs or .npy masks should be used instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .io import Datacube


def select_panel_polygon_interactive(
    calibration_cube: Datacube,
    *,
    band_index: int = 50,
    save_to: str | Path | None = None,
    on_close: Callable[[], None] | None = None,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    """
    Display a single band of the calibration cube and let the user draw a
    polygon around the calibration panel. Returns the polygon vertices and
    the resulting boolean mask.

    Parameters
    ----------
    calibration_cube : Datacube
        Dark-corrected calibration panel cube (or raw, depending on
        workflow preference).
    band_index : int
        Band index to display for panel selection.
    save_to : str, Path, or None
        If given, the polygon vertices are saved to a CSV file with columns
        ``x`` and ``y``.
    on_close : callable, optional
        Callback invoked after the polygon is confirmed.

    Returns
    -------
    vertices : list of (x, y)
    mask : np.ndarray
        Boolean mask with shape matching the cube's spatial dimensions.

    Notes
    -----
    This function must be run in an environment with an interactive
    matplotlib backend (e.g., ``%matplotlib qt`` or ``%matplotlib widget``
    in Jupyter).
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.path import Path as MplPath
    from matplotlib.widgets import PolygonSelector

    band_image = calibration_cube.data[:, :, band_index]

    fig, ax = plt.subplots()
    ax.imshow(band_image, cmap="gray")
    ax.set_title(
        "Draw a polygon around the calibration panel (close with right-click)"
    )
    ax.axis("off")

    vertices: list[tuple[float, float]] = []

    def onselect(verts):
        vertices.clear()
        vertices.extend(verts)
        poly = plt.Polygon(verts, closed=True, fill=False, color="red", linewidth=2)
        ax.add_patch(poly)
        plt.draw()

    selector = PolygonSelector(ax, onselect)  # noqa: F841 - keeps event loop alive
    plt.show(block=True)

    if not vertices:
        raise RuntimeError("No polygon vertices captured from the interactive selector.")

    ny, nx = calibration_cube.spatial_shape
    Y, X = np.mgrid[:ny, :nx]
    points = np.vstack((X.flatten(), Y.flatten())).T
    mask = MplPath(vertices).contains_points(points).reshape(ny, nx)

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(vertices, columns=["x", "y"]).to_csv(save_to, index=False)

    if on_close is not None:
        on_close()

    return vertices, mask
