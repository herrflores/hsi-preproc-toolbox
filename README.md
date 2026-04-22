# hsi-preproc-toolbox

Reproducible radiometric preprocessing for UAV-borne VNIR hyperspectral imagery in post-mining environments.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Overview

`hsi-preproc-toolbox` is an open-source Python package that implements the
standardized preprocessing workflow described in:

> Flores, H., Rudolph, T., Benndorf, J. (2026). *Standardizing UAV
> Hyperspectral Data for Monitoring Post-Mining Environments: A
> Reproducible Preprocessing Framework.* Green and Smart Mining
> Engineering. [DOI pending]

The toolbox transforms raw UAV VNIR hyperspectral digital numbers into
reflectance-calibrated hypercubes through a modular, documented, and
reproducible pipeline:

1. **Dark current subtraction** — per-band additive offset removal from
   same-day dark reference frames (no intermediate clipping).
2. **Empirical line calibration (ELC)** — single-panel reduced ELC using a
   NIST-traceable reference panel, with the **median** of the panel ROI as
   the DN estimator (more robust than the mean) and the offset term fixed
   to zero after dark correction.
3. **Optional Savitzky–Golay smoothing** — window length 11 bands
   (~55 nm at 5 nm sampling), polynomial order 3.
4. **Integrated quality control** — panel saturation screening, panel-based
   hold-out validation (R², RMSE, MAE, Bias per band over a usable
   520–930 nm range), spectral outlier detection, and negative-reflectance
   policy enforcement.

All reflectance values are handled in the fraction scale [0, 1]. Clipping
of non-physical values happens **only at the QC stage**, never silently in
intermediate steps. Every pipeline run produces a JSON processing log with
SHA-256 hashes of all inputs, the full configuration used, and QC
diagnostics, to support traceability across monitoring campaigns.

Although the reference implementation targets the HAIP BlackBird V2 sensor
(500–1000 nm, 100 bands, 10-bit, BIP interleave), the workflow is
structured so that it can be adapted to other UAV hyperspectral systems
when equivalent calibration inputs and sensor metadata are provided.

## Installation

```bash
git clone https://github.com/herrflores/hsi-preproc-toolbox.git
cd hsi-preproc-toolbox
pip install -e .
```

For development (tests, linting):

```bash
pip install -e ".[dev]"
pytest
```

## Quick start

```python
from hsi_preproc_toolbox import Pipeline, PipelineConfig

config = PipelineConfig(
    dark_frame_path="Calibration/dark_frame.hdr",
    panel_reference_path="Calibration/calibration_panel_HAIP.csv",
    panel_roi_path="Calibration/calibration_panel_polygon.csv",
    apply_smoothing=True,
    savgol_window=11,
    savgol_polyorder=3,
)

pipeline = Pipeline(config)
result = pipeline.run("flight_20251010.hdr")

# Reflectance cube (fraction, [0, 1])
reflectance = result.reflectance.data  # (rows, cols, bands)

# QC summary (R², RMSE, saturation rate, etc.)
print(result.qc_report.as_dict())

# Save ENVI cube + JSON processing log
result.save("outputs/flight_20251010_reflectance.hdr")
```

### Batch processing

For processing a full UAV campaign folder (multiple `.hdr` cubes sharing
the same calibration) in one call:

```python
from hsi_preproc_toolbox import PipelineConfig, batch_process_folder

config = PipelineConfig(
    dark_frame_path="Calibration/dark_frame.hdr",
    panel_reference_path="Calibration/calibration_panel_HAIP.csv",
    panel_roi_path="Calibration/calibration_panel_polygon.csv",
)

report = batch_process_folder(
    input_folder="campaign_folder/",
    output_folder="campaign_folder/Reflectance/",
    config=config,
)
print(report.summary())
```

### Interactive panel selection

If the calibration panel ROI has not yet been defined, a helper is
provided for drawing the polygon interactively:

```python
from hsi_preproc_toolbox import load_datacube
from hsi_preproc_toolbox.panel_selection import select_panel_polygon_interactive

cal_cube = load_datacube("Calibration/calibration_panel.hdr")
vertices, mask = select_panel_polygon_interactive(
    cal_cube,
    band_index=50,
    save_to="Calibration/calibration_panel_polygon.csv",
)
```

The saved CSV can then be reused across runs through
`PipelineConfig(panel_roi_path=...)`.

## Repository structure

```
hsi-preproc-toolbox/
├── src/hsi_preproc_toolbox/
│   ├── io.py               # ENVI .hdr/.img I/O
│   ├── dark_correction.py  # Section 2.3.2
│   ├── elc.py              # Section 2.3.3 (single-panel, median-based)
│   ├── smoothing.py        # Section 2.3.4 (Savitzky–Golay)
│   ├── qc.py               # Section 2.3.5 (integrated QC + hold-out)
│   ├── pipeline.py         # End-to-end orchestration with JSON logging
│   ├── batch.py            # Campaign-level batch processing
│   └── panel_selection.py  # Interactive polygon selector (optional)
├── examples/
│   └── run_pipeline.py     # CLI end-to-end example
├── tests/                  # Unit + smoke tests (15 passing)
├── data/README.md          # Expected input formats
└── docs/methodology.md     # Methodology reference
```

## Design decisions

Three choices encoded in the workflow matter both scientifically and for
reviewer responses:

1. **Reflectance in fraction scale [0, 1].** Standard in remote sensing.
   Percent-scale reporting is available on demand from the CSV loader and
   the QC metrics.
2. **Median-based panel DN estimator.** Robust to panel-edge mixing and
   occasional hot pixels, unlike the mean.
3. **No intermediate clipping.** Clipping of negative values happens once,
   explicitly, at the QC stage under a configurable policy
   (`clip` / `flag` / `none`). Earlier stages preserve full information so
   that QC diagnostics remain meaningful.

## Reproducibility

Each pipeline run emits a JSON processing log containing:

- Software and Python/platform versions
- SHA-256 hashes of raw cube, dark frame, and panel reference files
- Full configuration used (every parameter and threshold)
- Dark-signal diagnostics (mean, std across bands)
- ELC gain vector and panel pixel count
- QC diagnostics (saturation rate, hold-out R²/RMSE/MAE, outlier counts)
- UTC timestamps

These logs are intended to make monitoring campaigns verifiable after the
fact.

## Citing this software

If you use this toolbox in your research, please cite both the software
and the associated paper. See `CITATION.cff` for machine-readable metadata.

```bibtex
@software{flores_hsi_preproc_toolbox_2026,
  author  = {Flores, Hernan and Rudolph, Tobias and Benndorf, Jörg},
  title   = {hsi-preproc-toolbox: Reproducible radiometric preprocessing
             for UAV VNIR hyperspectral imagery in post-mining environments},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.XXXXXXX},
  url     = {https://github.com/herrflores/hsi-preproc-toolbox}
}
```

## License

MIT — see `LICENSE`.

## Acknowledgements

Developed as part of ongoing doctoral research on standardized UAV
hyperspectral monitoring for post-mining environments. Reference datasets
used to validate the workflow were acquired at Ibbenbüren, Bochum
(Harpener Teiche area), and Itzenplitz.
"# hsi-preproc-toolbox" 
