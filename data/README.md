# Data

This folder is intentionally kept empty in the repository. UAV hyperspectral
datacubes are too large to version in Git and contain site-specific metadata
that is not released as part of this toolbox.

## What goes here

To run the end-to-end example, place the following files in this directory:

| File | Description | Format |
|---|---|---|
| `flight_<date>.hdr` + binary | Raw UAV hyperspectral cube from HAIP BlackBird V2 | ENVI |
| `dark_<date>.hdr` + binary | Same-day dark reference acquisition | ENVI |
| `spectralon_50pct_nist.csv` | NIST-traceable panel reflectance spectrum | CSV with columns `wavelength_nm`, `reflectance` (0–1) |
| `panel_roi_<date>.npy` | Boolean mask of the panel ROI, matching the cube spatial shape | NumPy array |

## Obtaining the reference datasets

The reference campaigns used in the associated paper were acquired at:

- Ibbenbüren (July 2023) — tailing sludge
- Bochum / Harpener Teiche (October 2025) — mine water inflow and pond
- Itzenplitz (November 2025) — headframe corrosion

Sample / demonstration datasets for reproducing the pipeline may be
requested from the corresponding author (see main repository README).

## Format conventions

- Reflectance reference CSVs must use values in [0, 1], not percent.
- Wavelengths are expected in nanometers.
- Panel ROI masks must have the same spatial resolution as the cube
  (same `(rows, cols)`), with `True` indicating panel pixels.
