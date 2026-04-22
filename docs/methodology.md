# Methodology

This document summarizes the preprocessing workflow implemented in
`hsi-preproc-toolbox`. It corresponds to Section 2.3 of the associated paper:

> Flores, H. (2026). *Standardizing UAV Hyperspectral Data for Monitoring
> Post-Mining Environments: A Reproducible Preprocessing Framework.*
> Green and Smart Mining Engineering.

For the scientific rationale and validation results, refer to the paper
directly. This document is intended as an implementation reference.

## Workflow stages

### 1. Dark current subtraction

Per-band subtraction of an additive sensor baseline estimated from same-day
dark reference frames. Implements:

    DN_dc(x, y, λ) = DN_raw(x, y, λ) − DN_dark(λ)

Implementation: `hsi_preproc_toolbox.dark_correction.subtract_dark`

### 2. Empirical line calibration (ELC)

Single-panel reduced ELC using a NIST-traceable 50% Spectralon reference.
The additive term b(λ) is fixed to zero after dark correction; calibration
is a wavelength-dependent gain anchored to the known panel reflectance:

    a(λ) = R_ref(λ) / DN_panel(λ)
    R(x, y, λ) = a(λ) · DN_dc(x, y, λ)

Implementation: `hsi_preproc_toolbox.elc.empirical_line_calibration`

### 3. Optional Savitzky–Golay smoothing

Applied along the spectral dimension with configurable window length
(default 7 bands ≈ 35 nm) and polynomial order (default 2).

Implementation: `hsi_preproc_toolbox.smoothing.savgol_smooth`

### 4. Integrated quality control

- Panel saturation screening (pre-calibration)
- Negative reflectance filtering (configurable: clip / flag / none)
- Spectral outlier detection via band-wise MAD on first differences
- Panel-based hold-out validation (R² and RMSE over a usable wavelength range)

Implementation: `hsi_preproc_toolbox.qc.run_quality_control`

## Reproducibility

Each pipeline run emits a JSON processing log capturing:

- Software version and platform
- SHA-256 hashes of all input files
- Full configuration (all parameters and thresholds)
- Dark-signal diagnostics
- ELC gain vector
- QC diagnostics

These logs are intended to support per-campaign traceability and to make
multi-temporal analyses verifiable.
