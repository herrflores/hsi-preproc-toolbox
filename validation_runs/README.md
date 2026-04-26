# Validation Runs — Cross-Campaign Panel Validation Outputs

This directory accompanies the manuscript:

> **Flores, H., Rudolph, T., Benndorf, J. (2026)**. *Standardizing UAV Hyperspectral Data for Monitoring Post-Mining Environments: A Reproducible Preprocessing Framework*. **Green and Smart Mining Engineering**, in revision.

It contains the panel-based hold-out validation outputs for the three independent calibration events reported in **Table 3** and **Figure 8** of the manuscript, together with auditable processing logs and a reproducibility notebook.

**Toolbox concept DOI**: [10.5281/zenodo.19699318](https://doi.org/10.5281/zenodo.19699318)
**License**: MIT

---

## Directory structure

```
validation_runs/
├── README.md                                  # this file
├── generate_processing_logs.py                # script to (re)generate the JSON logs
├── reproduce_table3_and_figure8.ipynb         # reproducibility notebook
│
├── ibbenburen_2023_07/                        # Ibbenbüren campaign — tailing sludge
│   ├── panel_validation_metrics_per_band.csv
│   ├── processing_log.json
│   └── qc_summary.json
│
├── bochum_2025_10/                            # Bochum campaign — mine water inflow + pond
│   ├── panel_validation_metrics_per_band.csv
│   ├── processing_log.json
│   └── qc_summary.json
│
└── itzenplitz_2025_11/                        # Itzenplitz campaign — headframe corrosion
    ├── panel_validation_metrics_per_band.csv
    ├── processing_log.json
    └── qc_summary.json
```

---

## File descriptions

### `panel_validation_metrics_per_band.csv`

Per-band panel validation outputs, one row per spectral band (100 bands total, 500–1000 nm).

| Column | Units | Description |
|---|---|---|
| `wavelength_nm` | nm | Centre wavelength of the spectral band |
| `ref_reflectance` | % | NIST-traceable reference panel reflectance |
| `val_mean_reflectance` | % | Mean calibrated reflectance over hold-out validation pixels |
| `val_std_reflectance` | % | Standard deviation across validation pixels |
| `RMSE` | % | Root mean square error (validation − reference) |
| `MAE` | % | Mean absolute error |
| `Bias` | % | Mean signed difference (validation − reference) |

Validation parameters (identical for all three campaigns):
- 70/30 calibration / validation split
- Random seed = 42
- Median panel DN estimator
- Conservative interior polygon ROI (outermost 10–15 % of panel pixels excluded)

### `processing_log.json`

Auditable processing log following the structured schema described in **Section 2.4.5** of the manuscript. Each log records:

- **Campaign metadata**: site, date, weather, altitude, acquisition mode, integration time, gain
- **Software version**: toolbox version, concept DOI, dependencies
- **Preprocessing parameters**: dark subtraction, ELC, smoothing, QC settings
- **Input file SHA-256 checksums**: cryptographic verification of data integrity
- **Data provenance notes**: explicit statement on which data products are redistributed and which are available on request

### `qc_summary.json`

Compact summary of quality control diagnostics for each run:

- Validation method and parameters
- Pixel counts (calibration / validation)
- Aggregated metrics over the usable spectral range (520–930 nm): R², mean RMSE, mean MAE, mean bias
- QC pass/fail flag

---

## Reproducing Table 3 and Figure 8 of the manuscript

Open and run the notebook:

```bash
jupyter notebook reproduce_table3_and_figure8.ipynb
```

The notebook is self-contained and depends only on `pandas`, `numpy`, and `matplotlib`. It loads the per-band CSVs, computes per-campaign and cross-campaign metrics, and overlays the per-band RMSE and bias curves.

**Verified reproduction**:

| Metric | Manuscript (Table 3) | Reproduction |
|---|---|---|
| R² (cross-campaign) | 0.89 ± 0.02 | 0.89 ± 0.02 |
| Mean RMSE (cross-campaign) | 0.88 ± 0.18 % | 0.88 ± 0.18 % |

The notebook also re-verifies the SHA-256 checksums of the published CSVs against those recorded in `processing_log.json`, providing cryptographic confirmation that the data has not been altered.

---

## Regenerating the JSON logs

If the underlying CSVs are updated (e.g., after a new toolbox release that changes parameters or adds metadata fields), the JSON logs can be regenerated with:

```bash
python generate_processing_logs.py
```

This script reads each campaign's CSV, recomputes the SHA-256 checksum, and writes fresh `processing_log.json` and `qc_summary.json` files in place. It does not modify the CSV data.

---

## Notes on data provenance

- **Per-band validation CSVs** are released openly here. They are derived outputs (not raw images) and contain the information needed to reproduce all per-campaign and cross-campaign metrics reported in the manuscript.
- **Raw UAV hyperspectral cubes** (.img / .hdr, ~57 MB each) are not redistributed in this repository due to size and to site-access constraints with the operators of the post-mining sites surveyed.
- **Processed reflectance products** are available from the corresponding author upon reasonable request.

For questions about the data or the workflow, please contact: **hernan.flores@thga.de**
