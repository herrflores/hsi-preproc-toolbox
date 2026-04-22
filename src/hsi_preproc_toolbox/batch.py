"""
Batch processing utilities for UAV hyperspectral campaigns.

Mirrors the notebook workflow where all .hdr cubes in an input folder are
processed with the same pipeline configuration. Dark frame and calibration
panel cubes (identified by filename) are excluded from the batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .pipeline import Pipeline, PipelineConfig, PipelineResult


DEFAULT_EXCLUDED_STEMS = ("dark_frame", "calibration_panel")


@dataclass
class BatchReport:
    """Summary of a batch run."""

    processed: list[Path]
    failed: list[tuple[Path, str]]
    output_folder: Path

    def summary(self) -> str:
        lines = [
            f"Batch output folder: {self.output_folder}",
            f"Processed: {len(self.processed)} cubes",
        ]
        if self.failed:
            lines.append(f"Failed: {len(self.failed)} cubes")
            for path, err in self.failed:
                lines.append(f"  - {path.name}: {err}")
        return "\n".join(lines)


def batch_process_folder(
    input_folder: str | Path,
    output_folder: str | Path,
    config: PipelineConfig,
    *,
    excluded_stems: tuple[str, ...] = DEFAULT_EXCLUDED_STEMS,
    output_suffix: str = "_reflectance",
) -> BatchReport:
    """
    Process every .hdr cube in ``input_folder`` with the same configuration,
    writing reflectance-calibrated outputs to ``output_folder``.

    Files whose stem (filename without extension) matches any entry in
    ``excluded_stems`` are skipped.

    Parameters
    ----------
    input_folder : str or Path
        Folder containing .hdr + binary cubes.
    output_folder : str or Path
        Destination for reflectance cubes and per-cube JSON logs.
    config : PipelineConfig
        Pipeline configuration, applied identically to each cube.
    excluded_stems : tuple[str, ...]
        Filename stems (without .hdr) to skip.
    output_suffix : str
        Suffix added to output filenames before the .hdr extension.

    Returns
    -------
    BatchReport
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(config)
    processed: list[Path] = []
    failed: list[tuple[Path, str]] = []

    hdr_files = sorted(input_folder.glob("*.hdr"))
    for hdr in hdr_files:
        if hdr.stem in excluded_stems:
            continue
        out_hdr = output_folder / f"{hdr.stem}{output_suffix}.hdr"
        try:
            result: PipelineResult = pipeline.run(hdr)
            result.save(out_hdr)
            processed.append(out_hdr)
        except Exception as exc:  # pragma: no cover - reported to user
            failed.append((hdr, str(exc)))

    return BatchReport(processed=processed, failed=failed, output_folder=output_folder)
