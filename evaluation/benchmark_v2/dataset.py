"""
Dataset loader for benchmark v2.

Expected dataset root layout:
  <dataset_root>/
    manifest.json
    ... files referenced by samples ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

SplitName = Literal["dev", "test"]


@dataclass(frozen=True)
class TableSample:
    sample_id: str
    split: SplitName
    company: str
    report_id: str
    page_index: int
    page_image_path: str
    gt_markdown_path: str
    gt_structured_path: str
    gt_table_cells_path: Optional[str] = None
    source_pdf_path: Optional[str] = None
    annotator_id: Optional[str] = None
    annotation_passes: int = 1
    audited_by: Optional[str] = None
    notes: Optional[str] = None

    def resolve_path(self, root: Path, rel_path: Optional[str]) -> Optional[Path]:
        if not rel_path:
            return None
        return (root / rel_path).resolve()


class BenchmarkDatasetV2:
    """Schema-light, strict-enough loader for benchmark v2 manifests."""

    def __init__(self, dataset_root: str | Path, manifest_path: str = "manifest.json"):
        self.dataset_root = Path(dataset_root)
        self.manifest_path = self.dataset_root / manifest_path
        self._manifest: Optional[Dict[str, Any]] = None
        self._samples: Optional[List[TableSample]] = None

    @property
    def manifest(self) -> Dict[str, Any]:
        if self._manifest is None:
            self._manifest = self._load_manifest()
        return self._manifest

    @property
    def samples(self) -> List[TableSample]:
        if self._samples is None:
            self._samples = self._parse_samples(self.manifest)
        return self._samples

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Manifest root must be a JSON object")
        if "samples" not in data or not isinstance(data["samples"], list):
            raise ValueError("Manifest must contain a 'samples' array")
        return data

    def _parse_samples(self, manifest: Dict[str, Any]) -> List[TableSample]:
        parsed: List[TableSample] = []
        seen_ids: set[str] = set()
        for row in manifest.get("samples", []):
            if not isinstance(row, dict):
                raise ValueError("Each sample entry must be an object")
            required = [
                "sample_id",
                "split",
                "company",
                "report_id",
                "page_index",
                "page_image_path",
                "gt_markdown_path",
                "gt_structured_path",
            ]
            missing = [k for k in required if k not in row]
            if missing:
                raise ValueError(f"Sample is missing required fields: {missing}")

            split_raw = str(row["split"]).strip().lower()
            if split_raw not in {"dev", "test"}:
                raise ValueError(f"Invalid split '{split_raw}' for sample {row.get('sample_id')}")

            sample = TableSample(
                sample_id=str(row["sample_id"]),
                split=split_raw,  # type: ignore[arg-type]
                company=str(row["company"]).upper(),
                report_id=str(row["report_id"]),
                page_index=int(row["page_index"]),
                page_image_path=str(row["page_image_path"]),
                gt_markdown_path=str(row["gt_markdown_path"]),
                gt_structured_path=str(row["gt_structured_path"]),
                gt_table_cells_path=(
                    str(row["gt_table_cells_path"])
                    if row.get("gt_table_cells_path") is not None
                    else None
                ),
                source_pdf_path=(
                    str(row["source_pdf_path"]) if row.get("source_pdf_path") is not None else None
                ),
                annotator_id=(
                    str(row["annotator_id"]) if row.get("annotator_id") is not None else None
                ),
                annotation_passes=int(row.get("annotation_passes", 1)),
                audited_by=str(row["audited_by"]) if row.get("audited_by") else None,
                notes=str(row["notes"]) if row.get("notes") else None,
            )

            if sample.sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
            seen_ids.add(sample.sample_id)
            parsed.append(sample)
        return parsed

    def iter_split(self, split: SplitName) -> Iterable[TableSample]:
        split_norm = split.strip().lower()
        if split_norm not in {"dev", "test"}:
            raise ValueError(f"Invalid split: {split}")
        for s in self.samples:
            if s.split == split_norm:
                yield s

    def get_split_samples(self, split: SplitName) -> List[TableSample]:
        return list(self.iter_split(split))

    def validate_split_presence(self) -> None:
        dev_count = len(self.get_split_samples("dev"))
        test_count = len(self.get_split_samples("test"))
        if dev_count == 0 or test_count == 0:
            raise ValueError(
                f"Both dev and test must be present. Found dev={dev_count}, test={test_count}"
            )

    def validate_company_disjoint(self) -> None:
        dev_companies = {s.company for s in self.get_split_samples("dev")}
        test_companies = {s.company for s in self.get_split_samples("test")}
        overlap = dev_companies & test_companies
        if overlap:
            overlap_str = ", ".join(sorted(overlap))
            raise ValueError(
                f"Company-heldout split violated. Overlapping companies between dev/test: {overlap_str}"
            )

    def validate_referenced_files(self) -> None:
        for s in self.samples:
            refs = [
                s.page_image_path,
                s.gt_markdown_path,
                s.gt_structured_path,
                s.gt_table_cells_path,
                s.source_pdf_path,
            ]
            for rel in refs:
                if rel is None:
                    continue
                p = self.dataset_root / rel
                if not p.exists():
                    raise FileNotFoundError(f"Missing referenced file for {s.sample_id}: {p}")

    def validate(self, *, check_files: bool = False) -> None:
        self.validate_split_presence()
        self.validate_company_disjoint()
        if check_files:
            self.validate_referenced_files()

    def get_stats(self) -> Dict[str, Any]:
        self.validate_split_presence()
        dev = self.get_split_samples("dev")
        test = self.get_split_samples("test")
        return {
            "total_samples": len(self.samples),
            "dev_samples": len(dev),
            "test_samples": len(test),
            "dev_companies": sorted({s.company for s in dev}),
            "test_companies": sorted({s.company for s in test}),
            "single_annotator_fraction": (
                sum(1 for s in self.samples if (s.annotator_id and not s.audited_by)) / len(self.samples)
                if self.samples
                else 0.0
            ),
            "avg_annotation_passes": (
                sum(max(1, int(s.annotation_passes)) for s in self.samples) / len(self.samples)
                if self.samples
                else 0.0
            ),
        }

