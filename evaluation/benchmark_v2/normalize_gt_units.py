"""
Normalize benchmark v2 GT structured values to canonical VND.

Source of truth:
- gt_csv/<sample_id>/rows.csv

This tool detects the report unit from gt_markdown/<sample_id>.md, rescales
rows.csv values when needed, regenerates canonical gt_structured artifacts,
and rebuilds report-level GT structured files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .csv_codec import _parse_numeric_value, csv_to_canonical, load_csv_pack, save_csv_pack, update_meta
from .dataset import BenchmarkDatasetV2, IncludeScope
from .report_assembler import build_gt_structured_report_files
from .run import _detect_unit_scale_from_markdown


def _format_numeric(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.12f}".rstrip("0").rstrip(".")


def _normalize_rows_values(rows: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    out = rows.copy()
    values: List[str] = []
    for raw in out.get("value", []):
        parsed = _parse_numeric_value(raw)
        values.append(_format_numeric(None if parsed is None else parsed * multiplier))
    out["value"] = values
    return out


def normalize_gt_units(
    *,
    dataset_root: str | Path,
    include_scope: IncludeScope = "all",
    split: str = "all",
    force: bool = False,
    report_scale_overrides: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    ds = BenchmarkDatasetV2(dataset_root, include_scope=include_scope)
    if split == "dev":
        samples = ds.get_split_samples("dev")
    elif split == "test":
        samples = ds.get_split_samples("test")
    else:
        samples = ds.get_split_samples("dev") + ds.get_split_samples("test")

    changed: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    overrides = dict(report_scale_overrides or {})

    for sample in samples:
        gt_md_path = ds.dataset_root / sample.gt_markdown_path
        csv_root = ds.dataset_root / "gt_csv" / sample.sample_id
        if not gt_md_path.exists() or not csv_root.exists():
            continue

        meta = {}
        try:
            meta_path = csv_root / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if not isinstance(meta, dict):
                    meta = {}
        except Exception:
            meta = {}

        label, multiplier = _detect_unit_scale_from_markdown(gt_md_path.read_text(encoding="utf-8"))
        if sample.report_id in overrides:
            multiplier = float(overrides[sample.report_id])
            label = f"override:{sample.report_id}"
        if multiplier == 1.0:
            skipped.append({"sample_id": sample.sample_id, "reason": "already_vnd_or_unknown", "label": label})
            continue
        if not force and meta.get("value_unit_normalized_to") == "VND":
            skipped.append({"sample_id": sample.sample_id, "reason": "already_normalized", "label": label})
            continue

        pack = load_csv_pack(sample.sample_id, ds.dataset_root)
        rows = pack["rows"]
        normalized_rows = _normalize_rows_values(rows, multiplier)
        save_csv_pack(
            sample.sample_id,
            ds.dataset_root,
            cells=pack["cells"],
            rows=normalized_rows,
        )
        update_meta(
            sample.sample_id,
            ds.dataset_root,
            {
                "report_unit_detected": label,
                "report_unit_multiplier": multiplier,
                "value_unit_normalized_to": "VND",
            },
        )
        csv_to_canonical(sample.sample_id, ds.dataset_root, validate=True)
        changed.append(
            {
                "sample_id": sample.sample_id,
                "report_id": sample.report_id,
                "multiplier": multiplier,
                "label": label,
            }
        )

    report_counts = build_gt_structured_report_files(
        ds.dataset_root,
        split=split,  # type: ignore[arg-type]
        include_scope=include_scope,
    )

    return {
        "changed_samples": changed,
        "skipped_samples": skipped,
        "changed_count": len(changed),
        "skipped_count": len(skipped),
        "report_counts": report_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize benchmark v2 GT rows.csv values to VND")
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--split", default="all", choices=["dev", "test", "all"])
    parser.add_argument("--include-scope", default="all", choices=["all", "included", "not_included"])
    parser.add_argument("--force", action="store_true", help="Re-apply normalization even if meta says already normalized")
    parser.add_argument(
        "--report-scale",
        action="append",
        default=[],
        help="Override report-level multiplier as report_id=multiplier. Repeatable.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Optional JSON summary path",
    )
    args = parser.parse_args()

    overrides: Dict[str, float] = {}
    for raw in args.report_scale:
        key, sep, value = str(raw).partition("=")
        if not sep:
            raise ValueError(f"Invalid --report-scale value: {raw}")
        overrides[key.strip()] = float(value.strip())

    summary = normalize_gt_units(
        dataset_root=args.dataset_root,
        include_scope=args.include_scope,  # type: ignore[arg-type]
        split=args.split,
        force=bool(args.force),
        report_scale_overrides=overrides,
    )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
