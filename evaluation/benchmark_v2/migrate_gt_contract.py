"""
Migrate benchmark v2 GT CSV packs to the period-aware structured contract.

This expands legacy one-value-per-row `rows.csv` packs using the annotated
`cells.csv` value columns, then regenerates canonical page and report GT files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .csv_codec import migrate_rows_to_structured_contract
from .dataset import BenchmarkDatasetV2, IncludeScope
from .report_assembler import build_gt_structured_report_files


def migrate_gt_contract(
    *,
    dataset_root: str | Path,
    include_scope: IncludeScope = "all",
    split: str = "all",
    force: bool = False,
) -> Dict[str, Any]:
    ds = BenchmarkDatasetV2(dataset_root, include_scope=include_scope)
    if split == "dev":
        samples = ds.get_split_samples("dev")
    elif split == "test":
        samples = ds.get_split_samples("test")
    else:
        samples = ds.get_split_samples("dev") + ds.get_split_samples("test")

    migrated: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for sample in samples:
        try:
            result = migrate_rows_to_structured_contract(
                sample.sample_id,
                ds.dataset_root,
                force=force,
            )
            if result.get("changed"):
                migrated.append(result)
            else:
                skipped.append(result)
        except Exception as exc:
            failed.append(
                {
                    "sample_id": sample.sample_id,
                    "report_id": sample.report_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    report_counts = build_gt_structured_report_files(
        ds.dataset_root,
        split=split,  # type: ignore[arg-type]
        include_scope=include_scope,
    )

    return {
        "migrated_count": len(migrated),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "migrated_samples": migrated,
        "skipped_samples": skipped,
        "failed_samples": failed,
        "report_counts": report_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate benchmark v2 GT to the period-aware structured contract")
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--split", default="all", choices=["dev", "test", "all"])
    parser.add_argument("--include-scope", default="all", choices=["all", "included", "not_included"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", default=None, type=str)
    args = parser.parse_args()

    summary = migrate_gt_contract(
        dataset_root=args.dataset_root,
        include_scope=args.include_scope,  # type: ignore[arg-type]
        split=args.split,
        force=bool(args.force),
    )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
