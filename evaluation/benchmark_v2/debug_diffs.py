"""
Generate detailed raw OCR GT-vs-prediction diffs for benchmark v2.
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal

from evaluation.ocr_benchmark.metrics import extract_numeric_tokens

from .dataset import BenchmarkDatasetV2, IncludeScope, TableSample
from .metrics_raw import RawScope, calculate_raw_metrics

SplitChoice = Literal["dev", "test", "all"]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _parse_markdown_pipe_cells(markdown_text: str) -> List[str]:
    cells: List[str] = []
    for line in (markdown_text or "").splitlines():
        s = line.strip()
        if s.count("|") < 2:
            continue
        parts = [p.strip() for p in s.split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        if not parts:
            continue
        if all((set(p.replace(":", "").strip()) <= {"-"} and "-" in p) or p == "" for p in parts):
            continue
        cells.extend(parts)
    return cells


def _extract_table_only_text(markdown_text: str) -> str:
    rows: List[str] = []
    for line in (markdown_text or "").splitlines():
        s = line.strip()
        if s.count("|") < 2:
            continue
        parts = [p.strip() for p in s.split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        if not parts:
            continue
        if all((set(p.replace(":", "").strip()) <= {"-"} and "-" in p) or p == "" for p in parts):
            continue
        rows.append("| " + " | ".join(parts) + " |")
    return "\n".join(rows)


def _counter_delta(left: Iterable[str], right: Iterable[str]) -> List[Dict[str, Any]]:
    left_c = Counter(left)
    right_c = Counter(right)
    out: List[Dict[str, Any]] = []
    for key in sorted(left_c.keys() | right_c.keys()):
        delta = left_c[key] - right_c[key]
        if delta > 0:
            out.append({"value": key, "count": int(delta)})
    return out


def _collect_split_samples(ds: BenchmarkDatasetV2, split: SplitChoice) -> List[TableSample]:
    if split == "dev":
        return ds.get_split_samples("dev")
    if split == "test":
        return ds.get_split_samples("test")
    return ds.get_split_samples("dev") + ds.get_split_samples("test")


def build_debug_diffs(
    *,
    dataset_root: str | Path,
    predictions_root: str | Path,
    split: SplitChoice = "dev",
    include_scope: IncludeScope = "all",
    raw_suffix: str = ".raw.md",
    raw_scope: RawScope = "table_only",
) -> Dict[str, Any]:
    ds = BenchmarkDatasetV2(dataset_root, include_scope=include_scope)
    samples = _collect_split_samples(ds, split)
    pred_root = Path(predictions_root)

    sample_diffs: List[Dict[str, Any]] = []
    for sample in samples:
        gt_md_path = ds.dataset_root / sample.gt_markdown_path
        pred_md_path = pred_root / f"{sample.sample_id}{raw_suffix}"
        pred_ocr_debug_path = pred_root / f"{sample.sample_id}.ocr_debug.json"
        gt_md = _read_text(gt_md_path)
        pred_md = _read_text(pred_md_path) if pred_md_path.exists() else ""
        ocr_debug = _read_json(pred_ocr_debug_path) if pred_ocr_debug_path.exists() else None

        gt_table_text = _extract_table_only_text(gt_md)
        pred_table_text = _extract_table_only_text(pred_md)
        gt_cells = _parse_markdown_pipe_cells(gt_md)
        pred_cells = _parse_markdown_pipe_cells(pred_md)
        gt_numbers = extract_numeric_tokens(gt_table_text)
        pred_numbers = extract_numeric_tokens(pred_table_text)
        unified = list(
            difflib.unified_diff(
                gt_table_text.splitlines(),
                pred_table_text.splitlines(),
                fromfile=f"{sample.sample_id}.gt",
                tofile=f"{sample.sample_id}.pred",
                n=2,
                lineterm="",
            )
        )
        sample_diffs.append(
            {
                "sample_id": sample.sample_id,
                "company": sample.company,
                "report_id": sample.report_id,
                "page_index": sample.page_index,
                "page_image_path": str((ds.dataset_root / sample.page_image_path).resolve()),
                "raw_available": pred_md_path.exists(),
                "ocr_debug_path": str(pred_ocr_debug_path.resolve()) if pred_ocr_debug_path.exists() else None,
                "ocr_debug": ocr_debug,
                "telemetry": dict((ocr_debug or {}).get("telemetry") or {}),
                "raw_metrics": (
                    calculate_raw_metrics(pred_md, gt_md, scope=raw_scope).to_dict()
                    if pred_md_path.exists()
                    else None
                ),
                "missing_numbers": _counter_delta(gt_numbers, pred_numbers),
                "extra_numbers": _counter_delta(pred_numbers, gt_numbers),
                "missing_cells": _counter_delta(gt_cells, pred_cells)[:100],
                "extra_cells": _counter_delta(pred_cells, gt_cells)[:100],
                "table_diff_excerpt": unified[:120],
                "gt_table_text": gt_table_text,
                "pred_table_text": pred_table_text,
                "gt_raw_markdown": gt_md,
                "pred_raw_markdown": pred_md,
            }
        )

    return {
        "benchmark_version": "v2_raw_only",
        "split": split,
        "include_scope": include_scope,
        "dataset_root": str(Path(dataset_root).resolve()),
        "predictions_root": str(Path(predictions_root).resolve()),
        "sample_diffs": sample_diffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate detailed raw OCR benchmark v2 diffs")
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--predictions-root", required=True, type=str)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--include-scope", default="all", choices=["all", "included", "not_included"])
    parser.add_argument("--raw-suffix", default=".raw.md", type=str)
    parser.add_argument("--output", default="results/benchmark_v2_debug_diffs.json", type=str)
    args = parser.parse_args()

    payload = build_debug_diffs(
        dataset_root=args.dataset_root,
        predictions_root=args.predictions_root,
        split=args.split,  # type: ignore[arg-type]
        include_scope=args.include_scope,  # type: ignore[arg-type]
        raw_suffix=args.raw_suffix,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path.resolve())


if __name__ == "__main__":
    main()
