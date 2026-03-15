"""
Generate detailed GT-vs-prediction diffs for benchmark v2.

Outputs a JSON artifact suitable for manual inspection or the companion
Streamlit viewer in debug_app.py.
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from evaluation.ocr_benchmark.metrics import extract_numeric_tokens

from .dataset import BenchmarkDatasetV2, IncludeScope, TableSample
from .metrics_raw import RawScope, calculate_raw_metrics
from .metrics_structured import calculate_structured_metrics
from .report_assembler import assemble_report_structured_from_pages
from .structured_contract import coerce_numeric, extract_structured_rows, normalize_item

SplitChoice = Literal["dev", "test", "all"]
STATEMENTS = ("balance_sheet", "income_statement", "cash_flow")


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

def _structured_rows(obj: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    extracted = extract_structured_rows(obj, STATEMENTS)
    for key, rows in extracted.items():
        for row in rows:
            out[key].append({"statement": row["statement"], **normalize_item(row), "value": row.get("value")})
    return dict(out)


def _value_sort_key(row: Dict[str, Any]) -> Tuple[int, float, str]:
    num = coerce_numeric(row.get("value"))
    if num is not None:
        return (0, num, "")
    return (1, 0.0, str(row.get("value") or ""))


def _compare_structured_objects(prediction: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
    pred_rows = _structured_rows(prediction)
    gt_rows = _structured_rows(reference)

    missing_rows: List[Dict[str, Any]] = []
    extra_rows: List[Dict[str, Any]] = []
    value_mismatches: List[Dict[str, Any]] = []

    for key in sorted(gt_rows.keys() | pred_rows.keys()):
        gt_list = sorted(gt_rows.get(key, []), key=_value_sort_key)
        pred_list = sorted(pred_rows.get(key, []), key=_value_sort_key)
        matched = min(len(gt_list), len(pred_list))

        for idx in range(matched):
            gt_row = gt_list[idx]
            pred_row = pred_list[idx]
            gt_val = coerce_numeric(gt_row.get("value"))
            pred_val = coerce_numeric(pred_row.get("value"))
            if gt_val != pred_val:
                value_mismatches.append(
                    {
                        "row_key": key,
                        "gt_value": gt_row.get("value"),
                        "pred_value": pred_row.get("value"),
                        "gt_item_name": gt_row.get("item_name"),
                        "pred_item_name": pred_row.get("item_name"),
                    }
                )

        for row in gt_list[matched:]:
            missing_rows.append({"row_key": key, **row})
        for row in pred_list[matched:]:
            extra_rows.append({"row_key": key, **row})

    return {
        "metrics": calculate_structured_metrics(prediction, reference).to_dict(),
        "missing_rows": missing_rows,
        "extra_rows": extra_rows,
        "value_mismatches": value_mismatches,
    }


def _collect_split_samples(ds: BenchmarkDatasetV2, split: SplitChoice) -> List[TableSample]:
    if split == "dev":
        return ds.get_split_samples("dev")
    if split == "test":
        return ds.get_split_samples("test")
    return ds.get_split_samples("dev") + ds.get_split_samples("test")


def _group_by_report(samples: Iterable[TableSample]) -> Dict[str, List[TableSample]]:
    out: Dict[str, List[TableSample]] = {}
    for sample in samples:
        out.setdefault(sample.report_id, []).append(sample)
    return out


def build_debug_diffs(
    *,
    dataset_root: str | Path,
    predictions_root: str | Path,
    split: SplitChoice = "dev",
    include_scope: IncludeScope = "all",
    raw_suffix: str = ".raw.md",
    structured_suffix: str = ".structured.json",
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

    report_diffs: List[Dict[str, Any]] = []
    for report_id, report_samples in sorted(_group_by_report(samples).items()):
        gt_pages: List[Tuple[TableSample, Dict[str, Any]]] = []
        pred_pages: List[Tuple[TableSample, Dict[str, Any]]] = []
        errors: List[str] = []

        for sample in sorted(report_samples, key=lambda x: (x.page_index, x.sample_id)):
            gt_struct_path = ds.dataset_root / sample.gt_structured_path
            pred_struct_path = pred_root / f"{sample.sample_id}{structured_suffix}"
            gt_pages.append((sample, _read_json(gt_struct_path)))
            if pred_struct_path.exists():
                pred_pages.append((sample, _read_json(pred_struct_path)))
            else:
                errors.append(f"missing_structured_prediction: {pred_struct_path}")

        gt_assembled, gt_meta = assemble_report_structured_from_pages(gt_pages)
        pred_assembled = None
        pred_meta: Dict[str, Any] = {"conflicts": [], "conflict_count": 0}
        comparison = None
        if len(pred_pages) == len(report_samples) and pred_pages:
            pred_assembled, pred_meta = assemble_report_structured_from_pages(pred_pages)
            comparison = _compare_structured_objects(pred_assembled, gt_assembled)

        report_diffs.append(
            {
                "report_id": report_id,
                "company": report_samples[0].company,
                "split": report_samples[0].split,
                "page_count": len(report_samples),
                "sample_ids": [sample.sample_id for sample in report_samples],
                "structured_available": comparison is not None,
                "errors": errors,
                "gt_conflicts": gt_meta.get("conflicts", []),
                "pred_conflicts": pred_meta.get("conflicts", []),
                "comparison": comparison,
                "gt_structured": gt_assembled,
                "pred_structured": pred_assembled,
            }
        )

    return {
        "benchmark_version": "v2",
        "split": split,
        "include_scope": include_scope,
        "dataset_root": str(Path(dataset_root).resolve()),
        "predictions_root": str(Path(predictions_root).resolve()),
        "sample_diffs": sample_diffs,
        "report_diffs": report_diffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate detailed benchmark v2 diffs")
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--predictions-root", required=True, type=str)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--include-scope", default="all", choices=["all", "included", "not_included"])
    parser.add_argument("--raw-suffix", default=".raw.md", type=str)
    parser.add_argument("--structured-suffix", default=".structured.json", type=str)
    parser.add_argument("--output", default="results/benchmark_v2_debug_diffs.json", type=str)
    args = parser.parse_args()

    payload = build_debug_diffs(
        dataset_root=args.dataset_root,
        predictions_root=args.predictions_root,
        split=args.split,  # type: ignore[arg-type]
        include_scope=args.include_scope,  # type: ignore[arg-type]
        raw_suffix=args.raw_suffix,
        structured_suffix=args.structured_suffix,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path.resolve())


if __name__ == "__main__":
    main()
