"""
Report-level structured assembler for benchmark v2.

Builds report-level structured JSON by merging page-level structured files
grouped by report_id, ordered by page_index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Tuple

from .dataset import BenchmarkDatasetV2, TableSample

STATEMENTS = ("balance_sheet", "income_statement", "cash_flow")


def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _coerce_numeric(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def _row_key(statement: str, item: Dict[str, Any], *, fallback: str) -> str:
    code = str(item.get("item_code") or "").strip()
    if code:
        return f"{statement}|code:{_normalize_text(code)}"
    name = str(item.get("item_name") or "").strip()
    if name:
        return f"{statement}|name:{_normalize_text(name)}"
    return f"{statement}|fallback:{fallback}"


def _normalized_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "item_code": item.get("item_code"),
        "item_name": item.get("item_name"),
        "value": item.get("value"),
        "notes_ref": item.get("notes_ref"),
        "original_name": item.get("original_name"),
    }


def assemble_report_structured_from_pages(
    pages: List[Tuple[TableSample, Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Merge page-level structured objects into one report-level object.

    Guardrails:
    - row key: statement + item_code (fallback statement + normalized item_name)
    - duplicate same key + same value: keep one
    - duplicate same key + conflicting value: keep first, log conflict
    - preserve source trace for each merged row
    """
    merged: Dict[str, Any] = {
        "balance_sheet": {"items": []},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    key_index: Dict[str, Dict[str, int]] = {st: {} for st in STATEMENTS}
    row_sources: Dict[str, Dict[str, List[Dict[str, Any]]]] = {st: {} for st in STATEMENTS}
    conflicts: List[Dict[str, Any]] = []

    for sample, obj in sorted(pages, key=lambda x: (x[0].page_index, x[0].sample_id)):
        for statement in STATEMENTS:
            node = obj.get(statement, {})
            items = node.get("items", []) if isinstance(node, dict) else []
            if not isinstance(items, list):
                continue

            for item_idx, raw_item in enumerate(items):
                if not isinstance(raw_item, dict):
                    continue
                item = _normalized_item(raw_item)
                key = _row_key(statement, item, fallback=f"{sample.sample_id}:{item_idx}")
                source = {
                    "sample_id": sample.sample_id,
                    "page_index": sample.page_index,
                    "item_index": item_idx,
                }
                existing_idx = key_index[statement].get(key)
                if existing_idx is None:
                    key_index[statement][key] = len(merged[statement]["items"])
                    merged[statement]["items"].append(item)
                    row_sources[statement][key] = [source]
                    continue

                existing = merged[statement]["items"][existing_idx]
                row_sources[statement][key].append(source)

                old_num = _coerce_numeric(existing.get("value"))
                new_num = _coerce_numeric(item.get("value"))
                if old_num != new_num:
                    conflicts.append(
                        {
                            "report_id": sample.report_id,
                            "sample_id": sample.sample_id,
                            "statement": statement,
                            "row_key": key,
                            "kept_value": existing.get("value"),
                            "dropped_value": item.get("value"),
                        }
                    )

                # Keep first value, but enrich missing metadata fields.
                for field in ("notes_ref", "original_name", "item_name", "item_code"):
                    if (existing.get(field) is None or str(existing.get(field)).strip() == "") and (
                        item.get(field) is not None and str(item.get(field)).strip() != ""
                    ):
                        existing[field] = item.get(field)

    meta = {
        "row_sources": row_sources,
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
    }
    return merged, meta


def _collect_split_samples(ds: BenchmarkDatasetV2, split: Literal["dev", "test", "all"]) -> List[TableSample]:
    if split == "dev":
        return ds.get_split_samples("dev")
    if split == "test":
        return ds.get_split_samples("test")
    return ds.get_split_samples("dev") + ds.get_split_samples("test")


def _group_by_report(samples: Iterable[TableSample]) -> Dict[str, List[TableSample]]:
    out: Dict[str, List[TableSample]] = {}
    for s in samples:
        out.setdefault(s.report_id, []).append(s)
    return out


def build_gt_structured_report_files(
    dataset_root: str | Path,
    *,
    split: Literal["dev", "test", "all"] = "all",
    output_dir: str = "gt_structured_report",
    meta_dir: str = "gt_structured_report_meta",
) -> Dict[str, int]:
    ds = BenchmarkDatasetV2(dataset_root)
    samples = _collect_split_samples(ds, split)
    by_report = _group_by_report(samples)

    out_root = ds.dataset_root / output_dir
    meta_root = ds.dataset_root / meta_dir
    out_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    counts = {"reports_total": 0, "reports_saved": 0, "reports_failed": 0}
    for report_id, report_samples in sorted(by_report.items()):
        counts["reports_total"] += 1
        try:
            pages: List[Tuple[TableSample, Dict[str, Any]]] = []
            for s in sorted(report_samples, key=lambda x: (x.page_index, x.sample_id)):
                p = ds.dataset_root / s.gt_structured_path
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if not isinstance(obj, dict):
                    raise ValueError(f"Invalid JSON object: {p}")
                pages.append((s, obj))

            merged, meta = assemble_report_structured_from_pages(pages)
            (out_root / f"{report_id}.json").write_text(
                json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (meta_root / f"{report_id}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            counts["reports_saved"] += 1
        except Exception:
            counts["reports_failed"] += 1
    return counts


def build_prediction_structured_report_files(
    dataset_root: str | Path,
    predictions_root: str | Path,
    *,
    split: Literal["dev", "test", "all"] = "all",
    structured_suffix: str = ".structured.json",
    output_dir: str = "report_structured",
    meta_dir: str = "report_structured_meta",
    strict_missing: bool = False,
) -> Dict[str, int]:
    ds = BenchmarkDatasetV2(dataset_root)
    samples = _collect_split_samples(ds, split)
    by_report = _group_by_report(samples)
    pred_root = Path(predictions_root)

    out_root = pred_root / output_dir
    meta_root = pred_root / meta_dir
    out_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    counts = {"reports_total": 0, "reports_saved": 0, "reports_failed": 0}
    for report_id, report_samples in sorted(by_report.items()):
        counts["reports_total"] += 1
        try:
            pages: List[Tuple[TableSample, Dict[str, Any]]] = []
            for s in sorted(report_samples, key=lambda x: (x.page_index, x.sample_id)):
                p = pred_root / f"{s.sample_id}{structured_suffix}"
                if not p.exists():
                    if strict_missing:
                        raise FileNotFoundError(f"Missing structured prediction: {p}")
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if not isinstance(obj, dict):
                    raise ValueError(f"Invalid JSON object: {p}")
                pages.append((s, obj))

            if not pages:
                if strict_missing:
                    raise FileNotFoundError(f"No structured pages found for report {report_id}")
                counts["reports_failed"] += 1
                continue

            merged, meta = assemble_report_structured_from_pages(pages)
            (out_root / f"{report_id}.structured.json").write_text(
                json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (meta_root / f"{report_id}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            counts["reports_saved"] += 1
        except Exception:
            counts["reports_failed"] += 1
    return counts

