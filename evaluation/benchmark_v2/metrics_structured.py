"""
Structured output metrics for benchmark v2.

Evaluates UI-visible table rows and numeric values after parsing.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

STATEMENTS = ("balance_sheet", "income_statement", "cash_flow")


@dataclass
class StructuredMetricResult:
    schema_valid: float
    row_precision: float
    row_recall: float
    row_f1: float
    value_exact_accuracy: float
    value_tolerant_accuracy: float
    gt_row_count: int
    pred_row_count: int
    matched_row_count: int

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


def _normalize_text(s: str) -> str:
    s2 = (s or "").strip().lower()
    s2 = re.sub(r"\s+", " ", s2)
    return s2


def _coerce_value(v: Any) -> float | None:
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
    except ValueError:
        return None


def _is_schema_valid(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    for st in STATEMENTS:
        node = obj.get(st)
        if not isinstance(node, dict):
            return False
        items = node.get("items")
        if not isinstance(items, list):
            return False
    return True


def _make_row_key(statement: str, item: Dict[str, Any]) -> str:
    code = str(item.get("item_code") or "").strip()
    if code:
        return f"{statement}|code:{_normalize_text(code)}"
    name = str(item.get("item_name") or "").strip()
    notes_ref = str(item.get("notes_ref") or "").strip()
    if notes_ref:
        return f"{statement}|name:{_normalize_text(name)}|note:{_normalize_text(notes_ref)}"
    return f"{statement}|name:{_normalize_text(name)}"


def _extract_rows(obj: Dict[str, Any]) -> Dict[str, List[float | None]]:
    out: Dict[str, List[float | None]] = {}
    for st in STATEMENTS:
        st_obj = obj.get(st) or {}
        items = st_obj.get("items") or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            key = _make_row_key(st, item)
            out.setdefault(key, []).append(_coerce_value(item.get("value")))
    return out


def _prf(matched: int, pred_total: int, gt_total: int) -> Tuple[float, float, float]:
    precision = matched / pred_total if pred_total else (1.0 if gt_total == 0 else 0.0)
    recall = matched / gt_total if gt_total else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _values_close(a: float | None, b: float | None, abs_tol: float, rel_tol: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _match_count(
    pred_values: List[float | None],
    gt_values: List[float | None],
    *,
    predicate,
) -> int:
    used_gt: set[int] = set()
    matched = 0
    for pred in pred_values:
        best_idx = None
        best_distance = float("inf")
        for idx, gt in enumerate(gt_values):
            if idx in used_gt or not predicate(pred, gt):
                continue
            if pred is None or gt is None:
                distance = 0.0
            else:
                distance = abs(pred - gt)
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
        if best_idx is not None:
            used_gt.add(best_idx)
            matched += 1
    return matched


def _matched_row_count(pred_rows: Dict[str, List[float | None]], gt_rows: Dict[str, List[float | None]]) -> int:
    pred_counts = Counter({k: len(v) for k, v in pred_rows.items()})
    gt_counts = Counter({k: len(v) for k, v in gt_rows.items()})
    return sum(min(pred_counts[k], gt_counts[k]) for k in pred_counts.keys() & gt_counts.keys())


def calculate_structured_metrics(
    prediction: Dict[str, Any],
    reference: Dict[str, Any],
    *,
    abs_tolerance: float = 1.0,
    rel_tolerance: float = 1e-6,
) -> StructuredMetricResult:
    schema_valid = 1.0 if _is_schema_valid(prediction) else 0.0

    pred_rows = _extract_rows(prediction)
    gt_rows = _extract_rows(reference)

    pred_keys = set(pred_rows.keys())
    gt_keys = set(gt_rows.keys())
    matched_row_count = _matched_row_count(pred_rows, gt_rows)

    row_p, row_r, row_f1 = _prf(
        matched_row_count,
        sum(len(v) for v in pred_rows.values()),
        sum(len(v) for v in gt_rows.values()),
    )

    if matched_row_count == 0:
        exact_acc = 0.0 if gt_keys else 1.0
        tolerant_acc = exact_acc
    else:
        exact_ok = 0
        tolerant_ok = 0
        for k in pred_keys & gt_keys:
            pred_values = pred_rows.get(k, [])
            gt_values = gt_rows.get(k, [])
            exact_ok += _match_count(pred_values, gt_values, predicate=lambda a, b: a == b)
            tolerant_ok += _match_count(
                pred_values,
                gt_values,
                predicate=lambda a, b: _values_close(
                    a,
                    b,
                    abs_tol=abs_tolerance,
                    rel_tol=rel_tolerance,
                ),
            )
        denom = matched_row_count
        exact_acc = exact_ok / denom if denom else 1.0
        tolerant_acc = tolerant_ok / denom if denom else 1.0

    return StructuredMetricResult(
        schema_valid=float(schema_valid),
        row_precision=float(row_p),
        row_recall=float(row_r),
        row_f1=float(row_f1),
        value_exact_accuracy=float(exact_acc),
        value_tolerant_accuracy=float(tolerant_acc),
        gt_row_count=int(sum(len(v) for v in gt_rows.values())),
        pred_row_count=int(sum(len(v) for v in pred_rows.values())),
        matched_row_count=int(matched_row_count),
    )
