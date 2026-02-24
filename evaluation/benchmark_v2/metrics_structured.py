"""
Structured output metrics for benchmark v2.

Evaluates UI-visible table rows and numeric values after parsing.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Tuple

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
    return f"{statement}|name:{_normalize_text(name)}"


def _flatten_rows(obj: Dict[str, Any]) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {}
    for st in STATEMENTS:
        st_obj = obj.get(st) or {}
        items = st_obj.get("items") or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            key = _make_row_key(st, item)
            out[key] = _coerce_value(item.get("value"))
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


def calculate_structured_metrics(
    prediction: Dict[str, Any],
    reference: Dict[str, Any],
    *,
    abs_tolerance: float = 1.0,
    rel_tolerance: float = 1e-6,
) -> StructuredMetricResult:
    schema_valid = 1.0 if _is_schema_valid(prediction) else 0.0

    pred_rows = _flatten_rows(prediction)
    gt_rows = _flatten_rows(reference)

    pred_keys = set(pred_rows.keys())
    gt_keys = set(gt_rows.keys())
    matched_keys = pred_keys & gt_keys

    row_p, row_r, row_f1 = _prf(len(matched_keys), len(pred_keys), len(gt_keys))

    if not matched_keys:
        exact_acc = 0.0 if gt_keys else 1.0
        tolerant_acc = exact_acc
    else:
        exact_ok = 0
        tolerant_ok = 0
        for k in matched_keys:
            pv = pred_rows.get(k)
            gv = gt_rows.get(k)
            if pv == gv:
                exact_ok += 1
            if _values_close(pv, gv, abs_tol=abs_tolerance, rel_tol=rel_tolerance):
                tolerant_ok += 1
        denom = len(matched_keys)
        exact_acc = exact_ok / denom if denom else 1.0
        tolerant_acc = tolerant_ok / denom if denom else 1.0

    return StructuredMetricResult(
        schema_valid=float(schema_valid),
        row_precision=float(row_p),
        row_recall=float(row_r),
        row_f1=float(row_f1),
        value_exact_accuracy=float(exact_acc),
        value_tolerant_accuracy=float(tolerant_acc),
        gt_row_count=int(len(gt_keys)),
        pred_row_count=int(len(pred_keys)),
        matched_row_count=int(len(matched_keys)),
    )

