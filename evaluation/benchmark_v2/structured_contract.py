"""
Shared structured-output contract helpers for benchmark v2.

These helpers keep row identity logic consistent across:
- CSV canonicalization
- report assembly
- structured metrics
- debug diffs
- benchmark audits
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Tuple

STATEMENT_ITEM_FIELDS = (
    "item_code",
    "item_name",
    "value",
    "notes_ref",
    "original_name",
    "row_identity",
    "column_label",
    "period_key",
)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_text_ascii(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d").replace("Đ", "D")
    return normalize_text(text)


def normalize_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def coerce_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except Exception:
        return None


def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "item_code": item.get("item_code"),
        "item_name": item.get("item_name"),
        "value": item.get("value"),
        "notes_ref": item.get("notes_ref"),
        "original_name": item.get("original_name"),
        "row_identity": item.get("row_identity"),
        "column_label": item.get("column_label"),
        "period_key": item.get("period_key"),
    }


def build_row_identity(statement: str, item: Dict[str, Any], *, fallback: str) -> str:
    explicit = normalize_optional_text(item.get("row_identity"))
    if explicit:
        return f"{statement}|id:{normalize_text(explicit)}"

    code = normalize_optional_text(item.get("item_code"))
    note = normalize_optional_text(item.get("notes_ref"))
    name = normalize_optional_text(item.get("item_name"))
    original_name = normalize_optional_text(item.get("original_name"))

    if code:
        return f"{statement}|code:{normalize_text(code)}"
    if name and note:
        return f"{statement}|name:{normalize_text(name)}|note:{normalize_text(note)}"
    if name:
        return f"{statement}|name:{normalize_text(name)}"
    if original_name:
        return f"{statement}|orig:{normalize_text(original_name)}"
    return f"{statement}|fallback:{normalize_text(fallback)}"


def build_column_identity(item: Dict[str, Any]) -> str | None:
    period_key = normalize_optional_text(item.get("period_key"))
    column_label = normalize_optional_text(item.get("column_label"))
    if period_key:
        return f"period:{normalize_text(period_key)}"
    if column_label:
        return f"column:{normalize_text(column_label)}"
    return None


def build_row_key(statement: str, item: Dict[str, Any], *, fallback: str) -> str:
    row_identity = build_row_identity(statement, item, fallback=fallback)
    column_identity = build_column_identity(item)
    if column_identity:
        return f"{row_identity}|{column_identity}"
    return row_identity


def iter_statement_items(obj: Dict[str, Any], statements: Iterable[str]) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
    for statement in statements:
        node = obj.get(statement, {})
        items = node.get("items", []) if isinstance(node, dict) else []
        if not isinstance(items, list):
            continue
        for idx, raw_item in enumerate(items):
            if not isinstance(raw_item, dict):
                continue
            yield statement, idx, normalize_item(raw_item)


def values_close(a: float | None, b: float | None, *, abs_tol: float, rel_tol: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def detect_exact_factor(pred: float | None, gt: float | None) -> float | None:
    if pred is None or gt is None:
        return None
    if pred == 0.0 or gt == 0.0:
        return None
    ratio = abs(pred / gt)
    canonical = (1e-9, 1e-6, 1e-3, 1e3, 1e6, 1e9)
    for factor in canonical:
        if math.isclose(ratio, factor, rel_tol=1e-9, abs_tol=1e-12):
            return float(factor)
    return None


def sign_mismatch(pred: float | None, gt: float | None) -> bool:
    if pred is None or gt is None:
        return False
    if pred == 0.0 or gt == 0.0:
        return False
    return (pred < 0 < gt) or (gt < 0 < pred)


def count_repeated_row_identities(obj: Dict[str, Any], statements: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for statement, idx, item in iter_statement_items(obj, statements):
        key = build_row_identity(statement, item, fallback=f"{statement}:{idx}")
        counts[key] = counts.get(key, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}


def detect_unit_scale_from_text(markdown: str) -> Tuple[str | None, float]:
    norm = normalize_text_ascii(markdown)
    if not norm:
        return None, 1.0

    patterns = [
        (("ty vnd", "ty dong", "ty vietnam dong", "ty d"), 1_000_000_000.0),
        (("trieu vnd", "trieu dong", "trieu vietnam dong", "trieu d"), 1_000_000.0),
        (("nghin vnd", "nghin dong", "nghin vietnam dong", "nghin d"), 1_000.0),
        (("ngan vnd", "ngan dong", "ngan vietnam dong", "ngan d"), 1_000.0),
        (("vnd", "dong", "viet nam dong", "vietnam dong"), 1.0),
    ]
    for aliases, scale in patterns:
        for alias in aliases:
            if alias in norm:
                return alias, scale
    return None, 1.0


def extract_structured_rows(obj: Dict[str, Any], statements: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for statement, idx, item in iter_statement_items(obj, statements):
        key = build_row_key(statement, item, fallback=f"{statement}:{idx}")
        row = {
            "statement": statement,
            **item,
        }
        out.setdefault(key, []).append(row)
    return out
