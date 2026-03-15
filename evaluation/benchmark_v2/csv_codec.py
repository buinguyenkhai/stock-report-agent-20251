"""
CSV codec utilities for benchmark v2 annotation workflow.

CSV pack layout per sample:
  gt_csv/<sample_id>/
    cells.csv   : row_idx,col_idx,text
    rows.csv    : statement,item_code,item_name,value,notes_ref,original_name,row_identity,column_label,period_key
                  where value is canonical VND
    meta.json   : optional QA metadata
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .dataset import BenchmarkDatasetV2, TableSample
from .structured_contract import normalize_text_ascii

STATEMENTS = ("balance_sheet", "income_statement", "cash_flow")

CELLS_COLUMNS = ("row_idx", "col_idx", "text")
ROWS_COLUMNS = (
    "statement",
    "item_code",
    "item_name",
    "value",
    "notes_ref",
    "original_name",
    "row_identity",
    "column_label",
    "period_key",
)


@dataclass(frozen=True)
class CsvPackPaths:
    root: Path
    cells_csv: Path
    rows_csv: Path
    meta_json: Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "null":
        return None
    return s


def _parse_int(value: Any, *, field: str, minimum: Optional[int] = None) -> int:
    s = str(value).strip()
    if s == "":
        raise ValueError(f"{field} is empty")
    try:
        out = int(s)
    except Exception as e:
        raise ValueError(f"{field} must be an integer: {value}") from e
    if minimum is not None and out < minimum:
        raise ValueError(f"{field} must be >= {minimum}: {value}")
    return out


def _parse_numeric_value(raw: Any) -> float | None:
    s0 = _to_optional_text(raw)
    if s0 is None:
        return None

    s = s0.replace(" ", "")
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    if s.startswith("-"):
        negative = True
        s = s[1:]
    if s.startswith("+"):
        s = s[1:]
    if s.endswith("%"):
        s = s[:-1]

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        if s.count(",") > 1:
            s = s.replace(",", "")
        else:
            tail = s.split(",")[-1]
            if len(tail) in (1, 2):
                s = s.replace(",", ".")
            else:
                s = s.replace(",", "")
    elif "." in s:
        if s.count(".") > 1:
            parts = s.split(".")
            if len(parts[-1]) in (1, 2):
                s = "".join(parts[:-1]) + "." + parts[-1]
            else:
                s = "".join(parts)

    try:
        val = float(s)
    except Exception as e:
        raise ValueError(f"Invalid numeric value: {s0}") from e
    return -val if negative else val


def _csv_paths(dataset_root: str | Path, sample_id: str) -> CsvPackPaths:
    root = Path(dataset_root) / "gt_csv" / sample_id
    return CsvPackPaths(
        root=root,
        cells_csv=root / "cells.csv",
        rows_csv=root / "rows.csv",
        meta_json=root / "meta.json",
    )


def _find_sample(ds: BenchmarkDatasetV2, sample_id: str) -> TableSample:
    for s in ds.samples:
        if s.sample_id == sample_id:
            return s
    raise KeyError(f"sample_id not found in manifest: {sample_id}")


def _empty_df(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _read_csv_or_empty(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    expected = list(columns)
    if not path.exists():
        return _empty_df(expected)
    df = pd.read_csv(path, dtype=str).fillna("")
    for c in expected:
        if c not in df.columns:
            df[c] = ""
    return df[expected].copy()


def _write_df_csv(path: Path, df: pd.DataFrame, columns: Iterable[str]) -> None:
    cols = list(columns)
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    out = out[cols].fillna("")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def load_csv_pack(sample_id: str, dataset_root: str | Path) -> Dict[str, pd.DataFrame]:
    p = _csv_paths(dataset_root, sample_id)
    return {
        "cells": _read_csv_or_empty(p.cells_csv, CELLS_COLUMNS),
        "rows": _read_csv_or_empty(p.rows_csv, ROWS_COLUMNS),
    }


def save_csv_pack(
    sample_id: str,
    dataset_root: str | Path,
    *,
    cells: pd.DataFrame,
    rows: pd.DataFrame,
    meta_updates: Optional[Dict[str, Any]] = None,
) -> CsvPackPaths:
    p = _csv_paths(dataset_root, sample_id)
    _write_df_csv(p.cells_csv, cells, CELLS_COLUMNS)
    _write_df_csv(p.rows_csv, rows, ROWS_COLUMNS)
    if meta_updates is not None:
        meta = load_meta(sample_id, dataset_root)
        meta.update(meta_updates)
        meta["updated_at"] = _now_iso()
        p.meta_json.parent.mkdir(parents=True, exist_ok=True)
        p.meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_meta(sample_id: str, dataset_root: str | Path) -> Dict[str, Any]:
    p = _csv_paths(dataset_root, sample_id).meta_json
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def update_meta(sample_id: str, dataset_root: str | Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    p = _csv_paths(dataset_root, sample_id).meta_json
    meta = load_meta(sample_id, dataset_root)
    meta.update(updates or {})
    meta["updated_at"] = _now_iso()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _check_required_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> List[str]:
    req = list(required)
    missing = [c for c in req if c not in df.columns]
    if missing:
        return [f"{label}: missing required columns: {missing}"]
    return []


def _iter_non_empty_records(df: pd.DataFrame, columns: Iterable[str]) -> Iterable[Tuple[int, Dict[str, str]]]:
    cols = list(columns)
    for idx, row in df.iterrows():
        rec: Dict[str, str] = {}
        for c in cols:
            v = row.get(c, "")
            if v is None or pd.isna(v):
                v = ""
            rec[c] = str(v).strip()
        if all(v == "" for v in rec.values()):
            continue
        yield int(idx), rec


def _parse_csv_frames(
    *,
    cells: pd.DataFrame,
    rows: pd.DataFrame,
) -> Tuple[
    Dict[Tuple[int, int], str],
    List[Dict[str, Any]],
    List[str],
]:
    errors: List[str] = []
    errors.extend(_check_required_columns(cells, CELLS_COLUMNS, "cells.csv"))
    errors.extend(_check_required_columns(rows, ROWS_COLUMNS, "rows.csv"))
    if errors:
        return {}, [], errors

    cell_map: Dict[Tuple[int, int], str] = {}
    for i, rec in _iter_non_empty_records(cells, CELLS_COLUMNS):
        try:
            r = _parse_int(rec["row_idx"], field=f"cells.csv row {i} row_idx", minimum=0)
            c = _parse_int(rec["col_idx"], field=f"cells.csv row {i} col_idx", minimum=0)
        except ValueError as e:
            errors.append(str(e))
            continue
        key = (r, c)
        if key in cell_map:
            errors.append(f"cells.csv duplicate cell anchor at {key}")
            continue
        cell_map[key] = rec["text"]

    struct_rows: List[Dict[str, Any]] = []
    for i, rec in _iter_non_empty_records(rows, ROWS_COLUMNS):
        st = rec["statement"].strip().lower()
        if st not in STATEMENTS:
            errors.append(f"rows.csv row {i} invalid statement: {rec['statement']}")
            continue
        item_name = rec["item_name"].strip()
        if not item_name:
            errors.append(f"rows.csv row {i} item_name is required")
            continue
        try:
            value = _parse_numeric_value(rec["value"])
        except ValueError as e:
            errors.append(f"rows.csv row {i} {e}")
            continue
        struct_rows.append(
            {
                "statement": st,
                "item_code": _to_optional_text(rec["item_code"]),
                "item_name": item_name,
                "value": value,
                "notes_ref": _to_optional_text(rec["notes_ref"]),
                "original_name": _to_optional_text(rec["original_name"]),
                "row_identity": _to_optional_text(rec["row_identity"]),
                "column_label": _to_optional_text(rec["column_label"]),
                "period_key": _to_optional_text(rec["period_key"]),
            }
        )

    return cell_map, struct_rows, errors


def validate_csv_frames(
    *,
    cells: pd.DataFrame,
    rows: pd.DataFrame,
) -> List[str]:
    _cells, _struct_rows, errors = _parse_csv_frames(cells=cells, rows=rows)
    return errors


def validate_csv_pack(sample_id: str, dataset_root: str | Path) -> List[str]:
    p = _csv_paths(dataset_root, sample_id)
    errors: List[str] = []
    if not p.root.exists():
        errors.append(f"gt_csv pack not found: {p.root}")
        return errors

    for required_path in (p.cells_csv, p.rows_csv):
        if not required_path.exists():
            errors.append(f"missing required file: {required_path}")

    pack = load_csv_pack(sample_id, dataset_root)
    return errors + validate_csv_frames(cells=pack["cells"], rows=pack["rows"])


def _build_table_matrix(
    cell_map: Dict[Tuple[int, int], str],
) -> List[List[str]]:
    if not cell_map:
        return []
    max_row = max(r for r, _c in cell_map.keys())
    max_col = max(c for _r, c in cell_map.keys())

    grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for (r, c), text in cell_map.items():
        grid[r][c] = str(text or "")
    return grid


def _markdown_from_matrix(grid: List[List[str]]) -> str:
    if not grid:
        return ""

    def _esc(s: str) -> str:
        return str(s or "").replace("|", "\\|").replace("\n", "<br>")

    width = max((len(r) for r in grid), default=0)
    lines: List[str] = []
    for r in grid:
        cells = [_esc(r[i]) if i < len(r) else "" for i in range(width)]
        lines.append("| " + " | ".join(cells) + " |")

    if width > 0:
        sep = "| " + " | ".join(["---"] * width) + " |"
        if lines:
            lines = [lines[0], sep] + lines[1:]
    return "\n".join(lines).strip()


def _gt_cells_from_matrix(grid: List[List[str]]) -> Dict[str, Any]:
    rows: List[List[Dict[str, Any]]] = []
    for row in grid:
        out_row: List[Dict[str, Any]] = []
        for text in row:
            out_row.append({"text": str(text or "")})
        rows.append(out_row)
    return {"rows": rows}


def _structured_from_rows(parsed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "balance_sheet": {"items": []},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    for row in parsed_rows:
        st = row["statement"]
        item = {
            "item_code": row["item_code"],
            "item_name": row["item_name"],
            "value": row["value"],
            "notes_ref": row["notes_ref"],
            "original_name": row["original_name"],
            "row_identity": row["row_identity"],
            "column_label": row["column_label"],
            "period_key": row["period_key"],
        }
        out[st]["items"].append(item)
    return out


def build_canonical_from_frames(
    *,
    cells: pd.DataFrame,
    rows: pd.DataFrame,
    validate: bool = True,
) -> Dict[str, Any]:
    cell_map, parsed_rows, errors = _parse_csv_frames(cells=cells, rows=rows)
    if validate and errors:
        raise ValueError("CSV validation failed:\n" + "\n".join(f"- {e}" for e in errors))
    grid = _build_table_matrix(cell_map)
    return {
        "gt_markdown": _markdown_from_matrix(grid),
        "gt_structured": _structured_from_rows(parsed_rows),
        "gt_cells": _gt_cells_from_matrix(grid),
        "validation_errors": errors,
    }


def csv_to_canonical(sample_id: str, dataset_root: str | Path, validate: bool = True) -> Dict[str, Any]:
    ds = BenchmarkDatasetV2(dataset_root)
    sample = _find_sample(ds, sample_id)
    pack = load_csv_pack(sample_id, dataset_root)
    canonical = build_canonical_from_frames(
        cells=pack["cells"],
        rows=pack["rows"],
        validate=validate,
    )

    gt_md_path = ds.dataset_root / sample.gt_markdown_path
    gt_struct_path = ds.dataset_root / sample.gt_structured_path
    if sample.gt_table_cells_path:
        gt_cells_path = ds.dataset_root / sample.gt_table_cells_path
    else:
        gt_cells_path = ds.dataset_root / "gt_cells" / f"{sample_id}.json"

    gt_md_path.parent.mkdir(parents=True, exist_ok=True)
    gt_struct_path.parent.mkdir(parents=True, exist_ok=True)
    gt_cells_path.parent.mkdir(parents=True, exist_ok=True)

    gt_md_path.write_text(canonical["gt_markdown"], encoding="utf-8")
    gt_struct_path.write_text(
        json.dumps(canonical["gt_structured"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gt_cells_path.write_text(
        json.dumps(canonical["gt_cells"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    update_meta(
        sample_id,
        dataset_root,
        {
            "last_generated_at": _now_iso(),
            "last_generated_paths": {
                "gt_markdown_path": str(gt_md_path),
                "gt_structured_path": str(gt_struct_path),
                "gt_table_cells_path": str(gt_cells_path),
            },
        },
    )

    return {
        "sample_id": sample_id,
        "gt_markdown_path": str(gt_md_path),
        "gt_structured_path": str(gt_struct_path),
        "gt_table_cells_path": str(gt_cells_path),
        "validation_errors": canonical["validation_errors"],
    }


def _parse_markdown_pipe_rows(markdown_text: str) -> List[List[str]]:
    rows: List[List[str]] = []
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
        rows.append(parts)
    return rows


def _load_cells_from_gt_cells(path: Path) -> pd.DataFrame:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj.get("rows", []) if isinstance(obj, dict) else []
    if not isinstance(rows, list):
        return _empty_df(CELLS_COLUMNS)

    cell_rows: List[Dict[str, Any]] = []
    for r, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        for c, cell in enumerate(row):
            if isinstance(cell, dict):
                text = str(cell.get("text", ""))
            else:
                text = str(cell or "")
            cell_rows.append({"row_idx": r, "col_idx": c, "text": text})

    return pd.DataFrame(cell_rows, columns=list(CELLS_COLUMNS)).fillna("")


def _rows_from_structured(structured: Dict[str, Any]) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []
    for st in STATEMENTS:
        node = structured.get(st, {})
        items = node.get("items", []) if isinstance(node, dict) else []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            out_rows.append(
                {
                    "statement": st,
                    "item_code": item.get("item_code") if item.get("item_code") is not None else "",
                    "item_name": item.get("item_name") if item.get("item_name") is not None else "",
                    "value": item.get("value") if item.get("value") is not None else "",
                    "notes_ref": item.get("notes_ref") if item.get("notes_ref") is not None else "",
                    "original_name": (
                        item.get("original_name") if item.get("original_name") is not None else ""
                    ),
                    "row_identity": (
                        item.get("row_identity") if item.get("row_identity") is not None else ""
                    ),
                    "column_label": (
                        item.get("column_label") if item.get("column_label") is not None else ""
                    ),
                    "period_key": item.get("period_key") if item.get("period_key") is not None else "",
                }
            )
    return pd.DataFrame(out_rows, columns=list(ROWS_COLUMNS)).fillna("")


def canonical_to_csv(sample_id: str, dataset_root: str | Path) -> Dict[str, Any]:
    ds = BenchmarkDatasetV2(dataset_root)
    sample = _find_sample(ds, sample_id)

    gt_struct_path = ds.dataset_root / sample.gt_structured_path
    gt_md_path = ds.dataset_root / sample.gt_markdown_path
    gt_cells_path = (
        ds.dataset_root / sample.gt_table_cells_path
        if sample.gt_table_cells_path
        else ds.dataset_root / "gt_cells" / f"{sample_id}.json"
    )

    if not gt_struct_path.exists():
        raise FileNotFoundError(f"Missing gt_structured file: {gt_struct_path}")

    structured = json.loads(gt_struct_path.read_text(encoding="utf-8"))
    if not isinstance(structured, dict):
        raise ValueError(f"Invalid JSON object in {gt_struct_path}")
    rows_df = _rows_from_structured(structured)

    if gt_cells_path.exists():
        cells_df = _load_cells_from_gt_cells(gt_cells_path)
    else:
        md_text = gt_md_path.read_text(encoding="utf-8") if gt_md_path.exists() else ""
        md_rows = _parse_markdown_pipe_rows(md_text)
        cell_rows = [
            {"row_idx": r, "col_idx": c, "text": text}
            for r, row in enumerate(md_rows)
            for c, text in enumerate(row)
        ]
        cells_df = pd.DataFrame(cell_rows, columns=list(CELLS_COLUMNS)).fillna("")

    meta = load_meta(sample_id, dataset_root)
    save_csv_pack(
        sample_id,
        dataset_root,
        cells=cells_df,
        rows=rows_df,
        meta_updates={**meta, "last_imported_from_canonical_at": _now_iso()},
    )
    return {
        "sample_id": sample_id,
        "csv_root": str(_csv_paths(dataset_root, sample_id).root),
        "cells_count": int(len(cells_df)),
        "rows_count": int(len(rows_df)),
    }


_GENERIC_HEADER_KEYS = {
    "",
    "ma so",
    "chi tieu",
    "tai san",
    "nguon von",
    "thuyet minh",
}
_POINT_IN_TIME_DATE_RE = re.compile(r"(?P<day>\d{1,2})[/-](?P<month>\d{1,2})[/-](?P<year>\d{4})")
_DATE_RANGE_RE = re.compile(
    r"tu\s*\d{1,2}[/-]\d{1,2}[/-](?P<year>\d{4})\s*den\s*(?P<day>\d{1,2})[/-](?P<month>\d{1,2})[/-](?P=year)",
    re.IGNORECASE,
)
_QUARTER_RE = re.compile(r"quy\s*(?P<quarter>[ivx]+|\d)\s*(?:nam)?\s*(?P<year>\d{4})", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_LEADING_CODE_RE = re.compile(r"^\s*(?P<code>[A-Za-z]{0,3}\d{1,4}|[IVXLCDM]+)\s+(?P<rest>.+?)\s*$", re.IGNORECASE)


def _parse_report_period_context(report_id: str) -> Dict[str, int | None]:
    match = re.search(r"_Q(?P<quarter>\d)_(?P<year>\d{4})$", str(report_id))
    if not match:
        return {"year": None, "quarter": None}
    return {
        "year": int(match.group("year")),
        "quarter": int(match.group("quarter")),
    }


def _roman_quarter_to_int(value: str) -> int | None:
    value_norm = str(value or "").strip().upper()
    mapping = {"I": 1, "II": 2, "III": 3, "IV": 4}
    if value_norm in mapping:
        return mapping[value_norm]
    try:
        parsed = int(value_norm)
    except Exception:
        return None
    return parsed if 1 <= parsed <= 4 else None


def _infer_period_key(column_label: str, *, report_id: str) -> str:
    label = str(column_label or "").strip()
    if not label:
        return ""
    norm = normalize_text_ascii(label)
    context = _parse_report_period_context(report_id)
    report_year = context["year"]
    report_quarter = context["quarter"]

    date_range = _DATE_RANGE_RE.search(norm)
    if date_range:
        year = int(date_range.group("year"))
        month = int(date_range.group("month"))
        quarter = ((month - 1) // 3) + 1
        return f"{year}Q{quarter}_YTD"

    quarter_match = _QUARTER_RE.search(norm)
    if quarter_match:
        quarter = _roman_quarter_to_int(quarter_match.group("quarter"))
        year = int(quarter_match.group("year"))
        if quarter is not None:
            return f"{year}Q{quarter}"

    dates = list(_POINT_IN_TIME_DATE_RE.finditer(norm))
    if dates:
        day = int(dates[-1].group("day"))
        month = int(dates[-1].group("month"))
        year = int(dates[-1].group("year"))
        if day == 31 and month == 12:
            return f"{year}FY"
        return f"{year:04d}-{month:02d}-{day:02d}"

    if "nam nay" in norm and report_year:
        if report_quarter == 4:
            return f"{report_year}FY"
        if report_quarter:
            return f"{report_year}Q{report_quarter}_YTD"
        return str(report_year)
    if "nam truoc" in norm and report_year:
        previous_year = int(report_year) - 1
        if report_quarter == 4:
            return f"{previous_year}FY"
        if report_quarter:
            return f"{previous_year}Q{report_quarter}_YTD"
        return str(previous_year)

    year_match = _YEAR_RE.search(norm)
    if year_match:
        year = int(year_match.group(1))
        if "quy" not in norm and "nam" in norm:
            return f"{year}FY"
        return str(year)

    return ""


def _safe_cell(grid: List[List[str]], row_idx: int, col_idx: int) -> str:
    if row_idx < 0 or row_idx >= len(grid):
        return ""
    row = grid[row_idx]
    if col_idx < 0 or col_idx >= len(row):
        return ""
    return str(row[col_idx] or "").strip()


def _grid_width(grid: List[List[str]]) -> int:
    return max((len(row) for row in grid), default=0)


def _column_values(grid: List[List[str]], col_idx: int) -> List[str]:
    return [_safe_cell(grid, row_idx, col_idx) for row_idx in range(len(grid))]


def _best_matching_column(candidates: List[str], grid: List[List[str]], *, exclude: set[int]) -> int | None:
    normalized_candidates = {normalize_text_ascii(value) for value in candidates if str(value or "").strip()}
    if not normalized_candidates:
        return None

    best_col: int | None = None
    best_score = 0
    for col_idx in range(_grid_width(grid)):
        if col_idx in exclude:
            continue
        score = 0
        for cell in _column_values(grid, col_idx):
            if normalize_text_ascii(cell) in normalized_candidates:
                score += 1
        if score > best_score:
            best_col = col_idx
            best_score = score
    return best_col if best_score > 0 else None


def _detect_descriptor_columns(grid: List[List[str]], legacy_rows: List[Dict[str, Any]]) -> Dict[str, int | None]:
    name_candidates = [
        value
        for row in legacy_rows
        for value in (row.get("item_name"), row.get("original_name"))
        if str(value or "").strip()
    ]
    code_candidates = [row.get("item_code") for row in legacy_rows if str(row.get("item_code") or "").strip()]
    note_candidates = [row.get("notes_ref") for row in legacy_rows if str(row.get("notes_ref") or "").strip()]

    name_col = _best_matching_column(name_candidates, grid, exclude=set())
    exclude = {idx for idx in (name_col,) if idx is not None}
    code_col = _best_matching_column(code_candidates, grid, exclude=exclude)
    exclude = {idx for idx in (name_col, code_col) if idx is not None}
    note_col = _best_matching_column(note_candidates, grid, exclude=exclude)
    return {"name": name_col, "code": code_col, "note": note_col}


def _split_leading_code(text: str) -> Tuple[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return "", ""
    match = _LEADING_CODE_RE.match(raw)
    if not match:
        return "", raw
    return str(match.group("code") or "").strip(), str(match.group("rest") or "").strip()


def _collapse_legacy_rows(legacy_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str, str]] = set()
    for row in legacy_rows:
        key = (
            str(row.get("statement") or ""),
            str(row.get("item_code") or "").strip(),
            normalize_text_ascii(row.get("item_name")),
            str(row.get("notes_ref") or "").strip(),
            normalize_text_ascii(row.get("original_name")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _row_match_score(
    legacy_row: Dict[str, Any],
    row_cells: List[str],
    *,
    descriptor_cols: Dict[str, int | None],
) -> int:
    score = 0
    name_col = descriptor_cols.get("name")
    code_col = descriptor_cols.get("code")
    note_col = descriptor_cols.get("note")

    name = normalize_text_ascii(legacy_row.get("item_name"))
    original_name = normalize_text_ascii(legacy_row.get("original_name"))
    code = normalize_text_ascii(legacy_row.get("item_code"))
    note = normalize_text_ascii(legacy_row.get("notes_ref"))

    if code_col is not None and code:
        code_cell = normalize_text_ascii(row_cells[code_col] if code_col < len(row_cells) else "")
        if code_cell == code:
            score += 100
    if name_col is not None:
        raw_name_cell = row_cells[name_col] if name_col < len(row_cells) else ""
        code_prefix, stripped_name = _split_leading_code(raw_name_cell)
        name_cell = normalize_text_ascii(raw_name_cell)
        stripped_name_cell = normalize_text_ascii(stripped_name)
        if name_cell == name or stripped_name_cell == name:
            score += 70
        elif original_name and (name_cell == original_name or stripped_name_cell == original_name):
            score += 60
        if code and normalize_text_ascii(code_prefix) == code:
            score += 20
    if note_col is not None and note:
        note_cell = normalize_text_ascii(row_cells[note_col] if note_col < len(row_cells) else "")
        if note_cell == note:
            score += 20

    if score == 0:
        row_keys = {normalize_text_ascii(cell) for cell in row_cells if str(cell or "").strip()}
        if name and name in row_keys:
            score += 40
        elif original_name and original_name in row_keys:
            score += 30
        elif code and code in row_keys:
            score += 25
    return score


def _match_legacy_rows_to_grid(
    grid: List[List[str]],
    legacy_rows: List[Dict[str, Any]],
    *,
    descriptor_cols: Dict[str, int | None],
) -> List[Dict[str, Any]]:
    used_grid_rows: set[int] = set()
    matched: List[Dict[str, Any]] = []
    last_grid_row_idx = -1

    for legacy_idx, legacy_row in enumerate(legacy_rows):
        best_row_idx: int | None = None
        best_score = -1
        for grid_row_idx, row_cells in enumerate(grid):
            if grid_row_idx in used_grid_rows:
                continue
            score = _row_match_score(legacy_row, row_cells, descriptor_cols=descriptor_cols)
            if score <= 0:
                continue
            if score > best_score or (score == best_score and best_row_idx is not None and grid_row_idx < best_row_idx):
                best_row_idx = grid_row_idx
                best_score = score

        if best_row_idx is None:
            for grid_row_idx, row_cells in enumerate(grid):
                if grid_row_idx in used_grid_rows or grid_row_idx <= last_grid_row_idx:
                    continue
                has_numeric = False
                for cell in row_cells:
                    try:
                        parsed = _parse_numeric_value(cell)
                    except ValueError:
                        parsed = None
                    if parsed is not None:
                        has_numeric = True
                        break
                if has_numeric:
                    best_row_idx = grid_row_idx
                    break

        if best_row_idx is None:
            continue
        used_grid_rows.add(best_row_idx)
        last_grid_row_idx = best_row_idx
        matched.append(
            {
                "legacy_index": legacy_idx,
                "grid_row_idx": best_row_idx,
                "legacy_row": legacy_row,
                "grid_row": list(grid[best_row_idx]),
            }
        )
    return matched


def _header_label_for_column(grid: List[List[str]], col_idx: int, first_data_row_idx: int) -> str:
    labels: List[str] = []
    seen: set[str] = set()
    for row_idx in range(max(0, first_data_row_idx)):
        text = _safe_cell(grid, row_idx, col_idx)
        if not text:
            continue
        key = normalize_text_ascii(text)
        if key in _GENERIC_HEADER_KEYS or key in seen:
            continue
        seen.add(key)
        labels.append(text)
    return " ".join(labels).strip()


def _derive_value_columns(
    grid: List[List[str]],
    matched_rows: List[Dict[str, Any]],
    *,
    descriptor_cols: Dict[str, int | None],
) -> List[Tuple[int, str]]:
    excluded = {idx for idx in descriptor_cols.values() if idx is not None}
    if not matched_rows:
        return []

    first_data_row_idx = min(int(row["grid_row_idx"]) for row in matched_rows)
    candidates: List[Tuple[int, str]] = []
    for col_idx in range(_grid_width(grid)):
        if col_idx in excluded:
            continue
        numeric_count = 0
        non_empty_count = 0
        for row in matched_rows:
            text = _safe_cell(grid, int(row["grid_row_idx"]), col_idx)
            if not text:
                continue
            non_empty_count += 1
            try:
                parsed = _parse_numeric_value(text)
            except ValueError:
                parsed = None
            if parsed is not None:
                numeric_count += 1
        header_label = _header_label_for_column(grid, col_idx, first_data_row_idx)
        if numeric_count == 0:
            continue
        if non_empty_count == 0 and not header_label:
            continue
        candidates.append((col_idx, header_label))
    return candidates


def _build_explicit_row_identity(
    *,
    statement: str,
    item_code: str | None,
    item_name: str,
    notes_ref: str | None,
    occurrence: int,
    total_occurrences: int,
) -> str:
    code = str(item_code or "").strip()
    note = str(notes_ref or "").strip()
    name = str(item_name or "").strip()
    if code:
        return f"{statement}|code:{code}"
    if note:
        return f"{statement}|name:{name}|note:{note}"
    if total_occurrences > 1:
        return f"{statement}|name:{name}|occ:{occurrence}"
    return f"{statement}|name:{name}"


def migrate_rows_to_structured_contract(
    sample_id: str,
    dataset_root: str | Path,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    ds = BenchmarkDatasetV2(dataset_root)
    sample = _find_sample(ds, sample_id)
    pack = load_csv_pack(sample_id, dataset_root)
    rows_df = pack["rows"]
    cells_df = pack["cells"]
    meta = load_meta(sample_id, dataset_root)
    normalize_to_vnd = str(meta.get("value_unit_normalized_to") or "").strip().upper() == "VND"
    report_unit_multiplier = float(meta.get("report_unit_multiplier") or 1.0)
    if report_unit_multiplier <= 0:
        report_unit_multiplier = 1.0

    already_migrated = False
    if not rows_df.empty:
        for column in ("column_label", "period_key"):
            if column in rows_df.columns and rows_df[column].astype(str).str.strip().ne("").any():
                already_migrated = True
                break
    if already_migrated and not force:
        return {
            "sample_id": sample_id,
            "changed": False,
            "reason": "already_migrated",
            "rows_before": int(len(rows_df)),
            "rows_after": int(len(rows_df)),
        }

    cell_map, legacy_rows, errors = _parse_csv_frames(cells=cells_df, rows=rows_df)
    if errors:
        raise ValueError("CSV validation failed before migration:\n" + "\n".join(f"- {e}" for e in errors))
    legacy_rows = _collapse_legacy_rows(legacy_rows)
    if not legacy_rows:
        return {
            "sample_id": sample_id,
            "changed": False,
            "reason": "no_annotated_rows",
            "rows_before": int(len(rows_df)),
            "rows_after": int(len(rows_df)),
        }

    grid = _build_table_matrix(cell_map)
    descriptor_cols = _detect_descriptor_columns(grid, legacy_rows)
    matched_rows = _match_legacy_rows_to_grid(grid, legacy_rows, descriptor_cols=descriptor_cols)
    if len(matched_rows) != len(legacy_rows):
        raise ValueError(
            f"Could not align all legacy rows for {sample_id}: matched {len(matched_rows)}/{len(legacy_rows)}"
        )

    value_columns = _derive_value_columns(grid, matched_rows, descriptor_cols=descriptor_cols)
    if not value_columns:
        raise ValueError(f"No numeric value columns detected for {sample_id}")

    duplicate_totals: Dict[Tuple[str, str], int] = {}
    for row in matched_rows:
        legacy_row = row["legacy_row"]
        base = str(legacy_row.get("item_code") or legacy_row.get("item_name") or legacy_row.get("original_name") or "").strip()
        key = (str(legacy_row["statement"]), base)
        duplicate_totals[key] = duplicate_totals.get(key, 0) + 1

    duplicate_seen: Dict[Tuple[str, str], int] = {}
    migrated_rows: List[Dict[str, Any]] = []
    for row in matched_rows:
        legacy_row = row["legacy_row"]
        grid_row_idx = int(row["grid_row_idx"])
        statement = str(legacy_row["statement"])
        item_code = legacy_row.get("item_code")
        item_name = str(legacy_row["item_name"])
        notes_ref = legacy_row.get("notes_ref")
        original_name = legacy_row.get("original_name")
        base = str(item_code or item_name or original_name or "").strip()
        duplicate_key = (statement, base)
        occurrence = duplicate_seen.get(duplicate_key, 0) + 1
        duplicate_seen[duplicate_key] = occurrence
        row_identity = _build_explicit_row_identity(
            statement=statement,
            item_code=item_code,
            item_name=item_name,
            notes_ref=notes_ref,
            occurrence=occurrence,
            total_occurrences=duplicate_totals.get(duplicate_key, 1),
        )

        for col_idx, column_label in value_columns:
            raw_value = _safe_cell(grid, grid_row_idx, col_idx)
            if not raw_value:
                continue
            try:
                value = _parse_numeric_value(raw_value)
            except ValueError:
                continue
            if value is None:
                continue
            if normalize_to_vnd and report_unit_multiplier != 1.0:
                value = float(value) * report_unit_multiplier
            migrated_rows.append(
                {
                    "statement": statement,
                    "item_code": item_code or "",
                    "item_name": item_name,
                    "value": value,
                    "notes_ref": notes_ref or "",
                    "original_name": original_name or "",
                    "row_identity": row_identity,
                    "column_label": column_label,
                    "period_key": _infer_period_key(column_label, report_id=sample.report_id),
                }
            )

    if not migrated_rows:
        raise ValueError(f"No migrated rows generated for {sample_id}")

    migrated_df = pd.DataFrame(migrated_rows, columns=list(ROWS_COLUMNS)).fillna("")
    save_csv_pack(sample_id, dataset_root, cells=cells_df, rows=migrated_df)
    csv_to_canonical(sample_id, dataset_root, validate=True)
    update_meta(
        sample_id,
        dataset_root,
        {
            "structured_contract_version": "row_identity_column_period_v1",
            "rows_migrated_from_cells": True,
            "rows_migrated_at": _now_iso(),
            "legacy_row_count": int(len(rows_df)),
            "migrated_row_count": int(len(migrated_df)),
            "value_column_labels": [label for _idx, label in value_columns],
        },
    )
    return {
        "sample_id": sample_id,
        "changed": True,
        "rows_before": int(len(rows_df)),
        "rows_after": int(len(migrated_df)),
        "value_column_count": int(len(value_columns)),
        "value_column_labels": [label for _idx, label in value_columns],
    }


def compute_pilot_metrics(
    dataset_root: str | Path,
    *,
    sample_ids: Optional[Iterable[str]] = None,
    threshold: float = 0.02,
) -> Dict[str, Any]:
    ds = BenchmarkDatasetV2(dataset_root)
    selected = set(sample_ids) if sample_ids is not None else None

    included: List[str] = []
    total_rows = 0
    row_key_corr = 0
    value_corr = 0

    for s in ds.samples:
        if selected is not None and s.sample_id not in selected:
            continue
        meta = load_meta(s.sample_id, dataset_root)
        included.append(s.sample_id)
        total_rows += int(meta.get("total_rows", 0) or 0)
        row_key_corr += int(meta.get("row_key_corrections", 0) or 0)
        value_corr += int(meta.get("value_corrections", 0) or 0)

    mismatch_rate = ((row_key_corr + value_corr) / total_rows if total_rows > 0 else 0.0)
    pass_gate = mismatch_rate < float(threshold)

    return {
        "sample_count": len(included),
        "sample_ids": included,
        "total_rows": int(total_rows),
        "row_key_corrections": int(row_key_corr),
        "value_corrections": int(value_corr),
        "row_value_mismatch_rate": float(mismatch_rate),
        "threshold": float(threshold),
        "pass_gate": bool(pass_gate),
    }
