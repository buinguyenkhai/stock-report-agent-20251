"""
CSV codec utilities for benchmark v2 annotation workflow.

CSV pack layout per sample:
  gt_csv/<sample_id>/
    cells.csv   : row_idx,col_idx,text
    rows.csv    : statement,item_code,item_name,value,notes_ref,original_name
                  where value is canonical VND
    meta.json   : optional QA metadata
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .dataset import BenchmarkDatasetV2, TableSample

STATEMENTS = ("balance_sheet", "income_statement", "cash_flow")

CELLS_COLUMNS = ("row_idx", "col_idx", "text")
ROWS_COLUMNS = ("statement", "item_code", "item_name", "value", "notes_ref", "original_name")


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
