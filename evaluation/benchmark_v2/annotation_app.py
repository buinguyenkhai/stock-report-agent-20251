"""
Streamlit annotation app for benchmark v2 (CSV-first workflow).

Run:
  streamlit run evaluation/benchmark_v2/annotation_app.py
"""

from __future__ import annotations

import json
import csv
from datetime import datetime, timezone
import hashlib
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st

try:
    from .csv_codec import (
        CELLS_COLUMNS,
        ROWS_COLUMNS,
        build_canonical_from_frames,
        canonical_to_csv,
        csv_to_canonical,
        load_csv_pack,
        load_meta,
        save_csv_pack,
        validate_csv_frames,
    )
    from .dataset import BenchmarkDatasetV2, TableSample
    from .report_assembler import build_gt_structured_report_files
    from .render_page_images import render_page_images
except ImportError:
    # Support "streamlit run evaluation/benchmark_v2/annotation_app.py"
    # where relative imports do not have package context.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from evaluation.benchmark_v2.csv_codec import (
        CELLS_COLUMNS,
        ROWS_COLUMNS,
        build_canonical_from_frames,
        canonical_to_csv,
        csv_to_canonical,
        load_csv_pack,
        load_meta,
        save_csv_pack,
        validate_csv_frames,
    )
    from evaluation.benchmark_v2.dataset import BenchmarkDatasetV2, TableSample
    from evaluation.benchmark_v2.report_assembler import build_gt_structured_report_files
    from evaluation.benchmark_v2.render_page_images import render_page_images

STATEMENTS = ("balance_sheet", "income_statement", "cash_flow")
INCLUDE_SCOPE_OPTIONS = ("not_included", "included", "all")
SESSION_SELECTED_SAMPLE_ID = "annotation_selected_sample_id"
SESSION_SELECTED_SAMPLE_INDEX = "annotation_selected_sample_index"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dataset_dirs(dataset_root: Path) -> None:
    for rel in ("pdf", "images", "gt_markdown", "gt_structured", "gt_cells", "gt_csv"):
        (dataset_root / rel).mkdir(parents=True, exist_ok=True)


def _make_empty_manifest() -> Dict[str, Any]:
    return {
        "version": "1.0.0",
        "split_policy": "company_heldout_dev_test",
        "annotation_protocol": "single_annotator_two_pass",
        "samples": [],
    }


def _company_from_report_id(report_id: str) -> str:
    s = (report_id or "").strip()
    if not s:
        return "UNK"
    token = s.split("_")[0].strip()
    return (token or "UNK").upper()


def _assign_company_splits(
    companies: List[str],
    *,
    split_mode: str,
) -> Dict[str, str]:
    unique = sorted({c.upper() for c in companies if c})
    if not unique:
        return {}
    if split_mode == "all_dev":
        return {c: "dev" for c in unique}

    # Deterministic company-heldout split (about 80/20 by company count).
    ranked = sorted(unique, key=lambda c: hashlib.md5(c.encode("utf-8")).hexdigest())
    if len(ranked) < 2:
        return {ranked[0]: "dev"}

    test_count = max(1, int(round(len(ranked) * 0.2)))
    test_count = min(test_count, len(ranked) - 1)
    test_set = set(ranked[:test_count])
    return {c: ("test" if c in test_set else "dev") for c in ranked}


def _build_manifest_from_pdfs(
    *,
    dataset_root: Path,
    split_mode: str,
    annotator_id: str,
    max_pages_per_pdf: int,
) -> Dict[str, Any]:
    pdf_dir = dataset_root / "pdf"
    pdf_files = sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])

    samples: List[Dict[str, Any]] = []
    companies_seen: List[str] = []
    for pdf in pdf_files:
        report_id = pdf.stem
        company = _company_from_report_id(report_id)
        companies_seen.append(company)

        try:
            doc = fitz.open(pdf)
            page_count = len(doc)
            doc.close()
        except Exception:
            continue

        effective_pages = page_count
        if max_pages_per_pdf > 0:
            effective_pages = min(page_count, int(max_pages_per_pdf))

        for page_idx in range(1, effective_pages + 1):
            sample_id = f"{report_id}_p{page_idx:03d}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "split": "dev",  # patched below based on split mode
                    "company": company,
                    "report_id": report_id,
                    "page_index": int(page_idx),
                    "page_image_path": f"images/{sample_id}.png",
                    "gt_markdown_path": f"gt_markdown/{sample_id}.md",
                    "gt_structured_path": f"gt_structured/{sample_id}.json",
                    "gt_table_cells_path": f"gt_cells/{sample_id}.json",
                    "source_pdf_path": f"pdf/{pdf.name}",
                    "annotator_id": (annotator_id or "").strip(),
                    "annotation_passes": 2,
                }
            )

    split_map = _assign_company_splits(companies_seen, split_mode=split_mode)
    for sample in samples:
        company = str(sample.get("company", "")).upper()
        sample["split"] = split_map.get(company, "dev")
        if not sample.get("annotator_id"):
            sample.pop("annotator_id", None)

    out = _make_empty_manifest()
    out["samples"] = samples
    return out


def _write_manifest(dataset_root: Path, manifest: Dict[str, Any]) -> Path:
    _ensure_dataset_dirs(dataset_root)
    path = dataset_root / "manifest.json"
    if path.exists():
        backup_name = f"manifest.backup.{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        backup_path = path.parent / backup_name
        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _read_manifest(dataset_root: Path) -> Dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Manifest root must be a JSON object")
    return data


def _append_exclusion_log(dataset_root: Path, sample_id: str, reason: str) -> Path:
    log_path = dataset_root / "excluded_samples.json"
    payload = []
    if log_path.exists():
        try:
            obj = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                payload = obj
        except Exception:
            payload = []
    payload.append(
        {
            "sample_id": sample_id,
            "reason": reason or "non_table_page",
            "excluded_at": _now_iso(),
        }
    )
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return log_path


def _exclude_sample_from_manifest(dataset_root: Path, sample_id: str, reason: str) -> tuple[int, int]:
    manifest = _read_manifest(dataset_root)
    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("Manifest 'samples' must be a list")

    before = len(samples)
    kept = []
    removed = 0
    for s in samples:
        if isinstance(s, dict) and str(s.get("sample_id", "")) == sample_id:
            removed += 1
            continue
        kept.append(s)

    if removed == 0:
        raise KeyError(f"sample_id not found in manifest: {sample_id}")

    manifest["samples"] = kept
    _write_manifest(dataset_root, manifest)
    _append_exclusion_log(dataset_root, sample_id, reason)
    return before, len(kept)


def _include_registry_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root).resolve() / "included_samples.json"


def _manifest_sample_id_order(dataset_root: str | Path) -> List[str]:
    try:
        manifest = _read_manifest(Path(dataset_root).resolve())
    except Exception:
        return []
    sample_rows = manifest.get("samples", [])
    if not isinstance(sample_rows, list):
        return []
    out: List[str] = []
    for row in sample_rows:
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            out.append(sample_id)
    return out


def _load_include_registry(dataset_root: str | Path) -> Dict[str, Any]:
    path = _include_registry_path(dataset_root)
    default = {
        "version": "1.0.0",
        "mode": "include_table_pages",
        "included_sample_ids": [],
        "updated_at": "",
    }
    if not path.exists():
        return default

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    if not isinstance(obj, dict):
        return default

    ids_raw = obj.get("included_sample_ids", [])
    ids: List[str] = []
    seen: set[str] = set()
    if isinstance(ids_raw, list):
        for x in ids_raw:
            sid = str(x).strip()
            if sid and sid not in seen:
                seen.add(sid)
                ids.append(sid)
    return {
        "version": str(obj.get("version", "1.0.0")),
        "mode": str(obj.get("mode", "include_table_pages")),
        "included_sample_ids": ids,
        "updated_at": str(obj.get("updated_at", "")),
    }


def _save_include_registry(dataset_root: str | Path, registry: Dict[str, Any]) -> Path:
    path = _include_registry_path(dataset_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_ids = registry.get("included_sample_ids", [])
    deduped: List[str] = []
    seen: set[str] = set()
    if isinstance(raw_ids, list):
        for x in raw_ids:
            sid = str(x).strip()
            if sid and sid not in seen:
                seen.add(sid)
                deduped.append(sid)

    manifest_order = _manifest_sample_id_order(dataset_root)
    if manifest_order:
        include_set = set(deduped)
        ordered_known = [sid for sid in manifest_order if sid in include_set]
        ordered_unknown = sorted(sid for sid in include_set if sid not in set(manifest_order))
        deduped = ordered_known + ordered_unknown
    else:
        deduped = sorted(deduped)

    payload = {
        "version": "1.0.0",
        "mode": "include_table_pages",
        "included_sample_ids": deduped,
        "updated_at": _now_iso(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _is_included(sample_id: str, dataset_root: str | Path) -> bool:
    reg = _load_include_registry(dataset_root)
    return sample_id in set(reg.get("included_sample_ids", []))


def _set_included(sample_id: str, dataset_root: str | Path, included: bool) -> Dict[str, Any]:
    reg = _load_include_registry(dataset_root)
    include_set = set(reg.get("included_sample_ids", []))
    if included:
        include_set.add(sample_id)
    else:
        include_set.discard(sample_id)
    reg["included_sample_ids"] = sorted(include_set)
    _save_include_registry(dataset_root, reg)
    return _load_include_registry(dataset_root)


def _filter_by_include_scope(
    samples: List[TableSample], dataset_root: str | Path, include_scope: str
) -> List[TableSample]:
    if include_scope == "all":
        return list(samples)
    included_ids = set(_load_include_registry(dataset_root).get("included_sample_ids", []))
    if include_scope == "included":
        return [s for s in samples if s.sample_id in included_ids]
    # default behavior for invalid values: not_included
    return [s for s in samples if s.sample_id not in included_ids]


def _build_included_manifest(dataset_root: str | Path) -> Tuple[Dict[str, Any], Dict[str, int], List[str]]:
    manifest = _read_manifest(Path(dataset_root).resolve())
    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("Manifest 'samples' must be a list")

    include_ids = list(_load_include_registry(dataset_root).get("included_sample_ids", []))
    include_set = set(include_ids)
    if not include_set:
        raise ValueError("Include list is empty. Add at least one sample before finalizing.")

    kept: List[Dict[str, Any]] = []
    found_ids: set[str] = set()
    for row in samples:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("sample_id", "")).strip()
        if sid and sid in include_set:
            kept.append(row)
            found_ids.add(sid)

    missing = sorted(sid for sid in include_set if sid not in found_ids)
    included_manifest = dict(manifest)
    included_manifest["samples"] = kept
    counts = {
        "original_count": len(samples),
        "included_requested": len(include_set),
        "included_found": len(kept),
    }
    return included_manifest, counts, missing


def _render_manifest_setup(dataset_root: str) -> None:
    root = Path(dataset_root).resolve()
    st.warning(f"Manifest not found at `{root / 'manifest.json'}`")
    st.markdown(
        "A **manifest.json** lists your annotation samples (which PDF, which page, output paths). "
        "You can create it here."
    )

    split_mode = st.selectbox(
        "Split mode for auto-generation",
        options=["company_holdout_80_20", "all_dev"],
        index=0,
        help="company_holdout_80_20 assigns whole companies to dev/test deterministically.",
    )
    max_pages_per_pdf = int(
        st.number_input(
            "Max pages per PDF (0 = all pages)",
            min_value=0,
            value=0,
            step=1,
        )
    )
    annotator_id = st.text_input("Annotator ID (optional)", value="")

    c1, c2 = st.columns(2)
    if c1.button("Create Empty Manifest", type="secondary"):
        try:
            p = _write_manifest(root, _make_empty_manifest())
            st.success(f"Created `{p}`. Now add samples manually or use auto-generate.")
        except Exception as e:
            st.error(f"Failed to create manifest: {e}")

    if c2.button("Auto-Generate Manifest from pdf/*.pdf", type="primary"):
        try:
            manifest = _build_manifest_from_pdfs(
                dataset_root=root,
                split_mode=split_mode,
                annotator_id=annotator_id,
                max_pages_per_pdf=max_pages_per_pdf,
            )
            p = _write_manifest(root, manifest)
            st.success(
                f"Created `{p}` with {len(manifest.get('samples', []))} sample(s). "
                "If `pdf/` is empty, add PDFs and click again."
            )
            st.info("Click `Rerun` in Streamlit (or press `r`) to load the dataset.")
        except Exception as e:
            st.error(f"Auto-generation failed: {e}")


def _render_dataset_ops(dataset_root: str) -> None:
    root = Path(dataset_root).resolve()
    with st.expander("Dataset Ops (GUI pipeline)", expanded=False):
        st.caption("Use this panel to regenerate manifest from PDFs and render page images.")

        c1, c2, c3 = st.columns(3)
        split_mode = c1.selectbox(
            "Rebuild split mode",
            options=["company_holdout_80_20", "all_dev"],
            index=0,
            key="ops_split_mode",
        )
        max_pages_per_pdf = int(
            c2.number_input(
                "Rebuild max pages/PDF (0=all)",
                min_value=0,
                value=0,
                step=1,
                key="ops_max_pages",
            )
        )
        annotator_id = c3.text_input("Rebuild annotator_id", value="", key="ops_annotator")

        if st.button("Rebuild Manifest from pdf/*.pdf (replace)", key="ops_rebuild_manifest"):
            try:
                manifest = _build_manifest_from_pdfs(
                    dataset_root=root,
                    split_mode=split_mode,
                    annotator_id=annotator_id,
                    max_pages_per_pdf=max_pages_per_pdf,
                )
                p = _write_manifest(root, manifest)
                st.success(f"Rebuilt `{p}` with {len(manifest.get('samples', []))} sample(s).")
                st.rerun()
            except Exception as e:
                st.error(f"Rebuild failed: {e}")

        d1, d2, d3 = st.columns(3)
        render_split = d1.selectbox(
            "Render split",
            options=["all", "dev", "test"],
            index=0,
            key="ops_render_split",
        )
        dpi = int(
            d2.number_input(
                "Render DPI",
                min_value=72,
                max_value=600,
                value=200,
                step=10,
                key="ops_render_dpi",
            )
        )
        skip_existing = bool(d3.checkbox("Skip existing images", value=True, key="ops_render_skip"))
        if st.button("Render Images from Manifest", key="ops_render_images"):
            try:
                counts = render_page_images(
                    dataset_root=root,
                    split=render_split,  # type: ignore[arg-type]
                    dpi=dpi,
                    skip_existing=skip_existing,
                )
                st.success(
                    "Render done: "
                    f"total={counts['total']} success={counts['success']} "
                    f"failed={counts['failed']} skipped={counts['skipped']}"
                )
            except Exception as e:
                st.error(f"Render failed: {e}")

        e1, e2 = st.columns(2)
        assemble_split = e1.selectbox(
            "Assemble GT structured split",
            options=["all", "dev", "test"],
            index=0,
            key="ops_assemble_struct_split",
        )
        if e2.button("Build GT Report Structured Files", key="ops_assemble_structured"):
            try:
                counts = build_gt_structured_report_files(
                    dataset_root=root,
                    split=assemble_split,  # type: ignore[arg-type]
                    output_dir="gt_structured_report",
                    meta_dir="gt_structured_report_meta",
                )
                st.success(
                    "Built report-level GT structured files: "
                    f"total={counts['reports_total']} saved={counts['reports_saved']} "
                    f"failed={counts['reports_failed']}"
                )
            except Exception as e:
                st.error(f"Assemble failed: {e}")

        st.divider()
        include_registry = _load_include_registry(root)
        include_count = len(include_registry.get("included_sample_ids", []))
        st.caption(f"Include registry: {include_count} sample(s) marked as table pages.")

        replace_manifest = st.checkbox(
            "Replace active manifest.json with included-only manifest (backup first)",
            value=False,
            key="ops_replace_manifest_with_included",
        )
        if st.button("Build manifest.included.json from included samples", key="ops_build_included"):
            try:
                included_manifest, counts, missing = _build_included_manifest(root)
                included_path = root / "manifest.included.json"
                included_path.write_text(
                    json.dumps(included_manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if missing:
                    st.warning(
                        "Some included sample IDs were not found in manifest and were ignored: "
                        + ", ".join(missing[:20])
                        + (" ..." if len(missing) > 20 else "")
                    )
                if replace_manifest:
                    _write_manifest(root, included_manifest)
                st.success(
                    f"Included manifest built at `{included_path}`. "
                    f"original={counts['original_count']} included_found={counts['included_found']} "
                    f"included_requested={counts['included_requested']} "
                    f"{'(manifest.json replaced)' if replace_manifest else ''}"
                )
            except Exception as e:
                st.error(f"Build included manifest failed: {e}")


def get_sample_status(sample: TableSample, dataset_root: str | Path) -> str:
    meta = load_meta(sample.sample_id, dataset_root)
    if sample.audited_by or bool(meta.get("audited", False)):
        return "audited"
    if bool(meta.get("pass2_done", False)):
        return "pass2_done"
    if bool(meta.get("pass1_done", False)):
        return "pass1_done"
    return "not_started"


def filter_samples(
    *,
    samples: List[TableSample],
    dataset_root: str | Path,
    split: str,
    companies: List[str],
    statuses: List[str],
) -> List[TableSample]:
    out: List[TableSample] = []
    selected_companies = {c.upper() for c in companies}
    selected_statuses = set(statuses)
    for s in samples:
        if split != "all" and s.split != split:
            continue
        if selected_companies and s.company.upper() not in selected_companies:
            continue
        stv = get_sample_status(s, dataset_root)
        if selected_statuses and stv not in selected_statuses:
            continue
        out.append(s)
    return out


def _resolve_selected_sample(sample_id_list: List[str], selected_sample_id: Optional[str]) -> Tuple[str, int]:
    if not sample_id_list:
        raise ValueError("sample_id_list is empty")
    if selected_sample_id and selected_sample_id in sample_id_list:
        idx = sample_id_list.index(selected_sample_id)
        return selected_sample_id, idx
    return sample_id_list[0], 0


def _shift_sample_index(current_index: int, total: int, delta: int) -> int:
    if total <= 0:
        return 0
    return max(0, min(total - 1, int(current_index) + int(delta)))


def _render_page_image_if_missing(sample: TableSample, dataset_root: str | Path, dpi: int = 200) -> Optional[Path]:
    ds_root = Path(dataset_root)
    page_path = (ds_root / sample.page_image_path).resolve()
    if page_path.exists():
        return page_path

    if not sample.source_pdf_path:
        return None
    pdf_path = (ds_root / sample.source_pdf_path).resolve()
    if not pdf_path.exists():
        return None

    if sample.page_index < 1:
        return None

    doc = fitz.open(pdf_path)
    try:
        page_zero = sample.page_index - 1
        if page_zero >= len(doc):
            return None
        pix = doc[page_zero].get_pixmap(dpi=int(dpi), alpha=False)
        page_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(page_path)
    finally:
        doc.close()
    return page_path if page_path.exists() else None


def _load_dataset(dataset_root: str) -> BenchmarkDatasetV2:
    ds = BenchmarkDatasetV2(dataset_root)
    # App should allow early-stage datasets (e.g., only dev split so far).
    _ = ds.manifest
    _ = ds.samples
    return ds


def _default_rows_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ROWS_COLUMNS))


def _normalize_csv_df(df: pd.DataFrame, required_columns: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for c in required_columns:
        if c not in out.columns:
            out[c] = ""
    return out[list(required_columns)].fillna("")


def _read_csv_text_with_columns(
    csv_text: str,
    *,
    required_columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    text = (csv_text or "").strip()
    if not text:
        return pd.DataFrame(columns=list(required_columns))
    if required_columns == CELLS_COLUMNS:
        # cells.csv often contains pasted text with unquoted commas in `text`.
        return _read_cells_csv_relaxed(text)
    try:
        df = pd.read_csv(StringIO(text), dtype=str)
    except Exception as e:
        raise ValueError(f"Invalid CSV for {label}: {e}") from e
    return _normalize_csv_df(df, required_columns)


def _read_csv_path_with_columns(
    path: str | Path,
    *,
    required_columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return _read_csv_text_with_columns(
        p.read_text(encoding="utf-8-sig"),
        required_columns=required_columns,
        label=label,
    )


def _empty_csv_df(required_columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(required_columns))


def _df_to_csv_text(df: pd.DataFrame, required_columns: tuple[str, ...]) -> str:
    norm = _normalize_csv_df(df, required_columns)
    return norm.to_csv(index=False)


def _read_cells_csv_relaxed(csv_text: str) -> pd.DataFrame:
    reader = csv.reader(StringIO(csv_text))
    header = next(reader, None)
    if header is None:
        return _empty_csv_df(CELLS_COLUMNS)
    header = [str(h).strip().lstrip("\ufeff") for h in header]
    if len(header) < 2 or header[0] != "row_idx" or header[1] != "col_idx":
        raise ValueError("cells.csv header must start with: row_idx,col_idx,text")

    rows: list[dict[str, str]] = []
    for i, rec in enumerate(reader, start=2):
        if not rec or all(str(x).strip() == "" for x in rec):
            continue
        if len(rec) < 2:
            raise ValueError(f"cells.csv line {i} must include row_idx,col_idx,text")
        row_idx = str(rec[0]).strip()
        col_idx = str(rec[1]).strip()
        text = ",".join(str(x) for x in rec[2:]).strip() if len(rec) > 2 else ""
        rows.append({"row_idx": row_idx, "col_idx": col_idx, "text": text})
    return pd.DataFrame(rows, columns=list(CELLS_COLUMNS)).fillna("")


def _model_prompt_for_image_to_csv(sample_id: str, image_hint: str) -> str:
    return (
        "You are a careful data annotator extracting a table from a single page image.\n\n"
        "Task:\n"
        "- Read ONLY the table on the image.\n"
        "- Ignore all text outside the table.\n"
        "- Return 2 CSV files: cells.csv, rows.csv.\n"
        "- Keep Vietnamese text exactly as shown.\n\n"
        "Sample context:\n"
        f"- sample_id: {sample_id}\n"
        f"- image_hint: {image_hint or '(not provided)'}\n\n"
        "Schema requirements:\n"
        "1) cells.csv columns (required): row_idx,col_idx,text\n"
        "- row_idx and col_idx are 0-based integers.\n"
        "- One record per visible table cell anchor.\n"
        "- Keep reading order top-to-bottom, left-to-right.\n\n"
        "2) rows.csv columns (required): statement,item_code,item_name,value,notes_ref,original_name,row_identity,column_label,period_key\n"
        "- statement must be one of: balance_sheet, income_statement, cash_flow.\n"
        "- Add one row per financial line item that has a numeric value.\n"
        "- Keep item_name as printed.\n"
        "- value is numeric text in canonical VND using dot decimal, no thousands separators.\n"
        "- If page unit is triệu/tỷ/nghìn, convert the value to VND before writing rows.csv.\n"
        "- Parentheses negative numbers become negative (e.g., (123) -> -123).\n"
        "- row_identity is optional but recommended for repeated labels across pages/sections.\n"
        "- column_label and period_key are optional but recommended when a page has multiple value columns.\n"
        "- If item_code / notes_ref / original_name missing, keep empty.\n\n"
        "Output format:\n"
        "- Return exactly two fenced CSV blocks with these labels in this order:\n"
        "```csv cells.csv\n"
        "row_idx,col_idx,text\n"
        "...\n"
        "```\n"
        "```csv rows.csv\n"
        "statement,item_code,item_name,value,notes_ref,original_name,row_identity,column_label,period_key\n"
        "...\n"
        "```\n"
        "- Do not output explanations."
    )


def _render_sample_header(sample: TableSample, *, included: bool) -> None:
    include_state = "included" if included else "not_included"
    st.markdown(
        f"**Sample** `{sample.sample_id}` | split: `{sample.split}` | company: `{sample.company}` | "
        f"page: `{sample.page_index}` | include: `{include_state}`"
    )


def _render_validation(errors: List[str]) -> None:
    st.subheader("Validation")
    if not errors:
        st.success("CSV pack is valid.")
        return
    st.error(f"Found {len(errors)} validation issue(s).")
    for e in errors:
        st.write(f"- {e}")


def main() -> None:
    st.set_page_config(page_title="Benchmark v2 Annotation", page_icon="🧾", layout="wide")
    st.title("Benchmark v2 Annotation")
    st.caption("CSV-first editor for canonical generation: gt_markdown + gt_structured + optional gt_cells.")

    dataset_root = st.text_input(
        "Dataset root",
        value=st.session_state.get("annotation_dataset_root", "data/benchmark_v2"),
    )
    st.session_state["annotation_dataset_root"] = dataset_root

    manifest_path = Path(dataset_root).resolve() / "manifest.json"
    if not manifest_path.exists():
        _render_manifest_setup(dataset_root)
        return

    try:
        ds = _load_dataset(dataset_root)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return

    _render_dataset_ops(dataset_root)

    auto_validate = st.checkbox(
        "Auto-validate while typing (can be slower)",
        value=False,
        help="Disable this for smoother editing. You can validate manually before saving.",
    )

    companies = sorted({s.company for s in ds.samples})
    c1, c2, c3 = st.columns(3)
    split_filter = c1.selectbox("Split", options=["all", "dev", "test"], index=0)
    company_filter = c2.multiselect("Companies", options=companies, default=companies)
    include_scope = c3.selectbox(
        "Include Scope",
        options=list(INCLUDE_SCOPE_OPTIONS),
        index=0,
        help="not_included shows remaining pages to triage (default).",
    )

    filtered = filter_samples(
        samples=ds.samples,
        dataset_root=dataset_root,
        split=split_filter,
        companies=company_filter,
        statuses=["not_started", "pass1_done", "pass2_done", "audited"],
    )
    filtered = _filter_by_include_scope(filtered, dataset_root, include_scope)
    if not filtered:
        st.warning("No samples match current filters.")
        return

    sample_id_list = [s.sample_id for s in filtered]
    previous_selected = st.session_state.get(SESSION_SELECTED_SAMPLE_ID)
    resolved_id, resolved_idx = _resolve_selected_sample(sample_id_list, previous_selected)
    st.session_state[SESSION_SELECTED_SAMPLE_ID] = resolved_id
    st.session_state[SESSION_SELECTED_SAMPLE_INDEX] = resolved_idx

    nav_prev, nav_select, nav_next = st.columns([1, 4, 1])
    if nav_prev.button("Prev", disabled=resolved_idx == 0, key="sample_prev_btn"):
        new_idx = _shift_sample_index(resolved_idx, len(sample_id_list), -1)
        st.session_state[SESSION_SELECTED_SAMPLE_INDEX] = new_idx
        st.session_state[SESSION_SELECTED_SAMPLE_ID] = sample_id_list[new_idx]
        st.rerun()
    if nav_next.button(
        "Next",
        disabled=resolved_idx >= (len(sample_id_list) - 1),
        key="sample_next_btn",
    ):
        new_idx = _shift_sample_index(resolved_idx, len(sample_id_list), 1)
        st.session_state[SESSION_SELECTED_SAMPLE_INDEX] = new_idx
        st.session_state[SESSION_SELECTED_SAMPLE_ID] = sample_id_list[new_idx]
        st.rerun()

    selected_sample_id = nav_select.selectbox("Sample", options=sample_id_list, index=resolved_idx)
    if selected_sample_id != st.session_state.get(SESSION_SELECTED_SAMPLE_ID):
        st.session_state[SESSION_SELECTED_SAMPLE_ID] = selected_sample_id
        st.session_state[SESSION_SELECTED_SAMPLE_INDEX] = sample_id_list.index(selected_sample_id)

    sample = next(s for s in filtered if s.sample_id == selected_sample_id)
    sample_included = _is_included(sample.sample_id, dataset_root)

    _render_sample_header(sample, included=sample_included)

    draft_cells_key = f"draft_cells_{sample.sample_id}"
    draft_rows_key = f"draft_rows_{sample.sample_id}"
    editor_nonce_key = f"editor_nonce_{sample.sample_id}"
    validation_state_key = f"validation_errors_{sample.sample_id}"
    eye_md_key = f"eye_markdown_{sample.sample_id}"

    if draft_cells_key not in st.session_state:
        initial_pack = load_csv_pack(sample.sample_id, dataset_root)
        st.session_state[draft_cells_key] = initial_pack["cells"]
        st.session_state[draft_rows_key] = initial_pack["rows"]
        st.session_state[editor_nonce_key] = 0
    if eye_md_key not in st.session_state:
        md_path = (Path(dataset_root).resolve() / sample.gt_markdown_path).resolve()
        if md_path.exists():
            try:
                st.session_state[eye_md_key] = md_path.read_text(encoding="utf-8")
            except Exception:
                st.session_state[eye_md_key] = ""
        else:
            st.session_state[eye_md_key] = ""

    def _set_drafts(*, cells: pd.DataFrame, rows: pd.DataFrame) -> None:
        st.session_state[draft_cells_key] = _normalize_csv_df(cells, CELLS_COLUMNS)
        st.session_state[draft_rows_key] = _normalize_csv_df(rows, ROWS_COLUMNS)
        st.session_state[editor_nonce_key] = int(st.session_state.get(editor_nonce_key, 0)) + 1
        st.session_state[validation_state_key] = []

    toolbar = st.columns(4)
    if toolbar[0].button("Reload CSV from Disk"):
        try:
            pack = load_csv_pack(sample.sample_id, dataset_root)
            _set_drafts(cells=pack["cells"], rows=pack["rows"])
            st.success("Reloaded CSV pack from disk.")
            st.rerun()
        except Exception as e:
            st.error(f"Reload failed: {e}")

    if toolbar[1].button("Import Canonical -> CSV"):
        try:
            info = canonical_to_csv(sample.sample_id, dataset_root)
            pack = load_csv_pack(sample.sample_id, dataset_root)
            _set_drafts(cells=pack["cells"], rows=pack["rows"])
            st.success(f"Imported canonical files to CSV pack ({info['csv_root']}).")
            st.rerun()
        except Exception as e:
            st.error(f"Import failed: {e}")
    if toolbar[2].button("Include Current Sample (Table)", disabled=sample_included):
        try:
            _set_included(sample.sample_id, dataset_root, True)
            st.success(f"Marked `{sample.sample_id}` as included.")
            st.rerun()
        except Exception as e:
            st.error(f"Include failed: {e}")
    if toolbar[3].button("Remove From Included", disabled=not sample_included):
        try:
            _set_included(sample.sample_id, dataset_root, False)
            st.success(f"Removed `{sample.sample_id}` from included list.")
            st.rerun()
        except Exception as e:
            st.error(f"Uninclude failed: {e}")

    image_col, data_col = st.columns([1, 1])
    with image_col:
        st.subheader("Page Image")
        img_path = _render_page_image_if_missing(sample, dataset_root, dpi=200)
        if img_path and img_path.exists():
            st.image(str(img_path), width='stretch')
        else:
            st.info("No image available and could not auto-render from source PDF.")

    with data_col:
        st.subheader("CSV Editors")
        with st.expander("Direct CSV Import", expanded=False):
            st.caption("Paste raw CSV text directly into editors (no canonical conversion).")
            paste_cells_key = f"paste_cells_text_{sample.sample_id}"
            paste_rows_key = f"paste_rows_text_{sample.sample_id}"
            if paste_cells_key not in st.session_state:
                st.session_state[paste_cells_key] = _df_to_csv_text(
                    st.session_state[draft_cells_key], CELLS_COLUMNS
                )
            if paste_rows_key not in st.session_state:
                st.session_state[paste_rows_key] = _df_to_csv_text(
                    st.session_state[draft_rows_key], ROWS_COLUMNS
                )

            p1, p2 = st.columns(2)
            if p1.button("Load Current Editors into Paste Boxes", key=f"load_current_csv_{sample.sample_id}"):
                st.session_state[paste_cells_key] = _df_to_csv_text(
                    st.session_state[draft_cells_key], CELLS_COLUMNS
                )
                st.session_state[paste_rows_key] = _df_to_csv_text(
                    st.session_state[draft_rows_key], ROWS_COLUMNS
                )
                st.success("Loaded current editor data into paste boxes.")
            if p2.button("Clear Paste Boxes", key=f"clear_paste_csv_{sample.sample_id}"):
                st.session_state[paste_cells_key] = ""
                st.session_state[paste_rows_key] = ""
                st.success("Cleared paste boxes.")

            paste_tabs = st.tabs(["cells.csv text", "rows.csv text"])
            with paste_tabs[0]:
                pasted_cells = st.text_area(
                    "Paste cells.csv content",
                    key=paste_cells_key,
                    height=180,
                )
            with paste_tabs[1]:
                pasted_rows = st.text_area(
                    "Paste rows.csv content",
                    key=paste_rows_key,
                    height=180,
                )

            if st.button("Apply Pasted CSV(s)", key=f"apply_paste_csv_{sample.sample_id}"):
                try:
                    base_cells = st.session_state[draft_cells_key]
                    base_rows = st.session_state[draft_rows_key]
                    new_cells = base_cells
                    new_rows = base_rows
                    imported = 0

                    if pasted_cells.strip():
                        new_cells = _read_csv_text_with_columns(
                            pasted_cells,
                            required_columns=CELLS_COLUMNS,
                            label="cells.csv",
                        )
                        imported += 1
                    if pasted_rows.strip():
                        new_rows = _read_csv_text_with_columns(
                            pasted_rows,
                            required_columns=ROWS_COLUMNS,
                            label="rows.csv",
                        )
                        imported += 1

                    if imported == 0:
                        st.warning("Paste at least one CSV content block to import.")
                    else:
                        _set_drafts(cells=new_cells, rows=new_rows)
                        st.success(f"Imported {imported} CSV block(s) into editor.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Pasted CSV import failed: {e}")

            import_dir = st.text_input(
                "Or import from folder path containing cells.csv/rows.csv",
                value="",
                key=f"csv_import_dir_{sample.sample_id}",
            )
            if st.button("Import from Folder", key=f"import_csv_dir_{sample.sample_id}"):
                try:
                    if not import_dir.strip():
                        st.warning("Please enter a folder path.")
                    else:
                        p = Path(import_dir).expanduser().resolve()
                        if not p.exists() or not p.is_dir():
                            raise FileNotFoundError(f"Folder not found: {p}")
                        cells_path = p / "cells.csv"
                        rows_path = p / "rows.csv"

                        current_cells = st.session_state[draft_cells_key]
                        current_rows = st.session_state[draft_rows_key]
                        imported = 0

                        if cells_path.exists():
                            new_cells = _read_csv_path_with_columns(
                                cells_path,
                                required_columns=CELLS_COLUMNS,
                                label="cells.csv",
                            )
                            imported += 1
                        else:
                            new_cells = current_cells

                        if rows_path.exists():
                            new_rows = _read_csv_path_with_columns(
                                rows_path,
                                required_columns=ROWS_COLUMNS,
                                label="rows.csv",
                            )
                            imported += 1
                        else:
                            new_rows = current_rows

                        if imported == 0:
                            st.warning(
                                f"No CSV files found in `{p}`. Expected cells.csv/rows.csv."
                            )
                            return

                        _set_drafts(cells=new_cells, rows=new_rows)
                        st.success(f"Imported {imported} CSV file(s) from `{p}`.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Folder import failed: {e}")

        editor_nonce = int(st.session_state.get(editor_nonce_key, 0))
        tabs = st.tabs(["cells.csv", "rows.csv"])
        with tabs[0]:
            cells_df = st.data_editor(
                st.session_state[draft_cells_key],
                width='stretch',
                num_rows="dynamic",
                key=f"cells_editor_{sample.sample_id}_{editor_nonce}",
            )
        with tabs[1]:
            rows_df = st.data_editor(
                (
                    st.session_state[draft_rows_key]
                    if not st.session_state[draft_rows_key].empty
                    else _default_rows_df()
                ),
                width='stretch',
                num_rows="dynamic",
                key=f"rows_editor_{sample.sample_id}_{editor_nonce}",
            )
            st.caption(f"`statement` must be one of: {', '.join(STATEMENTS)}")

        st.session_state[draft_cells_key] = _normalize_csv_df(cells_df, CELLS_COLUMNS)
        st.session_state[draft_rows_key] = _normalize_csv_df(rows_df, ROWS_COLUMNS)

        if auto_validate:
            errors = validate_csv_frames(
                cells=st.session_state[draft_cells_key],
                rows=st.session_state[draft_rows_key],
            )
            st.session_state[validation_state_key] = errors
        else:
            errors = st.session_state.get(validation_state_key, [])

        actions = st.columns(3)
        do_validate = actions[0].button("Validate Current Edits", type="secondary")
        do_preview = actions[1].button("Generate Canonical Preview", type="secondary")
        do_save = actions[2].button("Save CSV + Canonical", type="primary")

        if do_validate:
            errors = validate_csv_frames(
                cells=st.session_state[draft_cells_key],
                rows=st.session_state[draft_rows_key],
            )
            st.session_state[validation_state_key] = errors

        _render_validation(errors)

        if do_preview:
            preview_errors = validate_csv_frames(
                cells=st.session_state[draft_cells_key],
                rows=st.session_state[draft_rows_key],
            )
            st.session_state[validation_state_key] = preview_errors
            if preview_errors:
                st.error("Please fix validation errors first.")
            else:
                try:
                    preview = build_canonical_from_frames(
                        cells=st.session_state[draft_cells_key],
                        rows=st.session_state[draft_rows_key],
                        validate=True,
                    )
                    st.session_state[eye_md_key] = preview["gt_markdown"]
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.markdown("**gt_markdown preview (raw)**")
                        st.code(preview["gt_markdown"], language="markdown")
                    with p2:
                        st.markdown("**gt_structured preview**")
                        st.json(preview["gt_structured"])
                    with p3:
                        st.markdown("**gt_cells preview**")
                        st.json(preview["gt_cells"])
                except Exception as e:
                    st.error(f"Preview generation failed: {e}")

        if do_save:
            save_errors = validate_csv_frames(
                cells=st.session_state[draft_cells_key],
                rows=st.session_state[draft_rows_key],
            )
            st.session_state[validation_state_key] = save_errors
            if save_errors:
                st.error("Cannot save. Please fix validation errors first.")
            else:
                try:
                    save_csv_pack(
                        sample.sample_id,
                        dataset_root,
                        cells=st.session_state[draft_cells_key],
                        rows=st.session_state[draft_rows_key],
                        meta_updates={"last_csv_save_at": _now_iso()},
                    )
                    out = csv_to_canonical(sample.sample_id, dataset_root, validate=True)
                    st.success(
                        "Saved CSV and canonical artifacts:\n"
                        f"- {out['gt_markdown_path']}\n"
                        f"- {out['gt_structured_path']}\n"
                        f"- {out['gt_table_cells_path']}"
                    )
                    md_text = Path(out["gt_markdown_path"]).read_text(encoding="utf-8")
                    st.session_state[eye_md_key] = md_text
                except Exception as e:
                    st.error(f"Save failed: {e}")

    with st.expander("Image vs Rendered Markdown (Eye-test)", expanded=False):
        compare_l, compare_r = st.columns(2)
        with compare_l:
            st.markdown("**Page Image**")
            if img_path and img_path.exists():
                st.image(str(img_path), width='stretch')
            else:
                st.info("No page image available.")
        with compare_r:
            refresh_eye = st.button(
                "Refresh From Current CSV",
                key=f"refresh_eye_markdown_{sample.sample_id}",
            )
            if refresh_eye:
                refresh_errors = validate_csv_frames(
                    cells=st.session_state[draft_cells_key],
                    rows=st.session_state[draft_rows_key],
                )
                st.session_state[validation_state_key] = refresh_errors
                if refresh_errors:
                    st.error("Cannot refresh eye-test due to validation errors.")
                else:
                    try:
                        refreshed = build_canonical_from_frames(
                            cells=st.session_state[draft_cells_key],
                            rows=st.session_state[draft_rows_key],
                            validate=True,
                        )
                        st.session_state[eye_md_key] = refreshed["gt_markdown"]
                        st.success("Rendered markdown refreshed.")
                    except Exception as e:
                        st.error(f"Refresh failed: {e}")

            eye_md = str(st.session_state.get(eye_md_key, "") or "")
            if eye_md.strip():
                st.markdown(eye_md)
            else:
                st.info("No rendered markdown yet. Use Preview/Save or refresh from current CSV.")

if __name__ == "__main__":
    main()
