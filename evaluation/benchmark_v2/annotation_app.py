"""
Streamlit annotation app for benchmark v2 (CSV-first workflow).

Run:
  streamlit run evaluation/benchmark_v2/annotation_app.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st

try:
    from .csv_codec import (
        ROWS_COLUMNS,
        build_canonical_from_frames,
        canonical_to_csv,
        compute_pilot_metrics,
        csv_to_canonical,
        load_csv_pack,
        load_meta,
        save_csv_pack,
        update_meta,
        validate_csv_frames,
        validate_csv_pack,
    )
    from .dataset import BenchmarkDatasetV2, TableSample
    from .render_page_images import render_page_images
except ImportError:
    # Support "streamlit run evaluation/benchmark_v2/annotation_app.py"
    # where relative imports do not have package context.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from evaluation.benchmark_v2.csv_codec import (
        ROWS_COLUMNS,
        build_canonical_from_frames,
        canonical_to_csv,
        compute_pilot_metrics,
        csv_to_canonical,
        load_csv_pack,
        load_meta,
        save_csv_pack,
        update_meta,
        validate_csv_frames,
        validate_csv_pack,
    )
    from evaluation.benchmark_v2.dataset import BenchmarkDatasetV2, TableSample
    from evaluation.benchmark_v2.render_page_images import render_page_images

STATEMENTS = ("balance_sheet", "income_statement", "cash_flow")


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


def _render_sample_header(sample: TableSample) -> None:
    st.markdown(
        f"**Sample** `{sample.sample_id}` | split: `{sample.split}` | company: `{sample.company}` | "
        f"page: `{sample.page_index}`"
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
    c1, c2 = st.columns(2)
    split_filter = c1.selectbox("Split", options=["all", "dev", "test"], index=0)
    company_filter = c2.multiselect("Companies", options=companies, default=companies)

    filtered = filter_samples(
        samples=ds.samples,
        dataset_root=dataset_root,
        split=split_filter,
        companies=company_filter,
        statuses=["not_started", "pass1_done", "pass2_done", "audited"],
    )
    if not filtered:
        st.warning("No samples match current filters.")
        return

    sample_id_list = [s.sample_id for s in filtered]
    selected_sample_id = st.selectbox("Sample", options=sample_id_list)
    sample = next(s for s in filtered if s.sample_id == selected_sample_id)

    _render_sample_header(sample)

    toolbar = st.columns(2)
    if toolbar[0].button("Import Canonical -> CSV"):
        try:
            info = canonical_to_csv(sample.sample_id, dataset_root)
            st.success(f"Imported canonical files to CSV pack ({info['csv_root']}).")
        except Exception as e:
            st.error(f"Import failed: {e}")
    if toolbar[1].button("Exclude Current Sample (Non-table)"):
        try:
            before, after = _exclude_sample_from_manifest(
                Path(dataset_root).resolve(), sample.sample_id, "non_table_page"
            )
            st.success(
                f"Excluded `{sample.sample_id}` from manifest ({before} -> {after} samples)."
            )
            st.rerun()
        except Exception as e:
            st.error(f"Exclude failed: {e}")

    image_col, data_col = st.columns([1, 2])
    with image_col:
        st.subheader("Page Image")
        img_path = _render_page_image_if_missing(sample, dataset_root, dpi=200)
        if img_path and img_path.exists():
            st.image(str(img_path), width='stretch')
        else:
            st.info("No image available and could not auto-render from source PDF.")

    with data_col:
        st.subheader("CSV Editors")
        pack = load_csv_pack(sample.sample_id, dataset_root)
        tabs = st.tabs(["cells.csv", "spans.csv", "rows.csv"])
        with tabs[0]:
            cells_df = st.data_editor(
                pack["cells"],
                width='stretch',
                num_rows="dynamic",
                key=f"cells_editor_{sample.sample_id}",
            )
        with tabs[1]:
            spans_df = st.data_editor(
                pack["spans"],
                width='stretch',
                num_rows="dynamic",
                key=f"spans_editor_{sample.sample_id}",
            )
        with tabs[2]:
            rows_df = st.data_editor(
                pack["rows"] if not pack["rows"].empty else _default_rows_df(),
                width='stretch',
                num_rows="dynamic",
                key=f"rows_editor_{sample.sample_id}",
            )
            st.caption(f"`statement` must be one of: {', '.join(STATEMENTS)}")

        validation_state_key = f"validation_errors_{sample.sample_id}"
        if auto_validate:
            errors = validate_csv_frames(cells=cells_df, spans=spans_df, rows=rows_df)
            st.session_state[validation_state_key] = errors
        else:
            errors = st.session_state.get(validation_state_key, [])

        actions = st.columns(3)
        do_validate = actions[0].button("Validate Current Edits", type="secondary")
        do_preview = actions[1].button("Generate Canonical Preview", type="secondary")
        do_save = actions[2].button("Save CSV + Canonical", type="primary")

        if do_validate:
            errors = validate_csv_frames(cells=cells_df, spans=spans_df, rows=rows_df)
            st.session_state[validation_state_key] = errors

        _render_validation(errors)

        if do_preview:
            preview_errors = validate_csv_frames(cells=cells_df, spans=spans_df, rows=rows_df)
            st.session_state[validation_state_key] = preview_errors
            if preview_errors:
                st.error("Please fix validation errors first.")
            else:
                try:
                    preview = build_canonical_from_frames(
                        cells=cells_df,
                        spans=spans_df,
                        rows=rows_df,
                        validate=True,
                    )
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.markdown("**gt_markdown preview (raw)**")
                        st.code(preview["gt_markdown"], language="markdown")
                        st.markdown("**Rendered markdown (eye-test)**")
                        st.markdown(preview["gt_markdown"])
                    with p2:
                        st.markdown("**gt_structured preview**")
                        st.json(preview["gt_structured"])
                    with p3:
                        st.markdown("**gt_cells preview**")
                        st.json(preview["gt_cells"])
                except Exception as e:
                    st.error(f"Preview generation failed: {e}")

        if do_save:
            save_errors = validate_csv_frames(cells=cells_df, spans=spans_df, rows=rows_df)
            st.session_state[validation_state_key] = save_errors
            if save_errors:
                st.error("Cannot save. Please fix validation errors first.")
            else:
                try:
                    save_csv_pack(
                        sample.sample_id,
                        dataset_root,
                        cells=cells_df,
                        spans=spans_df,
                        rows=rows_df,
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
                    with st.expander("Rendered markdown (saved file eye-test)", expanded=False):
                        st.markdown(md_text)
                except Exception as e:
                    st.error(f"Save failed: {e}")

    with st.expander("Pilot Gate Metrics (20-page CSV-first acceptance)", expanded=False):
        selected_for_pilot = st.multiselect(
            "Pilot sample IDs (leave empty = all filtered samples)",
            options=sample_id_list,
            default=[],
        )
        if st.button("Compute Pilot Metrics"):
            use_ids = selected_for_pilot if selected_for_pilot else sample_id_list
            metrics = compute_pilot_metrics(dataset_root, sample_ids=use_ids, threshold=0.02)
            st.json(metrics)
            if metrics.get("pass_gate", False):
                st.success("Pilot gate PASSED.")
            else:
                st.warning("Pilot gate FAILED.")


if __name__ == "__main__":
    main()
