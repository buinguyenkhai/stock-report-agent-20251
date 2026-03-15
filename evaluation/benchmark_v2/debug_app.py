"""
Lightweight Streamlit viewer for benchmark v2 debug diffs.

Run:
  streamlit run evaluation/benchmark_v2/debug_app.py -- --diff-json results/benchmark_v2_debug_diffs.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--diff-json", type=str, default="results/benchmark_v2_debug_diffs.json")
    args, _unknown = parser.parse_known_args()
    return args


def _load_payload(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _table_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        rows.append(
            {
                "row_key": item.get("row_key"),
                "statement": item.get("statement"),
                "item_code": item.get("item_code"),
                "item_name": item.get("item_name"),
                "notes_ref": item.get("notes_ref"),
                "row_identity": item.get("row_identity"),
                "column_label": item.get("column_label"),
                "period_key": item.get("period_key"),
                "value": item.get("value"),
            }
        )
    return rows


def _render_sample(sample: Dict[str, Any]) -> None:
    st.subheader(sample["sample_id"])
    cols = st.columns(4)
    cols[0].metric("CER", f"{sample['raw_metrics']['table_only_cer']:.4f}" if sample.get("raw_metrics") else "NA")
    cols[1].metric("WER", f"{sample['raw_metrics']['table_only_wer']:.4f}" if sample.get("raw_metrics") else "NA")
    cols[2].metric("Cell F1", f"{sample['raw_metrics']['table_cell_f1']:.4f}" if sample.get("raw_metrics") else "NA")
    cols[3].metric("Number F1", f"{sample['raw_metrics']['number_f1']:.4f}" if sample.get("raw_metrics") else "NA")

    image_path = sample.get("page_image_path")
    if image_path and Path(image_path).exists():
        st.image(image_path, caption=sample["sample_id"], use_container_width=True)

    ocr_debug = sample.get("ocr_debug")
    if isinstance(ocr_debug, dict):
        stats = ocr_debug.get("hybrid_ocr_stats")
        if isinstance(stats, dict):
            st.markdown("**Hybrid OCR stats**")
            stat_cols = st.columns(6)
            stat_cols[0].metric("Total cells", int(stats.get("total_cells", 0) or 0))
            stat_cols[1].metric("Surya cells", int(stats.get("surya_cells", 0) or 0))
            stat_cols[2].metric("Updated", int(stats.get("surya_cells_updated", 0) or 0))
            stat_cols[3].metric("Table cells", int(stats.get("table_cells", 0) or 0))
            stat_cols[4].metric("Garbled regions", int(stats.get("garbled_regions", 0) or 0))
            stat_cols[5].metric("Sanity skips", int(stats.get("surya_update_skipped_sanity", 0) or 0))
        with st.expander("OCR debug JSON"):
            st.json(ocr_debug)

    st.markdown("**Missing numbers**")
    st.dataframe(sample.get("missing_numbers", []), use_container_width=True)
    st.markdown("**Extra numbers**")
    st.dataframe(sample.get("extra_numbers", []), use_container_width=True)
    st.markdown("**Missing cells**")
    st.dataframe(sample.get("missing_cells", []), use_container_width=True)
    st.markdown("**Extra cells**")
    st.dataframe(sample.get("extra_cells", []), use_container_width=True)

    with st.expander("Unified table diff", expanded=True):
        st.code("\n".join(sample.get("table_diff_excerpt", [])) or "(no diff excerpt)", language="diff")
    with st.expander("Ground truth markdown"):
        st.code(sample.get("gt_raw_markdown", ""), language="markdown")
    with st.expander("Prediction markdown"):
        st.code(sample.get("pred_raw_markdown", ""), language="markdown")


def _render_report(report: Dict[str, Any]) -> None:
    st.subheader(report["report_id"])
    if report.get("comparison"):
        metrics = report["comparison"]["metrics"]
        cols = st.columns(5)
        cols[0].metric("Row F1", f"{metrics['row_f1']:.4f}")
        cols[1].metric("Row P", f"{metrics['row_precision']:.4f}")
        cols[2].metric("Row R", f"{metrics['row_recall']:.4f}")
        cols[3].metric("Exact", f"{metrics['value_exact_accuracy']:.4f}")
        cols[4].metric("Tolerance", f"{metrics['value_tolerant_accuracy']:.4f}")

        st.markdown("**Value mismatches**")
        st.dataframe(report["comparison"].get("value_mismatches", []), use_container_width=True)
        st.markdown("**Missing GT rows**")
        st.dataframe(_table_rows(report["comparison"].get("missing_rows", [])), use_container_width=True)
        st.markdown("**Extra predicted rows**")
        st.dataframe(_table_rows(report["comparison"].get("extra_rows", [])), use_container_width=True)
    else:
        st.warning("Structured comparison unavailable for this report.")
        if report.get("errors"):
            st.code("\n".join(report["errors"]))

    with st.expander("GT assembly conflicts"):
        st.dataframe(report.get("gt_conflicts", []), use_container_width=True)
    with st.expander("Prediction assembly conflicts"):
        st.dataframe(report.get("pred_conflicts", []), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Benchmark v2 Debug Viewer", layout="wide")
    args = _parse_args()
    st.title("Benchmark v2 Debug Viewer")

    diff_path = st.sidebar.text_input("Diff JSON", value=args.diff_json)
    payload = _load_payload(diff_path)
    if not payload:
        st.error(f"Could not load diff JSON: {diff_path}")
        st.stop()

    mode = st.sidebar.radio("View", options=["Samples", "Reports"])
    st.sidebar.caption(f"Split: {payload.get('split')} | Include scope: {payload.get('include_scope')}")

    if mode == "Samples":
        samples = payload.get("sample_diffs", [])
        sample_ids = [row["sample_id"] for row in samples]
        selected = st.sidebar.selectbox("Sample", options=sample_ids)
        sample = next(row for row in samples if row["sample_id"] == selected)
        _render_sample(sample)
    else:
        reports = payload.get("report_diffs", [])
        report_ids = [row["report_id"] for row in reports]
        selected = st.sidebar.selectbox("Report", options=report_ids)
        report = next(row for row in reports if row["report_id"] == selected)
        _render_report(report)


if __name__ == "__main__":
    main()
