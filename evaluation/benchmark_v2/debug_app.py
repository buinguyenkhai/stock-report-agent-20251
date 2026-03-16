"""
Lightweight Streamlit viewer for raw OCR benchmark v2 debug diffs.

Run:
  streamlit run evaluation/benchmark_v2/debug_app.py -- --diff-json results/benchmark_v2_debug_diffs.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

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


def _render_sample(sample: Dict[str, Any]) -> None:
    st.subheader(sample["sample_id"])
    cols = st.columns(4)
    cols[0].metric("CER", f"{sample['raw_metrics']['table_only_cer']:.4f}" if sample.get("raw_metrics") else "NA")
    cols[1].metric("WER", f"{sample['raw_metrics']['table_only_wer']:.4f}" if sample.get("raw_metrics") else "NA")
    cols[2].metric("Cell F1", f"{sample['raw_metrics']['table_cell_f1']:.4f}" if sample.get("raw_metrics") else "NA")
    cols[3].metric("Number F1", f"{sample['raw_metrics']['number_f1']:.4f}" if sample.get("raw_metrics") else "NA")

    telemetry = sample.get("telemetry") or {}
    tcols = st.columns(3)
    tcols[0].metric("Latency (ms)", f"{float(telemetry['total_latency_ms']):.1f}" if telemetry.get("total_latency_ms") is not None else "NA")
    tcols[1].metric("Peak VRAM reserved (MB)", f"{float(telemetry['peak_vram_reserved_mb']):.1f}" if telemetry.get("peak_vram_reserved_mb") is not None else "NA")
    tcols[2].metric("Peak VRAM allocated (MB)", f"{float(telemetry['peak_vram_allocated_mb']):.1f}" if telemetry.get("peak_vram_allocated_mb") is not None else "NA")

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


def main() -> None:
    st.set_page_config(page_title="Benchmark v2 Debug Viewer", layout="wide")
    args = _parse_args()
    st.title("Benchmark v2 Debug Viewer")

    diff_path = st.sidebar.text_input("Diff JSON", value=args.diff_json)
    payload = _load_payload(diff_path)
    if not payload:
        st.error(f"Could not load diff JSON: {diff_path}")
        st.stop()

    st.sidebar.caption(f"Split: {payload.get('split')} | Include scope: {payload.get('include_scope')}")
    samples = payload.get("sample_diffs", [])
    sample_ids = [row["sample_id"] for row in samples]
    selected = st.sidebar.selectbox("Sample", options=sample_ids)
    sample = next(row for row in samples if row["sample_id"] == selected)
    _render_sample(sample)


if __name__ == "__main__":
    main()
