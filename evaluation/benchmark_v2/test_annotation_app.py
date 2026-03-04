import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
fitz = pytest.importorskip("fitz")

from evaluation.benchmark_v2.annotation_app import (  # noqa: E402
    _build_included_manifest,
    _model_prompt_for_image_to_csv,
    _normalize_csv_df,
    _read_csv_text_with_columns,
    _build_manifest_from_pdfs,
    _exclude_sample_from_manifest,
    _filter_by_include_scope,
    _load_include_registry,
    _render_page_image_if_missing,
    _resolve_selected_sample,
    _set_included,
    _shift_sample_index,
    filter_samples,
    get_sample_status,
)
from evaluation.benchmark_v2.csv_codec import CELLS_COLUMNS, update_meta  # noqa: E402
from evaluation.benchmark_v2.dataset import BenchmarkDatasetV2  # noqa: E402


def _write_manifest(root: Path, sample_id: str) -> None:
    manifest = {
        "version": "1.0.0",
        "split_policy": "company_heldout_dev_test",
        "annotation_protocol": "single_annotator_two_pass",
        "samples": [
            {
                "sample_id": sample_id,
                "split": "dev",
                "company": "AAA",
                "report_id": "AAA_2024Q3",
                "page_index": 1,
                "page_image_path": f"images/{sample_id}.png",
                "gt_markdown_path": f"gt_markdown/{sample_id}.md",
                "gt_structured_path": f"gt_structured/{sample_id}.json",
                "source_pdf_path": "pdf/AAA_2024Q3.pdf",
            },
            {
                "sample_id": "BBB_2024Q3_p001",
                "split": "test",
                "company": "BBB",
                "report_id": "BBB_2024Q3",
                "page_index": 1,
                "page_image_path": "images/BBB_2024Q3_p001.png",
                "gt_markdown_path": "gt_markdown/BBB_2024Q3_p001.md",
                "gt_structured_path": "gt_structured/BBB_2024Q3_p001.json",
            },
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _seed_minimal_files(dataset_root: Path, sample_id: str) -> None:
    (dataset_root / "gt_markdown").mkdir(parents=True, exist_ok=True)
    (dataset_root / "gt_structured").mkdir(parents=True, exist_ok=True)
    (dataset_root / "pdf").mkdir(parents=True, exist_ok=True)

    (dataset_root / f"gt_markdown/{sample_id}.md").write_text(
        "| Name | Value |\n| --- | --- |\n| A | 1 |\n", encoding="utf-8"
    )
    structured = {
        "balance_sheet": {"items": []},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    (dataset_root / f"gt_structured/{sample_id}.json").write_text(
        json.dumps(structured, indent=2), encoding="utf-8"
    )

    pdf_path = dataset_root / "pdf/AAA_2024Q3.pdf"
    doc = fitz.open()
    doc.new_page(width=300, height=200)
    doc.save(pdf_path)
    doc.close()


def test_annotation_app_status_filter_and_image_render(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sample_id = "AAA_2024Q3_p001"
    _write_manifest(dataset_root, sample_id)
    _seed_minimal_files(dataset_root, sample_id)

    ds = BenchmarkDatasetV2(dataset_root)
    samples = ds.samples
    s1 = next(s for s in samples if s.sample_id == sample_id)

    assert get_sample_status(s1, dataset_root) == "not_started"
    update_meta(sample_id, dataset_root, {"pass1_done": True})
    assert get_sample_status(s1, dataset_root) == "pass1_done"
    update_meta(sample_id, dataset_root, {"pass2_done": True})
    assert get_sample_status(s1, dataset_root) == "pass2_done"

    filtered = filter_samples(
        samples=samples,
        dataset_root=dataset_root,
        split="dev",
        companies=["AAA"],
        statuses=["pass2_done"],
    )
    assert [s.sample_id for s in filtered] == [sample_id]

    image_path = _render_page_image_if_missing(s1, dataset_root, dpi=120)
    assert image_path is not None
    assert image_path.exists()


def test_manifest_autogeneration_from_pdf(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "pdf").mkdir(parents=True, exist_ok=True)

    pdf_path = dataset_root / "pdf/FPT_2024Q3.pdf"
    doc = fitz.open()
    doc.new_page(width=300, height=200)
    doc.new_page(width=300, height=200)
    doc.save(pdf_path)
    doc.close()

    manifest = _build_manifest_from_pdfs(
        dataset_root=dataset_root,
        split_mode="company_holdout_80_20",
        annotator_id="khai",
        max_pages_per_pdf=0,
    )
    samples = manifest.get("samples", [])
    assert len(samples) == 2
    assert samples[0]["sample_id"] == "FPT_2024Q3_p001"
    assert samples[1]["sample_id"] == "FPT_2024Q3_p002"
    assert samples[0]["source_pdf_path"] == "pdf/FPT_2024Q3.pdf"
    assert samples[0]["annotator_id"] == "khai"


def test_exclude_sample_from_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sample_id = "AAA_2024Q3_p001"
    _write_manifest(dataset_root, sample_id)

    before, after = _exclude_sample_from_manifest(dataset_root, sample_id, "non_table_page")
    assert before == 2
    assert after == 1

    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    ids = [s["sample_id"] for s in manifest["samples"]]
    assert sample_id not in ids

    log_path = dataset_root / "excluded_samples.json"
    assert log_path.exists()
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert any(x.get("sample_id") == sample_id for x in log)


def test_read_csv_text_and_normalize_columns() -> None:
    csv_text = "row_idx,col_idx,text,extra\n0,1,Tiền,x\n1,1,100,y\n"
    df = _read_csv_text_with_columns(
        csv_text,
        required_columns=CELLS_COLUMNS,
        label="cells.csv",
    )
    assert list(df.columns) == list(CELLS_COLUMNS)
    assert len(df) == 2
    assert df.iloc[0]["text"] == "Tiền,x"

    messy = df.rename(columns={"text": "txt"})
    normalized = _normalize_csv_df(messy, CELLS_COLUMNS)
    assert list(normalized.columns) == list(CELLS_COLUMNS)
    assert normalized.iloc[0]["text"] == ""


def test_read_cells_csv_relaxed_unquoted_comma_in_text() -> None:
    csv_text = (
        "row_idx,col_idx,text\n"
        "19,0,1. Đầu tư vào công ty liên doanh, liên kết.\n"
    )
    df = _read_csv_text_with_columns(
        csv_text,
        required_columns=CELLS_COLUMNS,
        label="cells.csv",
    )
    assert len(df) == 1
    assert df.iloc[0]["row_idx"] == "19"
    assert df.iloc[0]["col_idx"] == "0"
    assert df.iloc[0]["text"] == "1. Đầu tư vào công ty liên doanh, liên kết."


def test_model_prompt_contains_required_contract() -> None:
    sample_id = "FPT_2025_p002"
    image_path = "/tmp/images/FPT_2025_p002.png"
    prompt = _model_prompt_for_image_to_csv(sample_id, image_path)
    assert sample_id in prompt
    assert image_path in prompt
    assert "cells.csv" in prompt
    assert "rows.csv" in prompt
    assert "Return 2 CSV files" in prompt
    assert "Do not output explanations." in prompt


def test_include_registry_add_remove_and_dedupe(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sample_id = "AAA_2024Q3_p001"
    _write_manifest(dataset_root, sample_id)

    reg0 = _load_include_registry(dataset_root)
    assert reg0["included_sample_ids"] == []

    _set_included("BBB_2024Q3_p001", dataset_root, True)
    reg1 = _set_included(sample_id, dataset_root, True)
    assert reg1["included_sample_ids"] == [sample_id, "BBB_2024Q3_p001"]

    reg2 = _set_included(sample_id, dataset_root, True)
    assert reg2["included_sample_ids"] == [sample_id, "BBB_2024Q3_p001"]

    reg3 = _set_included(sample_id, dataset_root, False)
    assert reg3["included_sample_ids"] == ["BBB_2024Q3_p001"]


def test_filter_by_include_scope(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sample_id = "AAA_2024Q3_p001"
    _write_manifest(dataset_root, sample_id)
    ds = BenchmarkDatasetV2(dataset_root)

    _set_included(sample_id, dataset_root, True)
    included = _filter_by_include_scope(ds.samples, dataset_root, "included")
    not_included = _filter_by_include_scope(ds.samples, dataset_root, "not_included")
    all_samples = _filter_by_include_scope(ds.samples, dataset_root, "all")

    assert [s.sample_id for s in included] == [sample_id]
    assert [s.sample_id for s in not_included] == ["BBB_2024Q3_p001"]
    assert [s.sample_id for s in all_samples] == [sample_id, "BBB_2024Q3_p001"]


def test_build_included_manifest_ignores_missing_ids(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sample_id = "AAA_2024Q3_p001"
    _write_manifest(dataset_root, sample_id)

    _set_included(sample_id, dataset_root, True)
    _set_included("MISSING_SAMPLE_ID", dataset_root, True)

    manifest, counts, missing = _build_included_manifest(dataset_root)
    sample_ids = [row["sample_id"] for row in manifest["samples"]]
    assert sample_ids == [sample_id]
    assert counts["original_count"] == 2
    assert counts["included_requested"] == 2
    assert counts["included_found"] == 1
    assert missing == ["MISSING_SAMPLE_ID"]


def test_sample_navigation_helpers() -> None:
    ids = ["A", "B", "C"]
    selected_id, selected_idx = _resolve_selected_sample(ids, None)
    assert selected_id == "A"
    assert selected_idx == 0

    selected_id, selected_idx = _resolve_selected_sample(ids, "B")
    assert selected_id == "B"
    assert selected_idx == 1

    selected_id, selected_idx = _resolve_selected_sample(ids, "Z")
    assert selected_id == "A"
    assert selected_idx == 0

    assert _shift_sample_index(0, len(ids), -1) == 0
    assert _shift_sample_index(2, len(ids), 1) == 2
    assert _shift_sample_index(1, len(ids), 1) == 2
    assert _shift_sample_index(1, len(ids), -1) == 0
