import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
fitz = pytest.importorskip("fitz")

from evaluation.benchmark_v2.annotation_app import (  # noqa: E402
    _build_manifest_from_pdfs,
    _exclude_sample_from_manifest,
    _render_page_image_if_missing,
    filter_samples,
    get_sample_status,
)
from evaluation.benchmark_v2.csv_codec import update_meta  # noqa: E402
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
