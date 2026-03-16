import json
from pathlib import Path

from evaluation.benchmark_v2.run import run_benchmark


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
                "source_pdf_path": "pdf/BBB_2024Q3.pdf",
            },
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_run_benchmark_is_raw_only_and_aggregates_telemetry(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    pred_root = tmp_path / "predictions"
    dataset_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "gt_markdown").mkdir(parents=True, exist_ok=True)

    sample_id = "AAA_2024Q3_p001"
    _write_manifest(dataset_root, sample_id)
    gt_md = "| Item | Value |\n| --- | --- |\n| A | 100 |\n"
    (dataset_root / f"gt_markdown/{sample_id}.md").write_text(gt_md, encoding="utf-8")
    (pred_root / f"{sample_id}.raw.md").write_text(gt_md, encoding="utf-8")
    (pred_root / f"{sample_id}.ocr_debug.json").write_text(
        json.dumps(
            {
                "telemetry": {
                    "total_latency_ms": 123.4,
                    "peak_vram_reserved_mb": 2048.0,
                    "peak_vram_allocated_mb": 1024.0,
                    "cuda_enabled": True,
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (dataset_root / "gt_markdown/BBB_2024Q3_p001.md").write_text("", encoding="utf-8")

    result = run_benchmark(
        dataset_root=dataset_root,
        predictions_root=pred_root,
        split="dev",
        engine_name="test_engine",
        raw_scope="table_only",
        bootstrap_iters=10,
        seed=1,
    )

    assert result["benchmark_version"] == "v2_raw_only"
    assert "report_structured_results" not in result
    raw_summary = result["summary"]["raw"]
    assert "table_only_cer" in raw_summary
    assert "number_f1" in raw_summary
    telemetry = result["summary"]["telemetry"]
    assert telemetry["latency_ms"]["mean"] == 123.4
    assert telemetry["peak_vram_reserved_mb"]["max"] == 2048.0
    sample_result = result["sample_results"][0]
    assert sample_result["telemetry"]["cuda_enabled"] is True
    assert sample_result["ocr_debug"]["telemetry"]["total_latency_ms"] == 123.4


def test_run_benchmark_allows_single_split_with_include_scope(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    pred_root = tmp_path / "predictions"
    dataset_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "gt_markdown").mkdir(parents=True, exist_ok=True)

    sample_id = "AAA_2024Q3_p001"
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
            },
            {
                "sample_id": "AAA_2024Q3_p002",
                "split": "dev",
                "company": "AAA",
                "report_id": "AAA_2024Q3",
                "page_index": 2,
                "page_image_path": "images/AAA_2024Q3_p002.png",
                "gt_markdown_path": "gt_markdown/AAA_2024Q3_p002.md",
            },
        ],
    }
    (dataset_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (dataset_root / "included_samples.json").write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "mode": "include_table_pages",
                "included_sample_ids": [sample_id],
                "updated_at": "",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    gt_md = "| Item | Value |\n| --- | --- |\n| A | 100 |\n"
    (dataset_root / f"gt_markdown/{sample_id}.md").write_text(gt_md, encoding="utf-8")
    (pred_root / f"{sample_id}.raw.md").write_text(gt_md, encoding="utf-8")

    result = run_benchmark(
        dataset_root=dataset_root,
        predictions_root=pred_root,
        split="dev",
        include_scope="included",
        engine_name="test_engine",
        raw_scope="table_only",
        bootstrap_iters=10,
        seed=1,
    )

    assert result["include_scope"] == "included"
    assert result["dataset_stats"]["available_splits"] == ["dev"]
    assert len(result["sample_results"]) == 1
    assert result["summary"]["counts"]["samples_total"] == 1
