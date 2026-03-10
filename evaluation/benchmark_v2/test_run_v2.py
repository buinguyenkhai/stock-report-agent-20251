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
                "gt_structured_path": f"gt_structured/{sample_id}.json",
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


def _seed_ground_truth_and_predictions(dataset_root: Path, pred_root: Path, sample_id: str) -> None:
    (dataset_root / "gt_markdown").mkdir(parents=True, exist_ok=True)
    (dataset_root / "gt_structured").mkdir(parents=True, exist_ok=True)
    (pred_root).mkdir(parents=True, exist_ok=True)

    gt_md = "| Item | Value |\n| --- | --- |\n| A | 100 |\n"
    (dataset_root / f"gt_markdown/{sample_id}.md").write_text(gt_md, encoding="utf-8")

    gt_struct = {
        "balance_sheet": {"items": [{"item_code": "I", "item_name": "A", "value": 100.0}]},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    (dataset_root / f"gt_structured/{sample_id}.json").write_text(
        json.dumps(gt_struct, indent=2), encoding="utf-8"
    )

    (pred_root / f"{sample_id}.raw.md").write_text(
        "outside text 999\n| Item | Value |\n| --- | --- |\n| A | 100 |\n", encoding="utf-8"
    )
    (pred_root / f"{sample_id}.structured.json").write_text(
        json.dumps(gt_struct, indent=2), encoding="utf-8"
    )


def test_run_benchmark_includes_raw_scope_and_table_only_fields(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    pred_root = tmp_path / "predictions"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sample_id = "AAA_2024Q3_p001"

    _write_manifest(dataset_root, sample_id)
    _seed_ground_truth_and_predictions(dataset_root, pred_root, sample_id)

    result = run_benchmark(
        dataset_root=dataset_root,
        predictions_root=pred_root,
        split="dev",
        engine_name="test_engine",
        raw_scope="table_only",
        bootstrap_iters=10,
        seed=1,
    )

    assert result["raw_scope"] == "table_only"
    raw_summary = result["summary"]["raw"]
    assert "table_only_cer" in raw_summary
    assert "table_only_wer" in raw_summary
    assert "table_cell_f1" in raw_summary
    assert "number_f1" in raw_summary


def test_run_benchmark_structured_is_report_level(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    pred_root = tmp_path / "predictions"
    dataset_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "gt_markdown").mkdir(parents=True, exist_ok=True)
    (dataset_root / "gt_structured").mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "1.0.0",
        "split_policy": "company_heldout_dev_test",
        "annotation_protocol": "single_annotator_two_pass",
        "samples": [
            {
                "sample_id": "AAA_2024Q3_p001",
                "split": "dev",
                "company": "AAA",
                "report_id": "AAA_2024Q3",
                "page_index": 1,
                "page_image_path": "images/AAA_2024Q3_p001.png",
                "gt_markdown_path": "gt_markdown/AAA_2024Q3_p001.md",
                "gt_structured_path": "gt_structured/AAA_2024Q3_p001.json",
            },
            {
                "sample_id": "AAA_2024Q3_p002",
                "split": "dev",
                "company": "AAA",
                "report_id": "AAA_2024Q3",
                "page_index": 2,
                "page_image_path": "images/AAA_2024Q3_p002.png",
                "gt_markdown_path": "gt_markdown/AAA_2024Q3_p002.md",
                "gt_structured_path": "gt_structured/AAA_2024Q3_p002.json",
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
    (dataset_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Page-level GT/PRED for one report (AAA_2024Q3) split across two pages.
    (dataset_root / "gt_markdown/AAA_2024Q3_p001.md").write_text(
        "| Item | Value |\n| --- | --- |\n| A | 100 |\n", encoding="utf-8"
    )
    (dataset_root / "gt_markdown/AAA_2024Q3_p002.md").write_text(
        "| Item | Value |\n| --- | --- |\n| B | 200 |\n", encoding="utf-8"
    )
    (pred_root / "AAA_2024Q3_p001.raw.md").write_text(
        "| Item | Value |\n| --- | --- |\n| A | 100 |\n", encoding="utf-8"
    )
    (pred_root / "AAA_2024Q3_p002.raw.md").write_text(
        "| Item | Value |\n| --- | --- |\n| B | 200 |\n", encoding="utf-8"
    )

    gt_page1 = {
        "balance_sheet": {"items": [{"item_code": "110", "item_name": "A", "value": 100.0}]},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    gt_page2 = {
        "balance_sheet": {"items": [{"item_code": "120", "item_name": "B", "value": 200.0}]},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    (dataset_root / "gt_structured/AAA_2024Q3_p001.json").write_text(
        json.dumps(gt_page1, indent=2), encoding="utf-8"
    )
    (dataset_root / "gt_structured/AAA_2024Q3_p002.json").write_text(
        json.dumps(gt_page2, indent=2), encoding="utf-8"
    )
    (pred_root / "AAA_2024Q3_p001.structured.json").write_text(
        json.dumps(gt_page1, indent=2), encoding="utf-8"
    )
    (pred_root / "AAA_2024Q3_p002.structured.json").write_text(
        json.dumps(gt_page2, indent=2), encoding="utf-8"
    )

    # Minimal test split files to satisfy manifest references.
    (dataset_root / "gt_markdown/BBB_2024Q3_p001.md").write_text("", encoding="utf-8")
    (dataset_root / "gt_structured/BBB_2024Q3_p001.json").write_text(
        json.dumps(
            {
                "balance_sheet": {"items": []},
                "income_statement": {"items": []},
                "cash_flow": {"items": []},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = run_benchmark(
        dataset_root=dataset_root,
        predictions_root=pred_root,
        split="dev",
        engine_name="test_engine",
        raw_scope="table_only",
        bootstrap_iters=10,
        seed=1,
    )

    assert result["structured_scope"] == "report_only"
    assert len(result["sample_results"]) == 2
    assert len(result["report_structured_results"]) == 1
    report_result = result["report_structured_results"][0]
    assert report_result["report_id"] == "AAA_2024Q3"
    assert report_result["page_count"] == 2
    assert report_result["structured_metrics"]["row_f1"] == 1.0

    counts = result["summary"]["counts"]
    assert counts["samples_total"] == 2
    assert counts["reports_total"] == 1
    assert counts["reports_structured_scored"] == 1


def test_run_benchmark_allows_single_split_with_include_scope(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    pred_root = tmp_path / "predictions"
    dataset_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "gt_markdown").mkdir(parents=True, exist_ok=True)
    (dataset_root / "gt_structured").mkdir(parents=True, exist_ok=True)

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
                "gt_structured_path": f"gt_structured/{sample_id}.json",
            },
            {
                "sample_id": "AAA_2024Q3_p002",
                "split": "dev",
                "company": "AAA",
                "report_id": "AAA_2024Q3",
                "page_index": 2,
                "page_image_path": "images/AAA_2024Q3_p002.png",
                "gt_markdown_path": "gt_markdown/AAA_2024Q3_p002.md",
                "gt_structured_path": "gt_structured/AAA_2024Q3_p002.json",
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
    gt_struct = {
        "balance_sheet": {"items": [{"item_code": "110", "item_name": "A", "value": 100.0}]},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    (dataset_root / f"gt_markdown/{sample_id}.md").write_text(gt_md, encoding="utf-8")
    (dataset_root / f"gt_structured/{sample_id}.json").write_text(
        json.dumps(gt_struct, indent=2), encoding="utf-8"
    )
    (pred_root / f"{sample_id}.raw.md").write_text(gt_md, encoding="utf-8")
    (pred_root / f"{sample_id}.structured.json").write_text(
        json.dumps(gt_struct, indent=2), encoding="utf-8"
    )

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
    assert result["summary"]["counts"]["reports_total"] == 1
