import json
from pathlib import Path

from evaluation.benchmark_v2.debug_diffs import build_debug_diffs


def test_build_debug_diffs_reports_raw_and_structured_gaps(tmp_path: Path) -> None:
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

    gt_md = "| Item | Value |\n| --- | --- |\n| A | 100 |\n| B | 200 |\n"
    pred_md = "| Item | Value |\n| --- | --- |\n| A | 100 |\n"
    (dataset_root / "gt_markdown/AAA_2024Q3_p001.md").write_text(gt_md, encoding="utf-8")
    (pred_root / "AAA_2024Q3_p001.raw.md").write_text(pred_md, encoding="utf-8")
    (pred_root / "AAA_2024Q3_p001.ocr_debug.json").write_text(
        json.dumps({"hybrid_ocr_stats": {"total_cells": 12, "surya_cells_updated": 3}}, indent=2),
        encoding="utf-8",
    )

    gt_struct = {
        "balance_sheet": {
            "items": [
                {"item_code": "110", "item_name": "A", "value": 100.0},
                {"item_code": "120", "item_name": "B", "value": 200.0},
            ]
        },
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    pred_struct = {
        "balance_sheet": {"items": [{"item_code": "110", "item_name": "A", "value": 101.0}]},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    (dataset_root / "gt_structured/AAA_2024Q3_p001.json").write_text(
        json.dumps(gt_struct, indent=2), encoding="utf-8"
    )
    (pred_root / "AAA_2024Q3_p001.structured.json").write_text(
        json.dumps(pred_struct, indent=2), encoding="utf-8"
    )

    # Minimal test split files to satisfy the manifest.
    empty_struct = {
        "balance_sheet": {"items": []},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    (dataset_root / "gt_markdown/BBB_2024Q3_p001.md").write_text("", encoding="utf-8")
    (dataset_root / "gt_structured/BBB_2024Q3_p001.json").write_text(
        json.dumps(empty_struct, indent=2), encoding="utf-8"
    )

    payload = build_debug_diffs(
        dataset_root=dataset_root,
        predictions_root=pred_root,
        split="dev",
        include_scope="all",
    )

    assert len(payload["sample_diffs"]) == 1
    sample = payload["sample_diffs"][0]
    assert sample["missing_numbers"] == [{"value": "200", "count": 1}]
    assert sample["raw_metrics"]["number_f1"] < 1.0
    assert sample["ocr_debug"]["hybrid_ocr_stats"]["surya_cells_updated"] == 3

    assert len(payload["report_diffs"]) == 1
    report = payload["report_diffs"][0]
    assert report["structured_available"] is True
    assert len(report["comparison"]["missing_rows"]) == 1
    assert len(report["comparison"]["value_mismatches"]) == 1
