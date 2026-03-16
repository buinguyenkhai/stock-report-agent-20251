import json
from pathlib import Path

from evaluation.benchmark_v2.debug_diffs import build_debug_diffs


def test_build_debug_diffs_reports_raw_gaps_and_telemetry(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    pred_root = tmp_path / "predictions"
    dataset_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "gt_markdown").mkdir(parents=True, exist_ok=True)

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
            }
        ],
    }
    (dataset_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    gt_md = "| Item | Value |\n| --- | --- |\n| A | 100 |\n| B | 200 |\n"
    pred_md = "| Item | Value |\n| --- | --- |\n| A | 100 |\n"
    (dataset_root / "gt_markdown/AAA_2024Q3_p001.md").write_text(gt_md, encoding="utf-8")
    (pred_root / "AAA_2024Q3_p001.raw.md").write_text(pred_md, encoding="utf-8")
    (pred_root / "AAA_2024Q3_p001.ocr_debug.json").write_text(
        json.dumps(
            {
                "hybrid_ocr_stats": {"total_cells": 12, "surya_cells_updated": 3},
                "telemetry": {"total_latency_ms": 55.0},
            },
            indent=2,
        ),
        encoding="utf-8",
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
    assert sample["telemetry"]["total_latency_ms"] == 55.0
