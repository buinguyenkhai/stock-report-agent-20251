import json
from pathlib import Path

from evaluation.benchmark_v2.dataset import TableSample
from evaluation.benchmark_v2.report_assembler import (
    assemble_report_structured_from_pages,
    build_gt_structured_report_files,
    build_prediction_structured_report_files,
)


def _sample(sample_id: str, page_index: int, *, report_id: str = "AAA_2024Q3") -> TableSample:
    return TableSample(
        sample_id=sample_id,
        split="dev",
        company="AAA",
        report_id=report_id,
        page_index=page_index,
        page_image_path=f"images/{sample_id}.png",
        gt_markdown_path=f"gt_markdown/{sample_id}.md",
        gt_structured_path=f"gt_structured/{sample_id}.json",
    )


def test_assemble_report_structured_guardrails() -> None:
    s1 = _sample("AAA_2024Q3_p001", 1)
    s2 = _sample("AAA_2024Q3_p002", 2)
    s3 = _sample("AAA_2024Q3_p003", 3)
    page1 = {
        "balance_sheet": {"items": [{"item_code": "110", "item_name": "Tiền", "value": 100.0}]},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    # Same key + same value => keep both rows.
    page2 = {
        "balance_sheet": {
            "items": [{"item_code": "110", "item_name": "Tiền", "value": 100.0, "notes_ref": "5"}]
        },
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    # Same key + conflicting value => keep both rows and log a warning.
    page3 = {
        "balance_sheet": {"items": [{"item_code": "110", "item_name": "Tiền", "value": 120.0}]},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }

    merged, meta = assemble_report_structured_from_pages([(s1, page1), (s2, page2), (s3, page3)])
    items = merged["balance_sheet"]["items"]
    assert len(items) == 3
    assert [item["value"] for item in items] == [100.0, 100.0, 120.0]
    assert meta["conflict_count"] == 1
    row_key = "balance_sheet|code:110"
    assert len(meta["row_sources"]["balance_sheet"][row_key]) == 3


def test_assemble_report_structured_preserves_period_identity() -> None:
    s1 = _sample("AAA_2024Q3_p001", 1)
    s2 = _sample("AAA_2024Q3_p002", 2)
    page1 = {
        "cash_flow": {
            "items": [
                {
                    "item_name": "Lưu chuyển tiền thuần trong kỳ",
                    "period_key": "2024Q3_YTD",
                    "column_label": "Từ 1/1/2024 đến 30/9/2024",
                    "value": 12599097000000.0,
                }
            ]
        },
        "balance_sheet": {"items": []},
        "income_statement": {"items": []},
    }
    page2 = {
        "cash_flow": {
            "items": [
                {
                    "item_name": "Lưu chuyển tiền thuần trong kỳ",
                    "period_key": "2023Q3_YTD",
                    "column_label": "Từ 1/1/2023 đến 30/9/2023",
                    "value": -5100088000000.0,
                }
            ]
        },
        "balance_sheet": {"items": []},
        "income_statement": {"items": []},
    }

    merged, meta = assemble_report_structured_from_pages([(s1, page1), (s2, page2)])
    items = merged["cash_flow"]["items"]
    assert len(items) == 2
    keys = set(meta["row_sources"]["cash_flow"].keys())
    assert "cash_flow|name:lưu chuyển tiền thuần trong kỳ|column:từ 1/1/2024 đến 30/9/2024" in keys
    assert "cash_flow|name:lưu chuyển tiền thuần trong kỳ|column:từ 1/1/2023 đến 30/9/2023" in keys


def test_build_gt_and_prediction_report_structured_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    pred_root = tmp_path / "pred"
    (dataset_root / "gt_markdown").mkdir(parents=True, exist_ok=True)
    (dataset_root / "gt_structured").mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)

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

    (dataset_root / "gt_markdown/AAA_2024Q3_p001.md").write_text("", encoding="utf-8")
    (dataset_root / "gt_markdown/AAA_2024Q3_p002.md").write_text("", encoding="utf-8")
    (dataset_root / "gt_markdown/BBB_2024Q3_p001.md").write_text("", encoding="utf-8")
    page1 = {
        "balance_sheet": {"items": [{"item_code": "110", "item_name": "A", "value": 100.0}]},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    page2 = {
        "balance_sheet": {"items": [{"item_code": "120", "item_name": "B", "value": 200.0}]},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    empty = {"balance_sheet": {"items": []}, "income_statement": {"items": []}, "cash_flow": {"items": []}}
    (dataset_root / "gt_structured/AAA_2024Q3_p001.json").write_text(
        json.dumps(page1, indent=2), encoding="utf-8"
    )
    (dataset_root / "gt_structured/AAA_2024Q3_p002.json").write_text(
        json.dumps(page2, indent=2), encoding="utf-8"
    )
    (dataset_root / "gt_structured/BBB_2024Q3_p001.json").write_text(
        json.dumps(empty, indent=2), encoding="utf-8"
    )

    (pred_root / "AAA_2024Q3_p001.structured.json").write_text(
        json.dumps(page1, indent=2), encoding="utf-8"
    )
    (pred_root / "AAA_2024Q3_p002.structured.json").write_text(
        json.dumps(page2, indent=2), encoding="utf-8"
    )

    gt_counts = build_gt_structured_report_files(dataset_root, split="dev")
    assert gt_counts["reports_saved"] == 1
    gt_report = json.loads(
        (dataset_root / "gt_structured_report/AAA_2024Q3.json").read_text(encoding="utf-8")
    )
    assert len(gt_report["balance_sheet"]["items"]) == 2

    pred_counts = build_prediction_structured_report_files(
        dataset_root, pred_root, split="dev", strict_missing=True
    )
    assert pred_counts["reports_saved"] == 1
    assert (pred_root / "report_structured/AAA_2024Q3.structured.json").exists()
