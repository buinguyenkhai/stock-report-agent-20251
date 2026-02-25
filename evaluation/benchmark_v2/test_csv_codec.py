import json
from pathlib import Path

import pandas as pd
import pytest

from evaluation.benchmark_v2.csv_codec import (
    canonical_to_csv,
    compute_pilot_metrics,
    csv_to_canonical,
    load_csv_pack,
    load_meta,
    save_csv_pack,
    update_meta,
)


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
                "gt_table_cells_path": f"gt_cells/{sample_id}.json",
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


def _seed_canonical_files(root: Path, sample_id: str) -> None:
    (root / "gt_markdown").mkdir(parents=True, exist_ok=True)
    (root / "gt_structured").mkdir(parents=True, exist_ok=True)
    (root / "gt_cells").mkdir(parents=True, exist_ok=True)

    (root / f"gt_markdown/{sample_id}.md").write_text(
        "| Name | Value |\n| --- | --- |\n| A | 1 |\n",
        encoding="utf-8",
    )
    structured = {
        "balance_sheet": {
            "items": [
                {
                    "item_code": "I",
                    "item_name": "Cash",
                    "value": 1234.0,
                    "notes_ref": "5.1",
                    "original_name": "Tien",
                }
            ]
        },
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    (root / f"gt_structured/{sample_id}.json").write_text(
        json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gt_cells = {
        "rows": [
            [{"text": "Header", "col_span": 2}, {"text": "<MERGED>"}],
            [{"text": "Cash"}, {"text": "1,234"}],
        ]
    }
    (root / f"gt_cells/{sample_id}.json").write_text(
        json.dumps(gt_cells, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def test_csv_codec_round_trip_and_meta(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sample_id = "AAA_2024Q3_p001"
    _write_manifest(dataset_root, sample_id)
    _seed_canonical_files(dataset_root, sample_id)

    imported = canonical_to_csv(sample_id, dataset_root)
    assert imported["cells_count"] > 0
    assert imported["rows_count"] > 0

    out = csv_to_canonical(sample_id, dataset_root, validate=True)
    assert Path(out["gt_markdown_path"]).exists()
    assert Path(out["gt_structured_path"]).exists()
    assert Path(out["gt_table_cells_path"]).exists()

    generated_struct = json.loads(Path(out["gt_structured_path"]).read_text(encoding="utf-8"))
    assert generated_struct["balance_sheet"]["items"][0]["item_name"] == "Cash"
    assert generated_struct["balance_sheet"]["items"][0]["value"] == pytest.approx(1234.0)

    generated_cells = json.loads(Path(out["gt_table_cells_path"]).read_text(encoding="utf-8"))
    assert generated_cells["rows"][0][0]["col_span"] == 2

    update_meta(sample_id, dataset_root, {"pass1_done": True, "pass2_done": False})
    meta = load_meta(sample_id, dataset_root)
    assert meta["pass1_done"] is True
    assert meta["pass2_done"] is False

    update_meta(
        sample_id,
        dataset_root,
        {
            "total_rows": 100,
            "row_key_corrections": 1,
            "value_corrections": 0,
            "span_corrections": 2,
            "unresolved_span_loss_blocker": False,
        },
    )
    pilot = compute_pilot_metrics(dataset_root, sample_ids=[sample_id], threshold=0.02)
    assert pilot["row_value_mismatch_rate"] == pytest.approx(0.01)
    assert pilot["pass_gate"] is True


def test_csv_codec_rejects_span_overlap(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sample_id = "AAA_2024Q3_p001"
    _write_manifest(dataset_root, sample_id)
    _seed_canonical_files(dataset_root, sample_id)

    cells_df = pd.DataFrame(
        [
            {"row_idx": 0, "col_idx": 0, "text": "A"},
            {"row_idx": 0, "col_idx": 1, "text": "B"},
        ]
    )
    spans_df = pd.DataFrame([{"row_idx": 0, "col_idx": 0, "row_span": 1, "col_span": 2}])
    rows_df = pd.DataFrame(
        [
            {
                "statement": "balance_sheet",
                "item_code": "I",
                "item_name": "Cash",
                "value": "1,234",
                "notes_ref": "",
                "original_name": "",
            }
        ]
    )

    save_csv_pack(
        sample_id,
        dataset_root,
        cells=cells_df,
        spans=spans_df,
        rows=rows_df,
    )
    pack = load_csv_pack(sample_id, dataset_root)
    assert len(pack["cells"]) == 2

    with pytest.raises(ValueError, match="overlap"):
        csv_to_canonical(sample_id, dataset_root, validate=True)
