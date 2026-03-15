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
    migrate_rows_to_structured_contract,
    save_csv_pack,
    update_meta,
    validate_csv_pack,
)
from evaluation.benchmark_v2.migrate_gt_contract import migrate_gt_contract
from evaluation.benchmark_v2.normalize_gt_units import normalize_gt_units


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
            [{"text": "Name"}, {"text": "Value"}],
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
    assert generated_cells["rows"][0][0]["text"] == "Name"

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
        },
    )
    pilot = compute_pilot_metrics(dataset_root, sample_ids=[sample_id], threshold=0.02)
    assert pilot["row_value_mismatch_rate"] == pytest.approx(0.01)
    assert pilot["pass_gate"] is True


def test_csv_codec_validates_without_spans(tmp_path: Path) -> None:
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
        rows=rows_df,
    )
    pack = load_csv_pack(sample_id, dataset_root)
    assert len(pack["cells"]) == 2
    assert "spans" not in pack

    out = csv_to_canonical(sample_id, dataset_root, validate=True)
    assert Path(out["gt_markdown_path"]).exists()


def test_legacy_spans_csv_is_ignored(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sample_id = "AAA_2024Q3_p001"
    _write_manifest(dataset_root, sample_id)
    _seed_canonical_files(dataset_root, sample_id)

    csv_root = dataset_root / "gt_csv" / sample_id
    csv_root.mkdir(parents=True, exist_ok=True)
    (csv_root / "spans.csv").write_text(
        "row_idx,col_idx,row_span,col_span\n0,0,1,2\n",
        encoding="utf-8",
    )

    imported = canonical_to_csv(sample_id, dataset_root)
    assert imported["cells_count"] > 0

    pack = load_csv_pack(sample_id, dataset_root)
    assert "spans" not in pack


def test_validate_csv_pack_flags_missing_csv_directory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sample_id = "AAA_2024Q3_p001"
    _write_manifest(dataset_root, sample_id)

    errors = validate_csv_pack(sample_id, dataset_root)
    assert len(errors) == 1
    assert "gt_csv pack not found" in errors[0]


def test_normalize_gt_units_updates_rows_csv_and_canonical(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sample_id = "AAA_2024Q3_p001"
    _write_manifest(dataset_root, sample_id)
    _seed_canonical_files(dataset_root, sample_id)

    cells_df = pd.DataFrame(
        [
            {"row_idx": 0, "col_idx": 0, "text": "Chỉ tiêu"},
            {"row_idx": 0, "col_idx": 1, "text": "Giá trị"},
            {"row_idx": 1, "col_idx": 0, "text": "Tiền mặt"},
            {"row_idx": 1, "col_idx": 1, "text": "12"},
        ]
    )
    rows_df = pd.DataFrame(
        [
            {
                "statement": "balance_sheet",
                "item_code": "",
                "item_name": "Tiền mặt",
                "value": "12",
                "notes_ref": "",
                "original_name": "Tiền mặt",
            }
        ]
    )
    save_csv_pack(sample_id, dataset_root, cells=cells_df, rows=rows_df)
    (dataset_root / f"gt_markdown/{sample_id}.md").write_text(
        "BẢNG CÂN ĐỐI KẾ TOÁN\n\nĐơn vị tính: Triệu VND\n\n| Chỉ tiêu | Giá trị |\n| --- | --- |\n| Tiền mặt | 12 |\n",
        encoding="utf-8",
    )

    summary = normalize_gt_units(dataset_root=dataset_root, include_scope="all", split="dev")
    assert summary["changed_count"] == 1

    pack = load_csv_pack(sample_id, dataset_root)
    assert pack["rows"].iloc[0]["value"] == "12000000"

    regenerated = json.loads((dataset_root / f"gt_structured/{sample_id}.json").read_text(encoding="utf-8"))
    assert regenerated["balance_sheet"]["items"][0]["value"] == 12000000.0

    meta = load_meta(sample_id, dataset_root)
    assert meta["value_unit_normalized_to"] == "VND"
    assert meta["report_unit_multiplier"] == 1_000_000.0


def test_csv_codec_round_trips_identity_fields(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sample_id = "AAA_2024Q3_p001"
    _write_manifest(dataset_root, sample_id)
    _seed_canonical_files(dataset_root, sample_id)

    rows_df = pd.DataFrame(
        [
            {
                "statement": "cash_flow",
                "item_code": "",
                "item_name": "Lưu chuyển tiền thuần trong kỳ",
                "value": "12599097000000",
                "notes_ref": "",
                "original_name": "Lưu chuyển tiền thuần trong kỳ",
                "row_identity": "net_cash_flow",
                "column_label": "Từ 1/1/2024 đến 30/9/2024",
                "period_key": "2024Q3_YTD",
            }
        ]
    )
    cells_df = pd.DataFrame(
        [
            {"row_idx": 0, "col_idx": 0, "text": "Chỉ tiêu"},
            {"row_idx": 0, "col_idx": 1, "text": "Giá trị"},
            {"row_idx": 1, "col_idx": 0, "text": "Lưu chuyển tiền thuần trong kỳ"},
            {"row_idx": 1, "col_idx": 1, "text": "12.599.097"},
        ]
    )
    save_csv_pack(sample_id, dataset_root, cells=cells_df, rows=rows_df)

    out = csv_to_canonical(sample_id, dataset_root, validate=True)
    regenerated = json.loads(Path(out["gt_structured_path"]).read_text(encoding="utf-8"))
    item = regenerated["cash_flow"]["items"][0]
    assert item["row_identity"] == "net_cash_flow"
    assert item["column_label"] == "Từ 1/1/2024 đến 30/9/2024"
    assert item["period_key"] == "2024Q3_YTD"

    canonical_to_csv(sample_id, dataset_root)
    pack = load_csv_pack(sample_id, dataset_root)
    assert pack["rows"].iloc[0]["row_identity"] == "net_cash_flow"
    assert pack["rows"].iloc[0]["column_label"] == "Từ 1/1/2024 đến 30/9/2024"
    assert pack["rows"].iloc[0]["period_key"] == "2024Q3_YTD"


def test_migrate_rows_to_structured_contract_expands_multi_column_values(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sample_id = "AAA_Q4_2024_p001"
    _write_manifest(dataset_root, sample_id)
    _seed_canonical_files(dataset_root, sample_id)

    cells_df = pd.DataFrame(
        [
            {"row_idx": 0, "col_idx": 0, "text": "Mã số"},
            {"row_idx": 0, "col_idx": 1, "text": "Chỉ tiêu"},
            {"row_idx": 0, "col_idx": 2, "text": "31/12/2024"},
            {"row_idx": 0, "col_idx": 3, "text": "31/12/2023"},
            {"row_idx": 1, "col_idx": 0, "text": "110"},
            {"row_idx": 1, "col_idx": 1, "text": "Tiền và tương đương tiền"},
            {"row_idx": 1, "col_idx": 2, "text": "1.200"},
            {"row_idx": 1, "col_idx": 3, "text": "900"},
        ]
    )
    rows_df = pd.DataFrame(
        [
            {
                "statement": "balance_sheet",
                "item_code": "110",
                "item_name": "Tiền và tương đương tiền",
                "value": "1200",
                "notes_ref": "",
                "original_name": "Tiền và tương đương tiền",
            }
        ]
    )
    save_csv_pack(sample_id, dataset_root, cells=cells_df, rows=rows_df)

    summary = migrate_rows_to_structured_contract(sample_id, dataset_root)
    assert summary["changed"] is True
    assert summary["rows_after"] == 2

    pack = load_csv_pack(sample_id, dataset_root)
    assert len(pack["rows"]) == 2
    assert set(pack["rows"]["column_label"]) == {"31/12/2024", "31/12/2023"}
    assert set(pack["rows"]["period_key"]) == {"2024FY", "2023FY"}

    structured = json.loads((dataset_root / f"gt_structured/{sample_id}.json").read_text(encoding="utf-8"))
    items = structured["balance_sheet"]["items"]
    assert len(items) == 2
    assert items[0]["row_identity"] == "balance_sheet|code:110"


def test_migrate_rows_to_structured_contract_preserves_vnd_normalization(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sample_id = "AAA_Q1_2024_p001"
    _write_manifest(dataset_root, sample_id)
    _seed_canonical_files(dataset_root, sample_id)

    cells_df = pd.DataFrame(
        [
            {"row_idx": 0, "col_idx": 0, "text": "Chỉ tiêu"},
            {"row_idx": 0, "col_idx": 1, "text": "31/03/2024 triệu đồng"},
            {"row_idx": 0, "col_idx": 2, "text": "31/12/2023 triệu đồng"},
            {"row_idx": 1, "col_idx": 0, "text": "Tiền mặt"},
            {"row_idx": 1, "col_idx": 1, "text": "12"},
            {"row_idx": 1, "col_idx": 2, "text": "10"},
        ]
    )
    rows_df = pd.DataFrame(
        [
            {
                "statement": "balance_sheet",
                "item_code": "",
                "item_name": "Tiền mặt",
                "value": "12000000",
                "notes_ref": "",
                "original_name": "Tiền mặt",
            }
        ]
    )
    save_csv_pack(sample_id, dataset_root, cells=cells_df, rows=rows_df)
    update_meta(
        sample_id,
        dataset_root,
        {
            "value_unit_normalized_to": "VND",
            "report_unit_multiplier": 1_000_000.0,
        },
    )

    summary = migrate_rows_to_structured_contract(sample_id, dataset_root)
    assert summary["changed"] is True

    pack = load_csv_pack(sample_id, dataset_root)
    assert set(pack["rows"]["value"]) == {"12000000.0", "10000000.0"}


def test_vcb_q1_2022_p008_backfill_is_complete() -> None:
    dataset_root = Path("data/benchmark_v2")
    sample_id = "VCB_Q1_2022_p008"

    errors = validate_csv_pack(sample_id, dataset_root)
    assert errors == []

    pack = load_csv_pack(sample_id, dataset_root)
    rows = pack["rows"]
    assert len(rows) == 28
    assert set(rows["statement"]) == {"income_statement"}
    assert set(rows["item_code"]) == {"7", "8", "XI", "XII", "XIII", "XIV", "XV"}
    assert set(rows["column_label"]) == {
        "Quý I Năm nay Triệu VND",
        "Quý I Năm trước Triệu VND",
        "Lũy kế từ đầu năm Năm nay Triệu VND",
        "Lũy kế từ đầu năm Năm trước Triệu VND",
    }

    structured = json.loads((dataset_root / f"gt_structured/{sample_id}.json").read_text(encoding="utf-8"))
    assert len(structured["income_statement"]["items"]) == 28
    eps_items = [item for item in structured["income_statement"]["items"] if item.get("item_code") == "XV"]
    assert {item["value"] for item in eps_items} == {1682.0, 1459.0}


def test_included_gt_has_no_remaining_annotation_gap() -> None:
    summary = migrate_gt_contract(
        dataset_root=Path("data/benchmark_v2"),
        include_scope="included",
        split="dev",
        force=False,
    )
    assert summary["failed_count"] == 0
    skipped = summary["skipped_samples"]
    assert not any(sample.get("reason") == "no_annotated_rows" for sample in skipped)
