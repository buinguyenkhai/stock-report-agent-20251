from pathlib import Path

from PIL import Image

from evaluation.benchmark_v2.predict import (
    _content_crop_image,
    _extract_markdown_table_stats,
    _looks_structurally_collapsed,
)
from services.ocr.financial_table_reconstruction import OcrWord, reconstruct_table_from_words
from services.parser import AggregatedParser, ExtractionBundle, FinancialItem, ParsedReport, ParsedStatement


def test_markdown_stats_detect_collapsed_table_shape() -> None:
    markdown = """
| Code | Name | Value |
| --- | --- | --- |
| | TAI SAN NGAN HAN Tien va cac khoan tuong duong tien Dau tu tai chinh ngan han Phai thu khach hang Tra truoc cho nguoi ban Hang ton kho Tai san dai han va cac muc khac | 49.611 3.441 20.978 2.616 21.874 9.757 700.379 467.748 189.580 43.050 8.443 |
"""
    stats = _extract_markdown_table_stats(markdown)
    assert stats["pipe_row_count"] == 2
    assert stats["max_numeric_tokens_in_cell"] >= 10
    assert _looks_structurally_collapsed(stats) is True


def test_markdown_stats_leave_normal_table_healthy() -> None:
    markdown = """
| Ma so | Tai san | So cuoi ky | So dau ky |
| --- | --- | --- | --- |
| 110 | Tien va cac khoan tuong duong tien | 19.653.270 | 29.403.688 |
| 120 | Dau tu tai chinh ngan han | 6.758.243 | 10.413.625 |
| 130 | Cac khoan phai thu ngan han | 74.342.928 | 52.395.927 |
"""
    stats = _extract_markdown_table_stats(markdown)
    assert stats["pipe_row_count"] == 4
    assert stats["giant_cell_count"] == 0
    assert _looks_structurally_collapsed(stats) is False


def test_content_crop_image_trims_large_border(tmp_path: Path) -> None:
    img_path = tmp_path / "page.png"
    image = Image.new("RGB", (400, 300), "white")
    for x in range(120, 280):
        for y in range(80, 180):
            image.putpixel((x, y), (0, 0, 0))
    image.save(img_path)

    pdf_path, debug = _content_crop_image(img_path)
    try:
        assert pdf_path.exists()
        assert debug["crop_applied"] is True
        assert debug["final_size"][0] < debug["original_size"][0]
        assert debug["final_size"][1] < debug["original_size"][1]
    finally:
        pdf_path.unlink(missing_ok=True)


def test_parser_finalize_sets_unit_metadata_and_row_identity() -> None:
    parser = AggregatedParser(model="dummy")
    bundle = ExtractionBundle(
        balance_sheet="Đơn vị tính: triệu đồng\n| Chỉ tiêu | Quý III 2024 | Quý III 2023 |",
        metadata={"unit": "triệu đồng", "quarter": 3, "year": 2024},
    )
    report = ParsedReport(
        balance_sheet=ParsedStatement(
            items=[
                FinancialItem(item_name="Tiền và các khoản tương đương tiền", value=123000000.0),
                FinancialItem(
                    item_name="Tiền và các khoản tương đương tiền",
                    value=111000000.0,
                    period_key="2023Q3",
                ),
            ]
        )
    )

    finalized = parser._finalize_report(report, bundle)
    assert finalized.source_unit_multiplier_to_vnd == 1_000_000.0
    assert finalized.source_unit_label == "trieu dong"
    assert finalized.value_unit == "VND"
    assert finalized.balance_sheet.items[0].row_identity is not None
    assert "possible_scale_conflicts" in finalized.parse_audit


def _word(text: str, left: int, top: int, *, width: int = 20, height: int = 12, line: int = 1) -> OcrWord:
    return OcrWord(
        text=text,
        left=left,
        top=top,
        width=width,
        height=height,
        conf=90.0,
        line_key=(1, 1, line),
    )


def test_reconstruct_table_from_words_builds_markdown_grid() -> None:
    words = [
        _word("Mã", 10, 10, line=1),
        _word("số", 34, 10, line=1),
        _word("2024", 230, 10, width=32, line=1),
        _word("2023", 330, 10, width=32, line=1),
        _word("110", 10, 30, line=2),
        _word("Tiền", 50, 30, width=36, line=2),
        _word("100", 230, 30, width=28, line=2),
        _word("90", 330, 30, width=20, line=2),
        _word("120", 10, 50, line=3),
        _word("Đầu", 50, 50, width=28, line=3),
        _word("tư", 82, 50, width=20, line=3),
        _word("200", 230, 50, width=28, line=3),
        _word("180", 330, 50, width=28, line=3),
    ]

    markdown, debug = reconstruct_table_from_words(words, image_size=(600, 800))
    assert debug["reconstruction_applied"] is True
    assert debug["numeric_column_count"] == 2
    assert "| Chỉ tiêu |" in markdown or "| Mã số | Chỉ tiêu |" in markdown
    assert "Tiền" in markdown
    assert "200" in markdown
