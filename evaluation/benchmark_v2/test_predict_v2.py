from pathlib import Path

from PIL import Image

from evaluation.benchmark_v2.predict import (
    _compare_table_structures,
    _content_crop_image,
    _extract_markdown_table_stats,
    _looks_structurally_collapsed,
)
from services.ocr.financial_table_reconstruction import (
    OcrWord,
    reconstruct_table_from_tokens,
    reconstruct_table_from_words,
)
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


def _word(
    text: str,
    left: int,
    top: int,
    *,
    width: int = 20,
    height: int = 12,
    line: int = 1,
    source_tag: str = "baseline",
) -> OcrWord:
    return OcrWord(
        text=text,
        left=left,
        top=top,
        width=width,
        height=height,
        conf=90.0,
        line_key=(1, 1, line),
        source_tag=source_tag,
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
    assert debug["header_band_count"] >= 1
    assert "| Chỉ tiêu |" in markdown or "| Mã số | Chỉ tiêu |" in markdown
    assert "Tiền" in markdown
    assert "200" in markdown


def test_reconstruct_table_from_words_preserves_header_stack() -> None:
    words = [
        _word("Thuyết", 110, 10, width=44, line=1),
        _word("minh", 158, 10, width=36, line=1),
        _word("Quý", 230, 10, width=26, line=1),
        _word("I", 260, 10, width=10, line=1),
        _word("Quý", 330, 10, width=26, line=1),
        _word("I", 360, 10, width=10, line=1),
        _word("Năm", 230, 26, width=32, line=2),
        _word("nay", 266, 26, width=24, line=2),
        _word("Năm", 330, 26, width=32, line=2),
        _word("trước", 366, 26, width=40, line=2),
        _word("Triệu", 230, 42, width=36, line=3),
        _word("VND", 270, 42, width=28, line=3),
        _word("Triệu", 330, 42, width=36, line=3),
        _word("VND", 370, 42, width=28, line=3),
        _word("XV", 10, 62, line=4),
        _word("Lãi", 50, 62, width=24, line=4),
        _word("cơ", 78, 62, width=18, line=4),
        _word("bản", 100, 62, width=28, line=4),
        _word("15", 130, 62, width=18, line=4),
        _word("1.682", 230, 62, width=40, line=4, source_tag="surya_updated"),
        _word("1.459", 330, 62, width=40, line=4),
    ]

    markdown, debug = reconstruct_table_from_words(words, image_size=(600, 800))
    assert debug["reconstruction_applied"] is True
    assert debug["header_band_count"] >= 2
    assert debug["header_present"] is True
    assert debug["source_tag_counts"]["surya_updated"] == 1
    assert "Quý I Năm nay Triệu VND" in markdown
    assert "Thuyết minh" in markdown


def test_reconstruct_table_from_tokens_prefers_table_region() -> None:
    tokens = [
        {"text": "Noise", "left": 20, "top": 20, "right": 70, "bottom": 32, "confidence": 0.9},
        {"text": "Mã", "left": 10, "top": 110, "right": 30, "bottom": 122, "confidence": 0.9, "line_key": [1, 1, 1], "source_tag": "baseline"},
        {"text": "số", "left": 34, "top": 110, "right": 54, "bottom": 122, "confidence": 0.9, "line_key": [1, 1, 1], "source_tag": "baseline"},
        {"text": "2024", "left": 230, "top": 110, "right": 262, "bottom": 122, "confidence": 0.9, "line_key": [1, 1, 1], "source_tag": "baseline"},
        {"text": "2023", "left": 330, "top": 110, "right": 362, "bottom": 122, "confidence": 0.9, "line_key": [1, 1, 1], "source_tag": "baseline"},
        {"text": "110", "left": 10, "top": 130, "right": 30, "bottom": 142, "confidence": 0.9, "line_key": [1, 1, 2], "source_tag": "baseline"},
        {"text": "Tiền", "left": 50, "top": 130, "right": 86, "bottom": 142, "confidence": 0.9, "line_key": [1, 1, 2], "source_tag": "baseline"},
        {"text": "100", "left": 230, "top": 130, "right": 258, "bottom": 142, "confidence": 0.9, "line_key": [1, 1, 2], "source_tag": "surya_updated"},
        {"text": "90", "left": 330, "top": 130, "right": 350, "bottom": 142, "confidence": 0.9, "line_key": [1, 1, 2], "source_tag": "baseline"},
    ]

    markdown, debug = reconstruct_table_from_tokens(
        tokens,
        page_size=(600, 800),
        table_regions=[{"left": 0, "top": 100, "right": 500, "bottom": 200}],
    )
    assert debug["reconstruction_applied"] is True
    assert debug["selection_mode"] == "docling_table_region"
    assert debug["source_tag_counts"]["surya_updated"] == 1
    assert "Tiền" in markdown


def test_compare_table_structures_prefers_real_grid_reconstruction() -> None:
    baseline_stats = _extract_markdown_table_stats(
        """
| Code | Name | Value |
| --- | --- | --- |
| | Tai san ngan han Tien va cac khoan tuong duong tien Dau tu ngan han | 100 90 200 180 |
"""
    )
    candidate_stats = _extract_markdown_table_stats(
        """
| Mã số | Chỉ tiêu | 2024 | 2023 |
| --- | --- | --- | --- |
| 110 | Tiền | 100 | 90 |
| 120 | Đầu tư | 200 | 180 |
"""
    )
    comparison = _compare_table_structures(
        baseline_stats,
        candidate_stats,
        candidate_debug={
            "reconstructed_row_count": 2,
            "numeric_column_count": 2,
            "header_present": True,
        },
    )
    assert comparison["baseline_collapsed"] is True
    assert comparison["candidate_collapsed"] is False
    assert comparison["select_candidate"] is False

    comparison = _compare_table_structures(
        baseline_stats,
        candidate_stats,
        candidate_debug={
            "reconstructed_row_count": 3,
            "numeric_column_count": 2,
            "header_present": True,
        },
    )
    assert comparison["select_candidate"] is True
