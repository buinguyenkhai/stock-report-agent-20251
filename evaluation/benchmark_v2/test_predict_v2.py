from pathlib import Path

from PIL import Image

from evaluation.benchmark_v2.predict import (
    _content_crop_image,
    _extract_markdown_table_stats,
    _looks_structurally_collapsed,
)


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
