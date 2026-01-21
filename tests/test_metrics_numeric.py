import pytest

from evaluation.ocr_benchmark.metrics import extract_numeric_tokens


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("(1.234.567)", ["-1234567"]),
        ("1,234.56", ["1234.56"]),
        ("1.234,56", ["1234.56"]),
        ("12,5%", ["12.5%"]),
        ("1 234 567", ["1234567"]),
        ("-0", ["0"]),
    ],
)
def test_extract_numeric_tokens_normalization(raw: str, expected: list[str]) -> None:
    assert extract_numeric_tokens(raw) == expected


def test_extract_numeric_tokens_multitoken() -> None:
    text = "Doanh thu 1.234,56 va (7.890)"
    assert extract_numeric_tokens(text) == ["1234.56", "-7890"]
