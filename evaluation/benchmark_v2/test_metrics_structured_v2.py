import pytest

from evaluation.benchmark_v2.metrics_structured import calculate_structured_metrics


def _empty_structured() -> dict:
    return {
        "balance_sheet": {"items": []},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }


def test_structured_metrics_preserve_duplicate_rows() -> None:
    reference = _empty_structured()
    prediction = _empty_structured()

    reference["cash_flow"]["items"] = [
        {"item_code": "40", "item_name": "Net financing cash flow", "value": 5641.0},
        {"item_code": "40", "item_name": "Net financing cash flow", "value": -2846.0},
    ]
    prediction["cash_flow"]["items"] = [
        {"item_code": "40", "item_name": "Net financing cash flow", "value": 5641.0},
        {"item_code": "40", "item_name": "Net financing cash flow", "value": -2846.0},
    ]

    metrics = calculate_structured_metrics(prediction, reference)
    assert metrics.row_f1 == pytest.approx(1.0)
    assert metrics.value_exact_accuracy == pytest.approx(1.0)
    assert metrics.matched_row_count == 2
    assert metrics.gt_row_count == 2
    assert metrics.pred_row_count == 2


def test_structured_metrics_use_notes_ref_to_separate_same_name_rows() -> None:
    reference = _empty_structured()
    prediction = _empty_structured()

    reference["balance_sheet"]["items"] = [
        {"item_name": "Cho vay khách hàng", "notes_ref": "", "value": 617924570.0},
        {"item_name": "Cho vay khách hàng", "notes_ref": "9", "value": 626290777.0},
    ]
    prediction["balance_sheet"]["items"] = [
        {"item_name": "Cho vay khách hàng", "notes_ref": "", "value": 617924570.0},
        {"item_name": "Cho vay khách hàng", "notes_ref": "9", "value": 626290777.0},
    ]

    metrics = calculate_structured_metrics(prediction, reference)
    assert metrics.row_f1 == pytest.approx(1.0)
    assert metrics.value_exact_accuracy == pytest.approx(1.0)
    assert metrics.matched_row_count == 2


def test_structured_metrics_use_period_key_to_separate_multi_period_rows() -> None:
    reference = _empty_structured()
    prediction = _empty_structured()

    reference["cash_flow"]["items"] = [
        {"item_name": "Lưu chuyển tiền thuần trong kỳ", "period_key": "2024Q3_YTD", "value": 12599097000000.0},
        {"item_name": "Lưu chuyển tiền thuần trong kỳ", "period_key": "2023Q3_YTD", "value": -5100088000000.0},
    ]
    prediction["cash_flow"]["items"] = [
        {"item_name": "Lưu chuyển tiền thuần trong kỳ", "period_key": "2024Q3_YTD", "value": 12599097000000.0},
        {"item_name": "Lưu chuyển tiền thuần trong kỳ", "period_key": "2023Q3_YTD", "value": -5100088000000.0},
    ]

    metrics = calculate_structured_metrics(prediction, reference)
    assert metrics.row_f1 == pytest.approx(1.0)
    assert metrics.value_exact_accuracy == pytest.approx(1.0)
    assert metrics.matched_row_count == 2


def test_structured_metrics_use_row_identity_when_names_repeat() -> None:
    reference = _empty_structured()
    prediction = _empty_structured()

    reference["balance_sheet"]["items"] = [
        {"item_name": "Nguyên giá", "row_identity": "tangible_cost", "value": 9348348000000.0},
        {"item_name": "Nguyên giá", "row_identity": "intangible_cost", "value": 8160074000000.0},
    ]
    prediction["balance_sheet"]["items"] = [
        {"item_name": "Nguyên giá", "row_identity": "tangible_cost", "value": 9348348000000.0},
        {"item_name": "Nguyên giá", "row_identity": "intangible_cost", "value": 8160074000000.0},
    ]

    metrics = calculate_structured_metrics(prediction, reference)
    assert metrics.row_f1 == pytest.approx(1.0)
    assert metrics.value_exact_accuracy == pytest.approx(1.0)


def test_structured_metrics_prefer_column_label_over_period_key() -> None:
    reference = _empty_structured()
    prediction = _empty_structured()

    reference["balance_sheet"]["items"] = [
        {
            "item_code": "110",
            "item_name": "Tiền và tương đương tiền",
            "column_label": "31/12/2024",
            "period_key": "2024FY",
            "value": 1200.0,
        }
    ]
    prediction["balance_sheet"]["items"] = [
        {
            "item_code": "110",
            "item_name": "Tiền và tương đương tiền",
            "column_label": "31/12/2024",
            "period_key": "2024-12-31",
            "value": 1200.0,
        }
    ]

    metrics = calculate_structured_metrics(prediction, reference)
    assert metrics.row_f1 == pytest.approx(1.0)
    assert metrics.value_exact_accuracy == pytest.approx(1.0)
