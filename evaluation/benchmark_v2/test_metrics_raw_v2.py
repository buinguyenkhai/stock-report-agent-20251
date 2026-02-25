import pytest

from evaluation.benchmark_v2.metrics_raw import calculate_raw_metrics


def test_table_only_metrics_ignore_outside_table_text() -> None:
    reference = """
outside reference narrative 111
| Item | Value |
| --- | --- |
| A | 100 |
| B | 200 |
footer 222
"""
    hypothesis_a = """
random narrative alpha 333
| Item | Value |
| --- | --- |
| A | 100 |
| B | 200 |
tail 444
"""
    hypothesis_b = """
completely different narrative 999999
| Item | Value |
| --- | --- |
| A | 100 |
| B | 200 |
different footer 888888
"""

    m1 = calculate_raw_metrics(hypothesis_a, reference, scope="table_only")
    m2 = calculate_raw_metrics(hypothesis_b, reference, scope="table_only")

    assert m1.table_only_cer == pytest.approx(m2.table_only_cer)
    assert m1.table_only_wer == pytest.approx(m2.table_only_wer)
    assert m1.table_cell_f1 == pytest.approx(m2.table_cell_f1)
    assert m1.number_f1 == pytest.approx(m2.number_f1)


def test_table_only_metrics_deterministic_when_no_tables_present() -> None:
    m = calculate_raw_metrics("no table 123", "also no table 456", scope="table_only")
    assert m.table_only_cer == pytest.approx(0.0)
    assert m.table_only_wer == pytest.approx(0.0)
    assert m.table_cell_f1 == pytest.approx(1.0)
    assert m.number_f1 == pytest.approx(1.0)
