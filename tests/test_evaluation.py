import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.matchers import LLMBasedMatcher, normalize_name
from evaluation.canonical_format import FinancialItem, FinancialStatement, normalize_to_billions
from evaluation.metrics import evaluate_section


class TestNormalizeName:
    """Tests for name normalization."""
    
    def test_prefix_removal(self):
        """Test that prefixes are removed."""
        assert "tien" in normalize_name("I. Tiền")
        assert "doanh thu" in normalize_name("1. Doanh thu")
    
    def test_accent_removal(self):
        """Test accent removal."""
        normalized = normalize_name("Tiền và tương đương tiền")
        assert "tien" in normalized
        assert "tuong duong" in normalized


class TestNormalizeToBillions:
    """Tests for unit normalization."""
    
    def test_vnd_to_billions(self):
        """Test VND (full dong) to billions conversion."""
        # 1 billion VND
        result = normalize_to_billions(1_000_000_000, "VND")
        assert result == pytest.approx(1.0, rel=1e-6)
    
    def test_trieu_to_billions(self):
        """Test triệu VND to billions conversion."""
        # 1000 million = 1 billion
        result = normalize_to_billions(1000, "triệu VND")
        assert result == pytest.approx(1.0, rel=1e-6)
    
    def test_ty_to_billions(self):
        """Test tỷ VND to billions (no conversion)."""
        result = normalize_to_billions(1.5, "tỷ VND")
        assert result == pytest.approx(1.5, rel=1e-6)
    
    def test_nghin_to_billions(self):
        """Test nghìn VND to billions conversion."""
        # 1 million thousands = 1 billion
        result = normalize_to_billions(1_000_000, "nghìn VND")
        assert result == pytest.approx(1.0, rel=1e-6)


class TestEvaluateSection:
    """Tests for section evaluation."""
    
    def test_perfect_match(self):
        """Test evaluation with perfect matching."""
        ocr_statement = FinancialStatement(
            statement_type="BS",
            items=[
                FinancialItem(item_name="Tiền", value=100.0),
                FinancialItem(item_name="Tài sản", value=200.0),
            ]
        )
        vnstock_statement = FinancialStatement(
            statement_type="BS",
            items=[
                FinancialItem(item_name="Tiền", value=100.0),
                FinancialItem(item_name="Tài sản", value=200.0),
            ]
        )
        
        matcher = LLMBasedMatcher()
        result = evaluate_section(ocr_statement, vnstock_statement, matcher)
        
        assert result.match_rate == 1.0
        assert result.value_accuracy == 1.0
        assert result.mape == 0.0
    
    def test_value_error(self):
        """Test evaluation with value errors."""
        ocr_statement = FinancialStatement(
            statement_type="BS",
            items=[FinancialItem(item_name="Tiền", value=110.0)]  # 10% error
        )
        vnstock_statement = FinancialStatement(
            statement_type="BS",
            items=[FinancialItem(item_name="Tiền", value=100.0)]
        )
        
        matcher = LLMBasedMatcher()
        result = evaluate_section(ocr_statement, vnstock_statement, matcher, tolerance=0.05)
        
        assert result.match_rate == 1.0
        assert result.value_accuracy == 0.0  # 10% > 5% tolerance
        assert result.mape == pytest.approx(10.0, rel=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
