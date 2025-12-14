import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

class TestLLMUnitDetector:
    """Tests for LLM-based unit detection."""
    
    def test_detect_trieu_vnd(self):
        """Test detection of 'triệu VND' unit."""
        from services.llm_utils import LLMUnitDetector
        
        detector = LLMUnitDetector()
        
        text = """
        BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT
        Đơn vị tính: Triệu đồng
        Tại ngày 30 tháng 09 năm 2024
        """
        
        unit = detector.detect_unit(text)
        assert unit == "triệu VND"
    
    def test_detect_ty_vnd(self):
        """Test detection of 'tỷ VND' unit."""
        from services.llm_utils import LLMUnitDetector
        
        detector = LLMUnitDetector()
        
        text = """
        BÁO CÁO TÀI CHÍNH
        Đơn vị: Tỷ VNĐ
        """
        
        unit = detector.detect_unit(text)
        assert unit == "tỷ VND"
    
    def test_detect_vnd_default(self):
        """Test detection of VND (full dong) unit."""
        from services.llm_utils import LLMUnitDetector
        
        detector = LLMUnitDetector()
        
        text = """
        BÁO CÁO TÀI CHÍNH
        Đơn vị tính: VND
        """
        
        unit = detector.detect_unit(text)
        assert unit == "VND"


class TestLLMTableExtractor:
    """Tests for LLM-based table extraction."""
    
    def test_extract_sections(self):
        """Test extraction of BS, PL, CF sections."""
        from services.llm_utils import LLMTableExtractor
        
        extractor = LLMTableExtractor()

        markdown = """
# BẢNG CÂN ĐỐI KẾ TOÁN

| Chỉ tiêu | Mã số | Số cuối kỳ |
|----------|-------|------------|
| Tiền và tương đương tiền | 110 | 1,000,000 |
| Tài sản ngắn hạn | 100 | 5,000,000 |

# BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH

| Chỉ tiêu | Mã số | Số cuối kỳ |
|----------|-------|------------|
| Doanh thu | 01 | 10,000,000 |
| Lợi nhuận | 60 | 500,000 |

# BÁO CÁO LƯU CHUYỂN TIỀN TỆ

| Chỉ tiêu | Mã số | Số cuối kỳ |
|----------|-------|------------|
| Tiền thu từ bán hàng | 01 | 8,000,000 |
"""
        
        sections = extractor.extract_sections(markdown)
        
        assert "BS" in sections
        assert "PL" in sections
        assert "CF" in sections

class TestLLMItemMatcher:
    """Tests for LLM-based item matching."""
    
    def test_match_exact_names(self):
        """Test matching of identical names."""
        from services.llm_utils import LLMItemMatcher
        
        matcher = LLMItemMatcher()
        
        ocr_names = ["Tiền và tương đương tiền"]
        gt_names = ["Tiền và tương đương tiền"]
        
        mapping = matcher.batch_match(ocr_names, gt_names, "BS")
        
        assert mapping.get("Tiền và tương đương tiền") == "Tiền và tương đương tiền"
    
    def test_match_synonyms(self):
        """Test matching of synonymous names."""
        from services.llm_utils import LLMItemMatcher
        
        matcher = LLMItemMatcher()
        
        ocr_names = ["Tiền mặt và các khoản tương đương tiền"]
        gt_names = ["Tiền và tương đương tiền"]
        
        mapping = matcher.batch_match(ocr_names, gt_names, "BS")
        
        # Should match since they're semantically equivalent
        assert mapping.get("Tiền mặt và các khoản tương đương tiền") is not None
    
    def test_no_match_opposites(self):
        """Test that opposites are not matched."""
        from services.llm_utils import LLMItemMatcher
        
        matcher = LLMItemMatcher()
        
        ocr_names = ["Phải thu khách hàng"]
        gt_names = ["Phải trả người bán"]
        
        mapping = matcher.batch_match(ocr_names, gt_names, "BS")
        
        # Should NOT match since they're semantic opposites
        assert mapping.get("Phải thu khách hàng") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
