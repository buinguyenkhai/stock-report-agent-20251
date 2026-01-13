"""
Combined Financial Tables Extractor

Extracts all 3 main financial statement tables in a single LLM call.
Alternative to using 3 separate extractors.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from .base import BaseExtractor, ExtractionResult

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class FinancialTablesResult:
    """Result containing all 3 financial tables."""
    balance_sheet: str = ""
    income_statement: str = ""
    cash_flow: str = ""
    success: bool = True
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_extraction_results(self) -> Dict[str, ExtractionResult]:
        """Convert to individual ExtractionResult objects."""
        return {
            "balance_sheet": ExtractionResult(
                extractor_name="balance_sheet",
                content=self.balance_sheet,
                success=bool(self.balance_sheet),
                metadata={"from_combined": True}
            ),
            "income_statement": ExtractionResult(
                extractor_name="income_statement", 
                content=self.income_statement,
                success=bool(self.income_statement),
                metadata={"from_combined": True}
            ),
            "cash_flow": ExtractionResult(
                extractor_name="cash_flow",
                content=self.cash_flow,
                success=bool(self.cash_flow),
                metadata={"from_combined": True}
            ),
        }


class FinancialTablesExtractor(BaseExtractor):
    """
    Extracts all 3 main financial statement tables in one call.
    """
    
    EXTRACTOR_NAME = "financial_tables"
    
    # Markers for parsing the response
    BS_MARKER = "===BALANCE_SHEET_START==="
    BS_END = "===BALANCE_SHEET_END==="
    PL_MARKER = "===INCOME_STATEMENT_START==="
    PL_END = "===INCOME_STATEMENT_END==="
    CF_MARKER = "===CASH_FLOW_START==="
    CF_END = "===CASH_FLOW_END==="
    
    def get_system_prompt(self) -> str:
        return f"""Bạn là chuyên gia trích xuất dữ liệu tài chính từ báo cáo.
Nhiệm vụ: Tìm và trích xuất 3 BÁO CÁO TÀI CHÍNH CHÍNH từ văn bản:
1. BẢNG CÂN ĐỐI KẾ TOÁN (Balance Sheet)
2. BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH (Income Statement)
3. BÁO CÁO LƯU CHUYỂN TIỀN TỆ (Cash Flow Statement)

Quy tắc:
1. Trích xuất từng báo cáo riêng biệt
2. Bao gồm TẤT CẢ các trang của mỗi báo cáo (có thể có "tiếp theo")
3. Giữ nguyên định dạng markdown gốc
4. Sử dụng markers để phân tách:

{self.BS_MARKER}
[Nội dung Bảng cân đối kế toán]
{self.BS_END}

{self.PL_MARKER}
[Nội dung Báo cáo kết quả hoạt động kinh doanh]
{self.PL_END}

{self.CF_MARKER}
[Nội dung Báo cáo lưu chuyển tiền tệ]
{self.CF_END}

5. Nếu một báo cáo không tìm thấy, để trống giữa markers"""
    
    def get_prompt(self) -> str:
        return """Tìm và trích xuất 3 BÁO CÁO TÀI CHÍNH CHÍNH từ văn bản sau.
Sử dụng markers để phân tách từng báo cáo.

VĂN BẢN:
{markdown}

KẾT QUẢ (với markers):"""
    
    def extract_combined(self, markdown: str) -> FinancialTablesResult:
        """
        Extract all 3 tables and return structured result.
        """
        result = self.extract(markdown)
        
        if not result.success:
            return FinancialTablesResult(
                success=False,
                error=result.error or "Extraction failed"
            )
        
        # Parse the response using markers
        content = result.content
        
        bs = self._extract_between_markers(content, self.BS_MARKER, self.BS_END)
        pl = self._extract_between_markers(content, self.PL_MARKER, self.PL_END)
        cf = self._extract_between_markers(content, self.CF_MARKER, self.CF_END)
        
        return FinancialTablesResult(
            balance_sheet=bs,
            income_statement=pl,
            cash_flow=cf,
            success=True,
            metadata={
                "bs_found": bool(bs),
                "pl_found": bool(pl),
                "cf_found": bool(cf),
            }
        )
    
    async def extract_combined_async(self, markdown: str) -> FinancialTablesResult:
        """
        Extract all 3 tables asynchronously and return structured result.
        """
        result = await self.extract_async(markdown)
        
        if not result.success:
            return FinancialTablesResult(
                success=False,
                error=result.error or "Extraction failed"
            )
        
        # Parse the response using markers
        content = result.content
        
        bs = self._extract_between_markers(content, self.BS_MARKER, self.BS_END)
        pl = self._extract_between_markers(content, self.PL_MARKER, self.PL_END)
        cf = self._extract_between_markers(content, self.CF_MARKER, self.CF_END)
        
        return FinancialTablesResult(
            balance_sheet=bs,
            income_statement=pl,
            cash_flow=cf,
            success=True,
            metadata={
                "bs_found": bool(bs),
                "pl_found": bool(pl),
                "cf_found": bool(cf),
            }
        )
    
    def _extract_between_markers(self, text: str, start: str, end: str) -> str:
        """Extract text between start and end markers."""
        try:
            start_idx = text.find(start)
            end_idx = text.find(end)
            
            if start_idx == -1 or end_idx == -1:
                return ""
            
            content = text[start_idx + len(start):end_idx].strip()
            return content
        except Exception:
            return ""
