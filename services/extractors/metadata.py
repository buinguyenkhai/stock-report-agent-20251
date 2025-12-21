from typing import Optional, Literal
from pydantic import BaseModel, Field

from .base import BaseExtractor, ExtractionResult
from services.llm_factory import create_structured_llm
from logger import get_logger

logger = get_logger(__name__)


class ReportMetadata(BaseModel):
    """Structured metadata from financial report."""
    company_name: str = Field(description="Tên công ty/tập đoàn")
    stock_ticker: Optional[str] = Field(default=None, description="Mã chứng khoán (VD: FPT, VIC)")
    
    unit: Literal["VND", "nghìn VND", "triệu VND", "tỷ VND"] = Field(
        description="Đơn vị tiền tệ trong báo cáo"
    )
    
    report_type: Literal["consolidated", "parent"] = Field(
        default="consolidated",
        description="Loại báo cáo: hợp nhất (consolidated) hay công ty mẹ (parent)"
    )
    
    period_type: Literal["quarterly", "annual", "semi_annual"] = Field(
        description="Loại kỳ báo cáo"
    )
    
    quarter: Optional[int] = Field(default=None, description="Quý (1-4) nếu là báo cáo quý")
    year: int = Field(description="Năm báo cáo")
    
    is_ytd: bool = Field(
        default=False,
        description="True nếu số liệu là lũy kế từ đầu năm (Year-to-Date)"
    )
    
    currency: str = Field(default="VND", description="Loại tiền tệ")
    
    auditor: Optional[str] = Field(default=None, description="Công ty kiểm toán")
    audit_opinion: Optional[str] = Field(
        default=None,
        description="Ý kiến kiểm toán: chấp nhận toàn phần, ngoại trừ, từ chối..."
    )


class MetadataExtractor(BaseExtractor):
    """Extracts structured metadata from financial report."""
    
    EXTRACTOR_NAME = "metadata"
    
    def __init__(self, model: Optional[str] = None):
        super().__init__(model)
        self._structured_llm = None
    
    @property
    def structured_llm(self):
        """Lazy-load structured LLM."""
        if self._structured_llm is None:
            from .base import DEFAULT_EXTRACTION_MODEL
            self._structured_llm = create_structured_llm(
                model=self.model or DEFAULT_EXTRACTION_MODEL,
                schema=ReportMetadata,
                temperature=0.0,
            )
        return self._structured_llm
    
    def get_system_prompt(self) -> str:
        return """Bạn là chuyên gia phân tích báo cáo tài chính.
Nhiệm vụ: Trích xuất thông tin metadata từ báo cáo tài chính.

Hướng dẫn:
1. Tìm đơn vị tiền tệ (thường ở đầu mỗi bảng: "Đơn vị: VND", "Đơn vị tính: triệu đồng")
2. Xác định loại báo cáo từ tiêu đề (hợp nhất vs công ty mẹ)
3. Xác định kỳ báo cáo từ tiêu đề (Quý I, II, III, IV, Năm)
4. Tìm tên công ty và mã chứng khoán
5. Xác định có phải số liệu YTD không (thường có cột "Lũy kế từ đầu năm")
6. Tìm thông tin kiểm toán nếu có

Lưu ý về đơn vị:
- "VND", "đồng" → VND
- "nghìn đồng", "1.000 đồng" → nghìn VND
- "triệu đồng", "triệu VND" → triệu VND
- "tỷ đồng", "tỷ VND" → tỷ VND"""
    
    def get_prompt(self) -> str:
        return """Trích xuất metadata từ báo cáo tài chính sau.

VĂN BẢN:
{markdown}"""
    
    def extract(self, markdown: str) -> ExtractionResult:
        """Extract metadata using structured output."""
        try:
            # Only use first 20K chars for metadata (usually in header/first pages)
            sample = markdown[:20000]
            
            messages = [
                ("system", self.get_system_prompt()),
                ("human", self.get_prompt().format(markdown=sample)),
            ]
            
            metadata: ReportMetadata = self.structured_llm.invoke(messages)
            
            logger.info(
                f"Extracted metadata: {metadata.company_name}, "
                f"Unit={metadata.unit}, Period={metadata.period_type}, "
                f"Q{metadata.quarter}/{metadata.year}, YTD={metadata.is_ytd}"
            )
            
            return ExtractionResult(
                extractor_name=self.EXTRACTOR_NAME,
                content=metadata.model_dump_json(indent=2),
                success=True,
                metadata=metadata.model_dump()
            )
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return ExtractionResult(
                extractor_name=self.EXTRACTOR_NAME,
                content="",
                success=False,
                error=str(e)
            )
    
    def extract_structured(self, markdown: str) -> Optional[ReportMetadata]:
        """Extract metadata and return as ReportMetadata object."""
        result = self.extract(markdown)
        if result.success and result.metadata:
            return ReportMetadata(**result.metadata)
        return None
