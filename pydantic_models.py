from pydantic import BaseModel, Field
import uuid
from typing import Optional, Literal, List

class ReportRequest(BaseModel):
    """Yêu cầu tìm một báo cáo tài chính cụ thể."""
    request_id: str = Field(description="Mã định danh duy nhất cho yêu cầu này, ví dụ 'req_1', 'req_2'.",default_factory=lambda: f"req_{uuid.uuid4().hex[:4]}")
    stock_code: str = Field(description="Mã chứng khoán, ví dụ: 'FPT', 'VCB'.")
    year: int = Field(description="Năm của báo cáo.")
    period: Literal["Quý", "6 tháng", "Cả năm"] = Field(description="Kỳ báo cáo.")
    quarter: Optional[int] = Field(description="Quý của báo cáo (chỉ khi period là 'Quý').")

class AnalysisIntent(BaseModel):
    """Ý định phân tích tổng thể của người dùng, bao gồm tất cả các báo cáo cần thiết."""
    requests: List[ReportRequest] = Field(description="Danh sách TẤT CẢ các báo cáo cần thiết để trả lời câu hỏi của người dùng.")
    comparison_context: str = Field(description="Mô tả ngắn gọn mục tiêu so sánh hoặc phân tích là gì, ví dụ 'so sánh kết quả kinh doanh' hoặc 'phân tích các chỉ số chính'.")