from pydantic import BaseModel, Field
import uuid
from typing import Optional, Literal, List

class ReportRequest(BaseModel):
    """Yêu cầu tìm một báo cáo tài chính cụ thể."""
    request_id: str = Field(description="Mã định danh duy nhất cho yêu cầu này, ví dụ 'req_1', 'req_2'.",default_factory=lambda: f"req_{uuid.uuid4().hex[:4]}")
    stock_code: str = Field(description="Mã chứng khoán, ví dụ: 'FPT', 'VCB'.")
    year: Optional[int] = Field(default=None, description="Năm của báo cáo.")
    period: Optional[Literal["Quý", "6 tháng", "Cả năm", "Mới nhất"]] = Field(default=None, description="Kỳ báo cáo.")
    quarter: Optional[int] = Field(default=None, description="Quý của báo cáo (chỉ khi period là 'Quý').")
    consolidation_status: Optional[Literal["Hợp nhất", "Công ty mẹ"]] = Field(default=None, description="Loại báo cáo: Hợp nhất hay của Công ty mẹ.")

class AnalysisIntent(BaseModel):
    """Ý định phân tích tổng thể của người dùng, bao gồm tất cả các báo cáo cần thiết."""
    requests: List[ReportRequest] = Field(description="Danh sách TẤT CẢ các báo cáo cần thiết để trả lời câu hỏi của người dùng.")
    comparison_context: str = Field(description="Mô tả ngắn gọn mục tiêu so sánh hoặc phân tích là gì, ví dụ 'so sánh kết quả kinh doanh' hoặc 'phân tích các chỉ số chính'.")

class FinancialItem(BaseModel):
    """Một dòng trong báo cáo tài chính."""
    item_code: Optional[str] = Field(description="Mã số của chỉ tiêu (ví dụ: 110, 01).")
    item_name: str = Field(description="Tên chỉ tiêu (ví dụ: Tiền và các khoản tương đương tiền).")
    value: Optional[float] = Field(description="Giá trị tại thời điểm báo cáo (Cột kỳ này/Cuối kỳ).")
    notes_ref: Optional[str] = Field(description="Thuyết minh (nếu có).")

class FinancialNote(BaseModel):
    """Một mục thuyết minh trong báo cáo tài chính."""
    note_number: Optional[str] = Field(description="Số hiệu thuyết minh (ví dụ: V.01, 5.1).")
    note_title: Optional[str] = Field(description="Tiêu đề thuyết minh (ví dụ: Tiền gửi ngân hàng).")
    content: Optional[str] = Field(description="Nội dung tóm tắt hoặc số liệu chính của thuyết minh này.")

class FinancialReportData(BaseModel):
    """Dữ liệu cấu trúc được trích xuất từ báo cáo tài chính."""
    balance_sheet: List[FinancialItem] = Field(description="Bảng Cân đối kế toán.")
    income_statement: List[FinancialItem] = Field(description="Báo cáo Kết quả hoạt động kinh doanh.")
    cash_flow: List[FinancialItem] = Field(description="Báo cáo Lưu chuyển tiền tệ.")
    notes: List[FinancialNote] = Field(default=[], description="Các mục thuyết minh chính được trích xuất.")