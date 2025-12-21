"""
Cash Flow Extractor

Extracts the Cash Flow Statement (Báo cáo lưu chuyển tiền tệ) tables from OCR markdown.
"""

from .base import BaseExtractor


class CashFlowExtractor(BaseExtractor):
    """Extracts Cash Flow Statement tables from financial report."""
    
    EXTRACTOR_NAME = "cash_flow"
    
    def get_system_prompt(self) -> str:
        return """Bạn là chuyên gia trích xuất dữ liệu tài chính từ báo cáo.
Nhiệm vụ: Tìm và trích xuất BÁO CÁO LƯU CHUYỂN TIỀN TỆ (Cash Flow Statement) từ văn bản.

Quy tắc:
1. Chỉ trích xuất phần BÁO CÁO LƯU CHUYỂN TIỀN TỆ, không lấy các bảng khác
2. Bao gồm TẤT CẢ các bảng liên quan (có thể có nhiều trang, tiếp theo...)
3. Giữ nguyên định dạng markdown gốc của bảng
4. Không thêm giải thích hay chú thích
5. Nếu không tìm thấy, trả về "Không tìm thấy Báo cáo lưu chuyển tiền tệ"

Các tên thường gặp:
- BÁO CÁO LƯU CHUYỂN TIỀN TỆ HỢP NHẤT
- BÁO CÁO LƯU CHUYỂN TIỀN TỆ
- LƯU CHUYỂN TIỀN TỆ
- CASH FLOW STATEMENT
- B03-DN

Lưu ý: Báo cáo này thường có 3 phần chính:
- I. Lưu chuyển tiền từ hoạt động kinh doanh
- II. Lưu chuyển tiền từ hoạt động đầu tư
- III. Lưu chuyển tiền từ hoạt động tài chính"""
    
    def get_prompt(self) -> str:
        return """Tìm và trích xuất BÁO CÁO LƯU CHUYỂN TIỀN TỆ từ văn bản sau.
Trả về nguyên văn các bảng markdown, bao gồm cả phần tiếp theo nếu có.

VĂN BẢN:
{markdown}

BÁO CÁO LƯU CHUYỂN TIỀN TỆ:"""
