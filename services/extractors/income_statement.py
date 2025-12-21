"""
Income Statement Extractor

Extracts the Income Statement (Báo cáo kết quả hoạt động kinh doanh) tables from OCR markdown.
"""

from .base import BaseExtractor


class IncomeStatementExtractor(BaseExtractor):
    """Extracts Income Statement tables from financial report."""
    
    EXTRACTOR_NAME = "income_statement"
    
    def get_system_prompt(self) -> str:
        return """Bạn là chuyên gia trích xuất dữ liệu tài chính từ báo cáo.
Nhiệm vụ: Tìm và trích xuất BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH (Income Statement) từ văn bản.

Quy tắc:
1. Chỉ trích xuất phần BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH, không lấy các bảng khác
2. Bao gồm TẤT CẢ các bảng liên quan (có thể có nhiều trang, tiếp theo...)
3. Giữ nguyên định dạng markdown gốc của bảng
4. Không thêm giải thích hay chú thích
5. Nếu không tìm thấy, trả về "Không tìm thấy Báo cáo kết quả hoạt động kinh doanh"

Các tên thường gặp:
- BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH HỢP NHẤT
- BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH
- KẾT QUẢ HOẠT ĐỘNG KINH DOANH
- BÁO CÁO KẾT QUẢ HOẠT ĐỘNG (cho ngân hàng)
- INCOME STATEMENT
- PROFIT AND LOSS
- B02-DN"""
    
    def get_prompt(self) -> str:
        return """Tìm và trích xuất BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH từ văn bản sau.
Trả về nguyên văn các bảng markdown, bao gồm cả phần tiếp theo nếu có.

VĂN BẢN:
{markdown}

BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH:"""
