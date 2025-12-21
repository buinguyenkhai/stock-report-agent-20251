"""
Balance Sheet Extractor

Extracts the Balance Sheet (Bảng cân đối kế toán) tables from OCR markdown.
"""

from .base import BaseExtractor


class BalanceSheetExtractor(BaseExtractor):
    """Extracts Balance Sheet tables from financial report."""
    
    EXTRACTOR_NAME = "balance_sheet"
    
    def get_system_prompt(self) -> str:
        return """Bạn là chuyên gia trích xuất dữ liệu tài chính từ báo cáo.
Nhiệm vụ: Tìm và trích xuất BẢNG CÂN ĐỐI KẾ TOÁN (Balance Sheet) từ văn bản.

Quy tắc:
1. Chỉ trích xuất phần BẢNG CÂN ĐỐI KẾ TOÁN, không lấy các bảng khác
2. Bao gồm TẤT CẢ các bảng liên quan (có thể có nhiều trang, tiếp theo...)
3. Giữ nguyên định dạng markdown gốc của bảng
4. Không thêm giải thích hay chú thích
5. Nếu không tìm thấy, trả về "Không tìm thấy Bảng cân đối kế toán"

Các tên thường gặp:
- BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT
- BẢNG CÂN ĐỐI KẾ TOÁN
- BÁO CÁO TÌNH HÌNH TÀI CHÍNH (cho ngân hàng)
- BALANCE SHEET
- B01-DN"""
    
    def get_prompt(self) -> str:
        return """Tìm và trích xuất BẢNG CÂN ĐỐI KẾ TOÁN từ văn bản sau.
Trả về nguyên văn các bảng markdown, bao gồm cả phần tiếp theo nếu có.

VĂN BẢN:
{markdown}

BẢNG CÂN ĐỐI KẾ TOÁN:"""
