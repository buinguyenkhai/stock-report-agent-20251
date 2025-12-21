from .base import BaseExtractor


class NotesTablesExtractor(BaseExtractor):
    """Extracts detailed tables from notes/explanations."""
    
    EXTRACTOR_NAME = "notes_tables"
    
    def get_system_prompt(self) -> str:
        return """Bạn là chuyên gia trích xuất dữ liệu tài chính từ báo cáo.
Nhiệm vụ: Tìm và trích xuất CÁC BẢNG SỐ LIỆU CHI TIẾT trong THUYẾT MINH BÁO CÁO TÀI CHÍNH.

Quy tắc:
1. Chỉ trích xuất các bảng số liệu, không lấy phần văn bản giải thích
2. Bao gồm các bảng chi tiết như:
   - Chi tiết các khoản phải thu
   - Chi tiết hàng tồn kho
   - Chi tiết tài sản cố định
   - Chi tiết các khoản vay
   - Chi tiết doanh thu
   - v.v.
3. Giữ nguyên định dạng markdown của bảng
4. Không thêm giải thích hay chú thích
5. Nếu không tìm thấy, trả về "Không tìm thấy bảng chi tiết"

Lưu ý: Đây là các bảng NẰM TRONG phần Thuyết minh, KHÔNG phải 3 bảng chính 
(Cân đối kế toán, Kết quả kinh doanh, Lưu chuyển tiền tệ)"""
    
    def get_prompt(self) -> str:
        return """Tìm và trích xuất CÁC BẢNG SỐ LIỆU CHI TIẾT trong THUYẾT MINH từ văn bản sau.
Chỉ lấy các bảng, KHÔNG lấy phần văn bản giải thích.

VĂN BẢN:
{markdown}

BẢNG CHI TIẾT THUYẾT MINH:"""
