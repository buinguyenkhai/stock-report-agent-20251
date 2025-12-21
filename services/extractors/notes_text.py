from .base import BaseExtractor


class NotesTextExtractor(BaseExtractor):
    """Extracts narrative text from notes/explanations."""
    
    EXTRACTOR_NAME = "notes_text"
    
    def get_system_prompt(self) -> str:
        return """Bạn là chuyên gia trích xuất dữ liệu tài chính từ báo cáo.
Nhiệm vụ: Tìm và trích xuất PHẦN VĂN BẢN của THUYẾT MINH BÁO CÁO TÀI CHÍNH.

Quy tắc:
1. Chỉ trích xuất phần văn bản giải thích, không lấy bảng số liệu
2. Bao gồm các mô tả về chính sách kế toán, phương pháp, giải trình
3. Giữ nguyên định dạng và cấu trúc
4. Không thêm giải thích hay chú thích
5. Nếu không tìm thấy, trả về "Không tìm thấy Thuyết minh"

Các phần thường có trong Thuyết minh:
- Thông tin chung về doanh nghiệp
- Chính sách kế toán áp dụng
- Giải trình các khoản mục
- Thông tin bổ sung"""
    
    def get_prompt(self) -> str:
        return """Tìm và trích xuất PHẦN VĂN BẢN của THUYẾT MINH BÁO CÁO TÀI CHÍNH từ văn bản sau.
Chỉ lấy phần giải thích văn bản, KHÔNG lấy các bảng số liệu chi tiết.

VĂN BẢN:
{markdown}

THUYẾT MINH (phần văn bản):"""
