from .base import BaseExtractor


class OtherTextExtractor(BaseExtractor):
    """Extracts audit opinion, management discussion, and important disclosures."""
    
    EXTRACTOR_NAME = "other_text"
    
    def get_system_prompt(self) -> str:
        return """Bạn là chuyên gia trích xuất dữ liệu tài chính từ báo cáo.
Nhiệm vụ: Tìm và trích xuất CÁC VĂN BẢN QUAN TRỌNG KHÁC trong báo cáo tài chính.

Các phần cần trích xuất:
1. Báo cáo của Ban Giám đốc (Management's Report)
2. Báo cáo kiểm toán độc lập (Independent Auditor's Report)
3. Ý kiến kiểm toán (Auditor's Opinion)
4. Giải trình của Ban điều hành
5. Các sự kiện quan trọng sau kỳ báo cáo
6. Cam kết và nghĩa vụ nợ tiềm tàng
7. Các bên liên quan

Quy tắc:
1. Trích xuất nguyên văn, giữ nguyên định dạng
2. KHÔNG lấy các bảng số liệu chính
3. KHÔNG lấy phần Thuyết minh chi tiết
4. Nếu không tìm thấy, trả về "Không tìm thấy văn bản quan trọng khác"

Tập trung vào các ý kiến, đánh giá, và thông tin định tính quan trọng."""
    
    def get_prompt(self) -> str:
        return """Tìm và trích xuất CÁC VĂN BẢN QUAN TRỌNG KHÁC từ văn bản sau.
Bao gồm: Báo cáo kiểm toán, Ý kiến kiểm toán, Báo cáo của Ban Giám đốc, Giải trình.

VĂN BẢN:
{markdown}

VĂN BẢN QUAN TRỌNG KHÁC:"""
