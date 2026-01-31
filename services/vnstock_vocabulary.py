from typing import List, Dict, Literal


# Balance Sheet items (Bảng cân đối kế toán)
BALANCE_SHEET_ITEMS: List[str] = [
    # Assets - Tài sản
    "TÀI SẢN NGẮN HẠN",
    "Tiền và tương đương tiền",
    "Giá trị thuần đầu tư ngắn hạn",
    "Các khoản phải thu ngắn hạn",
    "Trả trước cho người bán ngắn hạn",
    "Phải thu về cho vay ngắn hạn",
    "Hàng tồn kho ròng",
    "Hàng tồn kho, ròng",
    "Tài sản lưu động khác",
    
    "TÀI SẢN DÀI HẠN",
    "Phải thu về cho vay dài hạn",
    "Phải thu dài hạn",
    "Phải thu dài hạn khác",
    "Tài sản cố định",
    "Đầu tư dài hạn",
    "Lợi thế thương mại",
    "Trả trước dài hạn",
    "Tài sản dài hạn khác",
    
    "TỔNG CỘNG TÀI SẢN",
    
    # Liabilities - Nợ phải trả
    "NỢ PHẢI TRẢ",
    "Nợ ngắn hạn",
    "Vay và nợ thuê tài chính ngắn hạn",
    "Người mua trả tiền trước ngắn hạn",
    "Nợ dài hạn",
    "Vay và nợ thuê tài chính dài hạn",
    
    # Equity - Vốn chủ sở hữu
    "VỐN CHỦ SỞ HỮU",
    "Vốn và các quỹ",
    "Vốn góp của chủ sở hữu",
    "Cổ phiếu phổ thông",
    "Các quỹ khác",
    "Quỹ đầu tư và phát triển",
    "Lãi chưa phân phối",
    "Vốn Ngân sách nhà nước và quỹ khác",
    "LỢI ÍCH CỦA CỔ ĐÔNG THIỂU SỐ",
    
    "TỔNG CỘNG NGUỒN VỐN",
]


# Income Statement items (Báo cáo kết quả hoạt động kinh doanh)
INCOME_STATEMENT_ITEMS: List[str] = [
    # Revenue - Doanh thu
    "Doanh thu",
    "Doanh thu bán hàng và cung cấp dịch vụ",
    "Các khoản giảm trừ doanh thu",
    "Doanh thu thuần",
    
    # Cost - Chi phí
    "Giá vốn hàng bán",
    "Lãi gộp",
    
    # Financial - Tài chính
    "Thu nhập tài chính",
    "Chi phí tài chính",
    "Chi phí tiền lãi vay",
    "Lãi/lỗ từ công ty liên doanh",
    "Lãi lỗ trong công ty liên doanh, liên kết",
    
    # Operating expenses - Chi phí hoạt động
    "Chi phí bán hàng",
    "Chi phí quản lý DN",
    
    # Operating profit - Lợi nhuận hoạt động
    "Lãi/Lỗ từ hoạt động kinh doanh",
    
    # Other - Khác
    "Thu nhập khác",
    "Thu nhập/Chi phí khác",
    "Lợi nhuận khác",
    
    # Tax and Net profit - Thuế và lợi nhuận
    "LN trước thuế",
    "Chi phí thuế TNDN hiện hành",
    "Chi phí thuế TNDN hoãn lại",
    "Lợi nhuận thuần",
    
    # Attribution - Phân bổ
    "Cổ đông thiểu số",
    "Cổ đông của Công ty mẹ",
    "Lợi nhuận sau thuế của Cổ đông công ty mẹ",
    
    # Growth metrics (optional) - Chỉ số tăng trưởng
    "Tăng trưởng doanh thu (%)",
    "Tăng trưởng lợi nhuận (%)",
]


# Cash Flow Statement items (Báo cáo lưu chuyển tiền tệ)
CASH_FLOW_ITEMS: List[str] = [
    # Operating activities - Hoạt động kinh doanh
    "Lãi/Lỗ ròng trước thuế",
    "Khấu hao TSCĐ",
    "Dự phòng RR tín dụng",
    "Lãi/Lỗ chênh lệch tỷ giá chưa thực hiện",
    "Lãi/Lỗ từ hoạt động đầu tư",
    "Thu nhập lãi",
    "Lưu chuyển tiền thuần từ HĐKD trước thay đổi VLĐ",
    
    # Working capital changes - Thay đổi vốn lưu động
    "Tăng/Giảm các khoản phải thu",
    "Tăng/Giảm hàng tồn kho",
    "Tăng/Giảm các khoản phải trả",
    "Tăng/Giảm chi phí trả trước",
    
    # Operating cash payments - Chi trả từ hoạt động
    "Chi phí lãi vay đã trả",
    "Tiền thu nhập doanh nghiệp đã trả",
    "Tiền thu khác từ các hoạt động kinh doanh",
    "Tiền chi khác từ các hoạt động kinh doanh",
    "Lưu chuyển tiền tệ ròng từ các hoạt động SXKD",
    
    # Investing activities - Hoạt động đầu tư
    "Mua sắm TSCĐ",
    "Tiền thu được từ thanh lý tài sản cố định",
    "Tiền chi cho vay, mua công cụ nợ của đơn vị khác",
    "Tiền thu hồi cho vay, bán lại các công cụ nợ của đơn vị khác",
    "Đầu tư vào các doanh nghiệp khác",
    "Tiền thu từ việc bán các khoản đầu tư vào doanh nghiệp khác",
    "Tiền thu cổ tức và lợi nhuận được chia",
    "Lưu chuyển từ hoạt động đầu tư",
    
    # Financing activities - Hoạt động tài chính
    "Tăng vốn cổ phần từ góp vốn và/hoặc phát hành cổ phiếu",
    "Chi trả cho việc mua lại, trả cổ phiếu",
    "Tiền thu được các khoản đi vay",
    "Tiền trả các khoản đi vay",
    "Tiền thanh toán vốn gốc đi thuê tài chính",
    "Cổ tức đã trả",
    "Lưu chuyển tiền từ hoạt động tài chính",
    
    # Summary - Tổng hợp
    "Lưu chuyển tiền thuần trong kỳ",
    "Tiền và tương đương tiền",
    "Ảnh hưởng của chênh lệch tỷ giá",
    "Tiền và tương đương tiền cuối kỳ",
]


# All items combined
ALL_ITEMS: List[str] = BALANCE_SHEET_ITEMS + INCOME_STATEMENT_ITEMS + CASH_FLOW_ITEMS


# Statement type mapping
StatementType = Literal["balance_sheet", "income_statement", "cash_flow"]

STATEMENT_ITEMS: Dict[StatementType, List[str]] = {
    "balance_sheet": BALANCE_SHEET_ITEMS,
    "income_statement": INCOME_STATEMENT_ITEMS,
    "cash_flow": CASH_FLOW_ITEMS,
}


def get_vocabulary_prompt_section() -> str:
    """
    Generate a prompt section with all vnstock item names.
    Used in the parser prompt to guide name normalization.
    """
    lines = []
    lines.append("## DANH MỤC CHỈ TIÊU CHUẨN (VNSTOCK)")
    lines.append("")
    lines.append("### Bảng cân đối kế toán:")
    for item in BALANCE_SHEET_ITEMS:
        lines.append(f"- {item}")
    
    lines.append("")
    lines.append("### Báo cáo kết quả hoạt động kinh doanh:")
    for item in INCOME_STATEMENT_ITEMS:
        lines.append(f"- {item}")
    
    lines.append("")
    lines.append("### Báo cáo lưu chuyển tiền tệ:")
    for item in CASH_FLOW_ITEMS:
        lines.append(f"- {item}")
    
    return "\n".join(lines)
