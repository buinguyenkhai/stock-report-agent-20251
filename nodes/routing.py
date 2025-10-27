from typing import Literal
from state import StockReportState

def should_continue_extraction(state: StockReportState) -> Literal["continue", "end_extraction"]:
    """Kiểm tra xem còn yêu cầu nào trong danh sách chờ không."""
    print("Router (Loop): Kiểm tra điều kiện lặp")
    return "continue" if state["pending_requests"] else "end_extraction"

def check_extraction_result(state: StockReportState) -> Literal["ask_user", "collect"]:
    """Kiểm tra kết quả của node trích xuất để quyết định nhánh đi tiếp theo."""
    print("Router (Result): Kiểm tra kết quả trích xuất")
    if state.get("clarification_prompt"):
        print("Quyết định: Cần hỏi người dùng.")
        return "ask_user"
    else:
        print("Quyết định: Có thể thu thập kết quả.")
        return "collect"