from typing import Literal
from state import StockReportState
from logger import get_logger

logger = get_logger(__name__)

def should_continue_extraction(state: StockReportState) -> Literal["continue", "end_extraction"]:
    """Kiểm tra xem còn yêu cầu nào trong danh sách chờ không."""
    pending = state.get("pending_requests", [])
    result = "continue" if pending else "end_extraction"
    logger.debug(f"Router (Loop): {len(pending) if pending else 0} pending requests -> {result}")
    return result

def check_extraction_result(state: StockReportState) -> Literal["ask_user", "collect"]:
    """Kiểm tra kết quả của node trích xuất để quyết định nhánh đi tiếp theo."""
    if state.get("clarification_prompt"):
        logger.debug("Router (Result): Needs user clarification")
        return "ask_user"
    else:
        logger.debug("Router (Result): Proceeding to collect")
        return "collect"