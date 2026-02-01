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


def check_extraction_result(state: StockReportState) -> Literal["ocr_report", "collect"]:
    """Quyết định bước tiếp theo sau khi đã có (hoặc không có) report_link.
    """
    if state.get("error_message") or not state.get("report_link"):
        logger.debug("Router (Result): No report link (error) -> collect")
        return "collect"

    logger.debug("Router (Result): Have report link -> OCR")
    return "ocr_report"
