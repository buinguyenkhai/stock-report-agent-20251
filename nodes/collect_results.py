from state import StockReportState
from logger import get_logger

logger = get_logger(__name__)

def collect_result_node(state: StockReportState) -> StockReportState:
    """Lưu kết quả của lần trích xuất vừa rồi vào collected_links."""
    logger.info("Bắt đầu Node: Thu thập Kết quả")

    request_id = state.get("current_request_id", "unknown")
    collected = dict(state.get("collected_links", {}))

    if state.get("report_link"):
        result = state["report_link"]
        logger.info(f"Thành công: Yêu cầu {request_id} -> {result}")
    elif state.get("error_message"):
        result = f"LỖI: {state['error_message']}"
        logger.warning(f"Thất bại: Yêu cầu {request_id} -> {result}")
    else:
        result = "LỖI: Không có link hoặc thông báo lỗi được trả về."
        logger.warning(f"Thất bại: Yêu cầu {request_id} -> {result}")

    collected[request_id] = result
    
    return {
        **state,
        "collected_links": collected
    }