from state import StockReportState

def collect_result_node(state: StockReportState) -> StockReportState:
    """Lưu kết quả của lần trích xuất vừa rồi vào collected_links."""
    print("Bắt đầu Node: Thu thập Kết quả")

    request_id = state["current_request_id"]
    collected = state["collected_links"]

    if state.get("report_link"):
        result = state["report_link"]
        print(f"Thành công: Yêu cầu {request_id} -> {result}")
    elif state.get("error_message"):
        result = f"LỖI: {state['error_message']}"
        print(f"Thất bại: Yêu cầu {request_id} -> {result}")
    else:
        result = "LỖI: Không có link hoặc thông báo lỗi được trả về."
        print(f"Thất bại: Yêu cầu {request_id} -> {result}")

    collected[request_id] = result
    
    return {
        **state,
        "collected_links": collected
    }