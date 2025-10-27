from state import StockReportState

def generate_final_response_node(state: StockReportState) -> StockReportState:
    """Tạo câu trả lời cuối cùng dựa trên kết quả thu thập được."""
    print("Bắt đầu Node: Tạo Phản hồi Cuối cùng")
    
    collected = state.get("collected_links", {})
    context = state.get("comparison_context", "")
    notification = state.get("notification")
    
    if not collected and notification:
        final_response = (
            f"Mục tiêu: {context}\n\n"
            f"Trạng thái:\n{notification}"
        )
        return {**state, "final_response": final_response}
        
    if not collected and not notification:
        return {**state, "final_response": "Rất tiếc, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại với một truy vấn khác."}

    response_parts = []
    response_parts.append(f"Mục tiêu phân tích: {context}")

    if notification:
        response_parts.append(f"\nThông báo:\n{notification}")

    if collected:
        response_parts.append("\nKết quả tìm kiếm:")
        for req_id, result in collected.items():
            if "LỖI" in result:
                response_parts.append(f"Yêu cầu {req_id}: Thất bại. {result}")
            else:
                response_parts.append(f"Yêu cầu {req_id}: Thành công. Link: {result}")
    
    return {**state, "final_response": "\n".join(response_parts)}