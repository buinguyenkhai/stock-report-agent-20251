from state import StockReportState

def ask_user_for_clarification_node(state: StockReportState) -> StockReportState:
    """Hiển thị prompt và chờ người dùng nhập liệu."""
    print("Bắt đầu Node: Hỏi người dùng")
    prompt = state.get("clarification_prompt")
    choices = state.get("possible_choices")
    if not prompt or not choices:
        return {**state, "error_message": "Lỗi logic: Thiếu prompt hoặc lựa chọn để hỏi người dùng."}
    print(prompt)

    while True:
        try:
            choice_idx = int(input("Vui lòng nhập lựa chọn của bạn (số): ")) - 1
            if 0 <= choice_idx < len(choices):
                selected_choice = choices[choice_idx]
                print(f"Bạn đã chọn: {selected_choice['title']}")
                # Cập nhật state với link đã được giải quyết
                return {
                    **state,
                    "report_link": selected_choice["link"],
                    "clarification_prompt": None,
                    "possible_choices": None
                }
            else:
                print(f"Lựa chọn không hợp lệ. Vui lòng chọn một số từ 1 đến {len(choices)}.")
        except ValueError:
            print("Vui lòng chỉ nhập số.")