from pydantic_models import AnalysisIntent, ReportRequest
from state import StockReportState
from tools import get_current_time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from datetime import datetime

def process_query_node(state: StockReportState) -> StockReportState:
    print("Bắt đầu Node: Xử lý Query")
    query = state["query"]

    tools = [get_current_time]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)
    llm_with_tools = llm.with_structured_output(AnalysisIntent)

    system_prompt = """Bạn là một chuyên gia phân tích tài chính thông minh. Nhiệm vụ của bạn là phân tích yêu cầu của người dùng và chia nó thành một danh sách các yêu cầu báo cáo riêng lẻ.

    QUY TẮC:
    1.  Bạn PHẢI trả lời bằng cách gọi hàm `AnalysisIntent` với danh sách TẤT CẢ các báo cáo cần thiết.
    2.  Sử dụng tool `get_current_time` để biết ngày hiện tại. Dựa vào đó, nếu người dùng yêu cầu một báo cáo trong tương lai (ví dụ: hỏi BCTC Quý 4 vào tháng 10), hãy hiểu rằng báo cáo đó chưa tồn tại và KHÔNG đưa nó vào danh sách yêu cầu.
    3.  Nếu người dùng không nói rõ "quý", "6 tháng" hay "cả năm" (ví dụ: "so sánh FPT 2023 và 2024"), hãy giả định họ muốn xem báo cáo "Cả năm".
    4.  Điền vào `comparison_context` một mô tả ngắn gọn về những gì người dùng muốn làm với các báo cáo này.
    """
    # Few-shot examples
    examples = [
        {
            "input": "phân tích bctc của fpt quý 3 năm 2024",
            "output": AnalysisIntent(
                requests=[ReportRequest(stock_code="FPT", year=2024, period="Quý", quarter=3)],
                comparison_context="Phân tích báo cáo tài chính Quý 3 2024 của FPT."
            )
        },
        {
            "input": "so sánh kết quả kinh doanh của VCB và TCB trong quý 1 2024",
            "output": AnalysisIntent(
                requests=[
                    ReportRequest(stock_code="VCB", year=2024, period="Quý", quarter=1),
                    ReportRequest(stock_code="TCB", year=2024, period="Quý", quarter=1)
                ],
                comparison_context="So sánh kết quả kinh doanh của VCB và TCB trong Quý 1 2024."
            )
        },
        {
            "input": "xem giúp mình con HPG quý 1, quý 2 với quý 3 năm 2025 nó tăng trưởng thế nào",
            "output": AnalysisIntent(
                requests=[
                    ReportRequest(stock_code="HPG", year=2025, period="Quý", quarter=1),
                    ReportRequest(stock_code="HPG", year=2025, period="Quý", quarter=2),
                    ReportRequest(stock_code="HPG", year=2025, period="Quý", quarter=3)
                ],
                comparison_context="Phân tích sự tăng trưởng của HPG qua Quý 1, 2, và 3 của năm 2025."
            )
        },
        {
            "input": "lấy cho tôi báo cáo tài chính hợp nhất quý 2 2024 của FPT",
            "output": AnalysisIntent(
                requests=[ReportRequest(stock_code="FPT", year=2024, period="Quý", quarter=2, consolidation_status="Hợp nhất")],
                comparison_context="Phân tích báo cáo tài chính hợp nhất Quý 2 2024 của FPT."
            )
        },
        {
            "input": "tìm báo cáo mới nhất của VNM",
            "output": AnalysisIntent(
                requests=[ReportRequest(stock_code="VNM", period="Mới nhất")],
                comparison_context="Tìm báo cáo tài chính mới nhất của VNM."
            )
        }
    ]

    for example in examples:
        example["output"] = example["output"].model_dump_json(indent=2)

    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}"),
        ("ai", "{output}"),
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("user", "{query}")
    ])

    chain = final_prompt | llm_with_tools

    try:
        analysis_intent = chain.invoke({"query": query})
        # Loại bỏ các báo cáo tương lai
        now = datetime.now()
        valid_requests = []
        future_requests_messages = []

        if not analysis_intent.requests:
            return {
                **state,
                "pending_requests": [],
                "comparison_context": analysis_intent.comparison_context,
                "notification": "Tôi nhận thấy yêu cầu của bạn dành cho một báo cáo trong tương lai và chưa được phát hành. Do đó, không có tác vụ tìm kiếm nào được thực hiện.",
                "collected_links": {}
            }

        for req in analysis_intent.requests:
            if req.year is not None:
                end_month = 12
                if req.period == "Quý" and req.quarter:
                    end_month = req.quarter * 3
                elif req.period == "6 tháng":
                    end_month = 6
                
                report_is_in_future = False
                if req.year > now.year:
                    report_is_in_future = True
                elif req.year == now.year and end_month >= now.month:
                     report_is_in_future = True

                if report_is_in_future:
                    req_str = f"{req.stock_code} {req.period} {req.quarter}/{req.year}" if req.period == "Quý" else f"{req.stock_code} {req.period}/{req.year}"
                    future_requests_messages.append(f"- {req_str}")
                    continue
            
            valid_requests.append(req)
        
        notification = None
        if future_requests_messages:
            notification = "Một số báo cáo bạn yêu cầu chưa đến kỳ phát hành và đã được bỏ qua:\n" + "\n".join(future_requests_messages)

        return {
            **state,
            "pending_requests": valid_requests,
            "comparison_context": analysis_intent.comparison_context,
            "notification": notification,
            "collected_links": {}
        }
    except Exception as e:
        print(f"Lỗi khi xử lý query: {e}")
        return {
            **state,
            "pending_requests": [],
            "collected_links": {},
            "error_message": f"Lỗi nghiêm trọng khi xử lý query: {e}"
        }