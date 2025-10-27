from pydantic_models import AnalysisIntent, ReportRequest
from state import StockReportState
from tools import get_current_time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

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
        return {
            **state,
            "pending_requests": analysis_intent.requests,
            "comparison_context": analysis_intent.comparison_context,
            "collected_links": {}
        }
    except Exception as e:
        print(f"Lỗi khi xử lý query: {e}")
        return {
            **state,
            "error_message": "Lỗi xử lý query."
        }