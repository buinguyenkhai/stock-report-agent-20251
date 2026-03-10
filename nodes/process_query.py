import runtime_env  # noqa: F401
from pydantic_models import AnalysisIntent, ReportRequest
from state import StockReportState
from tools import get_current_time
from config import settings, QUARTER_END_MONTHS
from logger import get_logger
from services.llm_factory import create_llm_for_task
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from datetime import datetime

logger = get_logger(__name__)

def process_query_node(state: StockReportState) -> StockReportState:
    """Process user query and extract report requests using LLM."""
    logger.info("Bắt đầu Node: Xử lý Query")
    query = state.get("query", "")
    
    if not query or not query.strip():
        logger.warning("Empty query received")
        return {
            **state,
            "pending_requests": [],
            "collected_links": {},
            "error_message": "Truy vấn trống. Vui lòng nhập câu hỏi của bạn."
        }

    tools = [get_current_time]
    # Use llm_model from state if provided, otherwise fallback to settings
    model_to_use = state.get("llm_model") or settings.llm_model
    llm = create_llm_for_task(
        "query_processing", 
        model=model_to_use
    ).bind_tools(tools)
    llm_with_tools = llm.with_structured_output(AnalysisIntent)

    system_prompt = """Bạn là một chuyên gia phân tích tài chính thông minh. Nhiệm vụ của bạn là phân tích yêu cầu của người dùng và chia nó thành một danh sách các yêu cầu báo cáo.

    QUY TẮC:
    1.  Bạn PHẢI trả lời bằng cách gọi hàm `AnalysisIntent`. Do giới hạn UI, danh sách `requests` chỉ được có TỐI ĐA 1 yêu cầu.
    2.  Sử dụng tool `get_current_time` để biết ngày hiện tại. Dựa vào đó, nếu người dùng yêu cầu một báo cáo trong tương lai (ví dụ: hỏi BCTC Quý 4 vào tháng 10), hãy hiểu rằng báo cáo đó chưa tồn tại và KHÔNG đưa nó vào danh sách yêu cầu.
    3.  Nếu người dùng không nói rõ "quý", "6 tháng" hay "cả năm" (ví dụ: "so sánh FPT 2023 và 2024"), hãy giả định họ muốn xem báo cáo "Cả năm".
    4.  Hệ thống UI hiện chỉ hỗ trợ xử lý 1 báo cáo mỗi lần. Nếu người dùng yêu cầu nhiều báo cáo (so sánh nhiều mã / nhiều kỳ), hãy đưa RA 1 yêu cầu đại diện (ưu tiên yêu cầu đầu tiên theo thứ tự người dùng nêu), và mô tả rõ giới hạn này trong `comparison_context`.
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
                requests=[ReportRequest(stock_code="VCB", year=2024, period="Quý", quarter=1)],
                comparison_context="So sánh kết quả kinh doanh của VCB và TCB trong Quý 1 2024 (giới hạn hệ thống: chỉ xử lý 1 báo cáo/lần, ưu tiên VCB)."
            )
        },
        {
            "input": "xem giúp mình con HPG quý 1, quý 2 với quý 3 năm 2025 nó tăng trưởng thế nào",
            "output": AnalysisIntent(
                requests=[ReportRequest(stock_code="HPG", year=2025, period="Quý", quarter=1)],
                comparison_context="Phân tích sự tăng trưởng của HPG qua Quý 1, 2, và 3 của năm 2025 (giới hạn hệ thống: chỉ xử lý 1 báo cáo/lần, ưu tiên Quý 1)."
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
        logger.debug(f"Invoking LLM with query: {query[:100]}...")
        analysis_intent = chain.invoke({"query": query})
        logger.info(f"LLM returned {len(analysis_intent.requests) if analysis_intent.requests else 0} requests")
        
        # Loại bỏ các báo cáo tương lai
        now = datetime.now()
        valid_requests = []
        future_requests_messages = []

        if not analysis_intent.requests:
            logger.warning("No requests extracted from query")
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
                    end_month = QUARTER_END_MONTHS.get(req.quarter, req.quarter * 3)
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
        
        def _req_to_str(req: ReportRequest) -> str:
            code = (req.stock_code or "").upper().strip() or "UNKNOWN"
            cons = f" - {req.consolidation_status}" if req.consolidation_status else ""
            if req.period == "Quý" and req.quarter and req.year:
                return f"{code} Quý {req.quarter}/{req.year}{cons}"
            if req.period and req.year:
                return f"{code} {req.period}/{req.year}{cons}"
            if req.period == "Mới nhất":
                return f"{code} Mới nhất{cons}"
            return f"{code}{cons}"

        notification = None
        if future_requests_messages:
            notification = "Một số báo cáo bạn yêu cầu chưa đến kỳ phát hành và đã được bỏ qua:\n" + "\n".join(future_requests_messages)
            logger.info(f"Filtered out {len(future_requests_messages)} future reports")

        if len(valid_requests) > 1:
            selected = valid_requests[0]
            skipped = valid_requests[1:]
            skipped_preview = ", ".join([_req_to_str(r) for r in skipped[:3]])
            more = "" if len(skipped) <= 3 else f" (+{len(skipped) - 3} khác)"
            limit_msg = (
                "⚠️ Hệ thống hiện chỉ xử lý 1 báo cáo mỗi lần. "
                f"Đã chọn: {_req_to_str(selected)}. "
                f"Bỏ qua {len(skipped)} yêu cầu còn lại: {skipped_preview}{more}."
            )
            notification = (notification + "\n\n" + limit_msg) if notification else limit_msg
            valid_requests = [selected]

        logger.info(f"Successfully processed query with {len(valid_requests)} valid requests")
        return {
            **state,
            "pending_requests": valid_requests,
            "comparison_context": analysis_intent.comparison_context,
            "notification": notification,
            "collected_links": {}
        }
    except Exception as e:
        logger.error(f"Lỗi khi xử lý query: {e}", exc_info=True)
        return {
            **state,
            "pending_requests": [],
            "collected_links": {},
            "error_message": f"Lỗi nghiêm trọng khi xử lý query: {e}"
        }
