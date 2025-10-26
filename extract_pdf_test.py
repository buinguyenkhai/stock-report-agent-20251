import json
import uuid
import regex as re
from datetime import datetime
from dotenv import load_dotenv
from typing import TypedDict, Optional, Literal, List, Dict

from pydantic import BaseModel, Field
from playwright.sync_api import sync_playwright
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
load_dotenv()

# Pydantic Models

class ReportRequest(BaseModel):
    """Yêu cầu tìm một báo cáo tài chính cụ thể."""
    request_id: str = Field(description="Mã định danh duy nhất cho yêu cầu này, ví dụ 'req_1', 'req_2'.",default_factory=lambda: f"req_{uuid.uuid4().hex[:4]}")
    stock_code: str = Field(description="Mã chứng khoán, ví dụ: 'FPT', 'VCB'.")
    year: int = Field(description="Năm của báo cáo.")
    period: Literal["Quý", "6 tháng", "Cả năm"] = Field(description="Kỳ báo cáo.")
    quarter: Optional[int] = Field(description="Quý của báo cáo (chỉ khi period là 'Quý').")

class AnalysisIntent(BaseModel):
    """Ý định phân tích tổng thể của người dùng, bao gồm tất cả các báo cáo cần thiết."""
    requests: List[ReportRequest] = Field(description="Danh sách TẤT CẢ các báo cáo cần thiết để trả lời câu hỏi của người dùng.")
    comparison_context: str = Field(description="Mô tả ngắn gọn mục tiêu so sánh hoặc phân tích là gì, ví dụ 'so sánh kết quả kinh doanh' hoặc 'phân tích các chỉ số chính'.")

# Tools

@tool
def get_current_time() -> str:
    """Trả về ngày và tháng hiện tại để giúp LLM suy luận về sự tồn tại của các báo cáo theo quý."""
    return datetime.now().strftime("Hôm nay là ngày %d tháng %m năm %Y")

# State của agent
class StockReportState(TypedDict):
    query: str
    pending_requests: List[ReportRequest]
    collected_links: Dict[str, str]
    comparison_context: str
    current_request_id: Optional[str]
    stock_code: Optional[str]
    year: Optional[int]
    period: Optional[Literal["Quý", "6 tháng", "Cả năm"]]
    quarter: Optional[int]
    consolidation_status: Optional[Literal["Hợp nhất", "Công ty mẹ"]]
    report_link: Optional[str]
    error_message: Optional[str]
    clarification_prompt: Optional[str]
    possible_choices: Optional[List[dict]]
    notification: Optional[str]
    final_response: Optional[str]

# Nodes

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
        print("LLM đã phân tích thành công ý định của người dùng:")
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

def prepare_next_extraction_node(state: StockReportState) -> StockReportState:
    """Lấy yêu cầu tiếp theo từ danh sách chờ và cập nhật State."""
    print("Bắt đầu Node: Chuẩn bị Trích xuất")
    pending = state["pending_requests"]
    if not pending:
        return {**state, "error_message": "Không có yêu cầu nào đang chờ."}
    next_request = pending.pop(0) # Lấy và xóa yêu cầu đầu tiên
    print(f"Đang xử lý yêu cầu: {next_request.request_id} - {next_request.stock_code} {next_request.period} {next_request.quarter}/{next_request.year}")

    return {
        **state,
        "pending_requests": pending,
        "current_request_id": next_request.request_id,
        "stock_code": next_request.stock_code,
        "year": next_request.year,
        "period": next_request.period,
        "quarter": next_request.quarter,
        # Reset các trường kết quả của lần lặp trước
        "report_link": None,
        "error_message": None,
        "clarification_prompt": None,
        "notification": None,
    }

def extract_report_link_node(state: StockReportState) -> StockReportState:
    """Node để trích xuất link PDF."""
    print(f"Bắt đầu Node: Trích xuất link cho {state['stock_code']}")
    # Khởi tạo
    stock_code = state["stock_code"]
    year = state["year"]
    output_state = { "report_link": None, "error_message": None, "clarification_prompt": None, "notification": None }
    try:
        with sync_playwright() as p:
            # Truy cập vietstock
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            url = f"https://finance.vietstock.vn/{stock_code.upper()}/tai-tai-lieu.htm?doctype=1"
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            # Chọn năm
            year_selector = 'select.dropdown-year'
            page.wait_for_selector(year_selector, timeout=30000) 
            page.select_option(year_selector, str(year))
            page.wait_for_function("""
                        (year) => {
                            const firstReport = document.querySelector("div.p-t-xs p.i-b-d a");
                            return firstReport && firstReport.innerText.includes(year);
                        }
                    """, arg=str(year), timeout=60000)
            reports_data = page.query_selector_all("div.p-t-xs p.i-b-d")
            # Lấy tên và link của tất cả pdf báo cáo trong năm
            scraped_reports = []
            for row in reports_data:
                title_element = row.query_selector("a")
                if title_element:
                    title = title_element.inner_text().strip()
                    link = title_element.get_attribute('href')
                    if link and not link.startswith('http'):
                        link = "https://finance.vietstock.vn" + link
                    # Bỏ thời gian tạo
                    cleaned_title = re.sub(r'\s*\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}\s*$', '', title)
                    scraped_reports.append({"title": cleaned_title, "link": link})
            browser.close()
            if not scraped_reports:
                output_state["error_message"] = f"Không tìm thấy báo cáo nào cho mã {stock_code} năm {year}."
                return {**state, **output_state}
    except Exception as e:
        output_state["error_message"] = f"Lỗi khi scraping web: {str(e)}"
        return {**state, **output_state}
    available_reports = {
        "Cả năm": {"Hợp nhất": [], "Công ty mẹ": []},
        "6 tháng": {"Hợp nhất": [], "Công ty mẹ": []},
        "Quý": {
            1: {"Hợp nhất": [], "Công ty mẹ": []}, 2: {"Hợp nhất": [], "Công ty mẹ": []},
            3: {"Hợp nhất": [], "Công ty mẹ": []}, 4: {"Hợp nhất": [], "Công ty mẹ": []}
        }
    }

    for report in scraped_reports:
        title_lower = report["title"].lower()
        consol_status = "Hợp nhất" if "hợp nhất" in title_lower else "Công ty mẹ"

        if "kiểm toán" in title_lower: # Báo cáo năm
            available_reports["Cả năm"][consol_status].append(report)
        elif "soát xét" in title_lower: # Báo cáo 6 tháng
            available_reports["6 tháng"][consol_status].append(report)
        elif "quý" in title_lower: # Báo cáo quý
            for q in range(1, 5):
                if f"quý {q}" in title_lower:
                    available_reports["Quý"][q][consol_status].append(report)
                    break

    user_period = state.get("period")
    user_consol_status = state.get("consolidation_status")
    user_quarter = state.get("quarter")

    # Trường hợp 1: Người dùng đã cung cấp đủ thông tin
    if user_period and user_consol_status:
        found_reports = []
        if user_period == "Quý" and user_quarter:
            found_reports = available_reports["Quý"][user_quarter][user_consol_status]
        elif user_period == "6 tháng":
            found_reports = available_reports["6 tháng"][user_consol_status]
        elif user_period == "Cả năm":
            found_reports = available_reports["Cả năm"][user_consol_status]

        if found_reports:
            selected_report = found_reports[0]
            output_state["report_link"] = selected_report["link"]
            output_state["notification"] = f"Đã tìm thấy báo cáo '{selected_report['title']}' theo yêu cầu."
            return {**state, **output_state}
        else:
            req_str = f"{user_period} Quý {user_quarter}" if user_period == "Quý" else user_period
            output_state["error_message"] = f"Không tìm thấy báo cáo '{req_str} - {user_consol_status}' bạn yêu cầu."
            return {**state, **output_state}

    # Trường hợp 2: Agent tự tìm và hỏi lại
    possible_choices = []
    requested_quarter_failed = False 

    # Tìm chính xác quý người dùng yêu cầu (nếu họ yêu cầu quý)
    if user_period == "Quý" and user_quarter:
        for cons_stat in ["Hợp nhất", "Công ty mẹ"]:
            if available_reports["Quý"][user_quarter][cons_stat]:
                report_item = available_reports["Quý"][user_quarter][cons_stat][0]
                possible_choices.append({"period": "Quý", "quarter": user_quarter, "consolidation_status": cons_stat, **report_item})

        # Nếu không tìm thấy quý yêu cầu
        if not possible_choices:
            requested_quarter_failed = True
            # Bắt đầu tìm lùi từ quý trước đó
            for q_fallback in range(user_quarter - 1, 0, -1):
                found_in_fallback_quarter = False
                for cons_stat in ["Hợp nhất", "Công ty mẹ"]:
                    if available_reports["Quý"][q_fallback][cons_stat]:
                        report_item = available_reports["Quý"][q_fallback][cons_stat][0]
                        possible_choices.append({"period": "Quý", "quarter": q_fallback, "consolidation_status": cons_stat, **report_item})
                        found_in_fallback_quarter = True
                if found_in_fallback_quarter:
                    break 

    # Nếu không có lựa chọn nào từ quý (hoặc người dùng không hỏi quý) thì tìm "6 tháng" và "Cả năm"
    if not possible_choices:
        if not requested_quarter_failed: 
            for cons_stat in ["Hợp nhất", "Công ty mẹ"]:
                if user_period != "Cả năm" and available_reports["6 tháng"][cons_stat]:
                    report_item = available_reports["6 tháng"][cons_stat][0]
                    possible_choices.append({"period": "6 tháng", "consolidation_status": cons_stat, **report_item})

                if user_period != "6 tháng" and available_reports["Cả năm"][cons_stat]:
                    report_item = available_reports["Cả năm"][cons_stat][0]
                    possible_choices.append({"period": "Cả năm", "consolidation_status": cons_stat, **report_item})

    if len(possible_choices) == 0:
        if requested_quarter_failed:
             output_state["error_message"] = f"Không tìm thấy báo cáo Quý {user_quarter} và các quý trước đó cho năm {year}."
        else:
             output_state["error_message"] = f"Không tìm thấy báo cáo tài chính nào phù hợp cho năm {year}."

    elif len(possible_choices) == 1:
        selected = possible_choices[0]
        output_state["report_link"] = selected["link"]
        notification_text = f"Chỉ tìm thấy một báo cáo phù hợp duy nhất: '{selected['title']}'. Hệ thống sẽ tự động xử lý."
        if requested_quarter_failed:
             notification_text = f"Không tìm thấy báo cáo Quý {user_quarter}. " + notification_text
        output_state["notification"] = notification_text
        output_state["possible_choices"] = possible_choices
    else: 
        prompt_text = ""
        if requested_quarter_failed:
            prompt_text = f"Không tìm thấy báo cáo cho Quý {user_quarter} Năm {year}. \nTuy nhiên, tôi đã tìm thấy các báo cáo gần nhất sau:\n"
        else:
            prompt_text = f"Tôi đã tìm thấy các báo cáo sau cho {stock_code.upper()} năm {year}:\n"

        for i, choice in enumerate(possible_choices):
            prompt_text += f"{i+1}. {choice['title']}\n"
        prompt_text += "Bạn muốn tôi phân tích báo cáo nào?"
        output_state["clarification_prompt"] = prompt_text
        output_state["possible_choices"] = possible_choices
    return {**state, **output_state}

def ask_user_for_clarification_node(state: StockReportState) -> StockReportState:
    """Hiển thị prompt và chờ người dùng nhập liệu."""
    print("Bắt đầu Node: Hỏi người dùng")
    prompt = state.get("clarification_prompt")
    choices = state.get("possible_choices")
    if not prompt or not choices:
        return {**state, "error_message": "Lỗi logic: Thiếu prompt hoặc lựa chọn để hỏi người dùng."}
    print("YÊU CẦU NHẬP LIỆU")
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

# Routers
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

# Graph

graph_builder = StateGraph(StockReportState)

# Nodes

graph_builder.add_node("process_query", process_query_node)
graph_builder.add_node("prepare_next_extraction", prepare_next_extraction_node)
graph_builder.add_node("extract_report_link", extract_report_link_node)
graph_builder.add_node("ask_user", ask_user_for_clarification_node)
graph_builder.add_node("collect_result", collect_result_node)

# Edges

graph_builder.add_edge(START, "process_query")
graph_builder.add_conditional_edges(
    "process_query",
    should_continue_extraction,
    {"continue": "prepare_next_extraction", "end_extraction": END}
)
graph_builder.add_conditional_edges(
    "collect_result",
    should_continue_extraction,
    {"continue": "prepare_next_extraction", "end_extraction": END}
)
graph_builder.add_edge("prepare_next_extraction", "extract_report_link")
graph_builder.add_edge("ask_user", "collect_result")
graph_builder.add_conditional_edges(
    "extract_report_link",
    check_extraction_result,
    {"ask_user": "ask_user", "collect": "collect_result"}
)

agent = graph_builder.compile()

query = "so sánh fpt quý 2 và quý 3 năm 2024"
final_state = agent.invoke({"query": query})

print("AGENT ĐÃ HOÀN TẤT")
with open("result.json", 'w', encoding='utf-8') as f:
    json.dump(final_state, f, ensure_ascii=False, indent=4)

# Lưu graph
with open("graph.png", "wb") as f:
    f.write(agent.get_graph().draw_mermaid_png())