import json
from dotenv import load_dotenv
import regex as re

from typing import TypedDict, Optional, Literal
from pydantic import BaseModel, Field
from playwright.sync_api import sync_playwright

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END

USE_OLLAMA = False
load_dotenv()

# State của Graph
class StockReportState(TypedDict):
    stock_code: str
    quarter: int
    year: int
    report_link: Optional[str]
    error_message: Optional[str]
    confirmation_prompt: Optional[str]
    notification: Optional[str]

# Định nghĩa cấu trúc JSON
class ReportSelection(BaseModel):
    """Cấu trúc output cho tác vụ lựa chọn báo cáo."""
    match_type: Literal["exact", "alternative", "none"] = Field(description="Loại kết quả khớp tìm thấy.")
    selected_title: Optional[str] = Field(description="Tiêu đề của báo cáo được chọn.")
    selected_link: Optional[str] = Field(description="URL của báo cáo được chọn.")
    reason: str = Field(description="Giải thích ngắn gọn cho lựa chọn hoặc lý do không tìm thấy.")

def extract_report_link_node(state: StockReportState) -> StockReportState:
    """Node để trích xuất link PDF."""
    print(f"Bắt đầu Node: Trích xuất link cho {state['stock_code']}")
    # Khởi tạo
    output_state = {"report_link": None, "error_message": None, "confirmation_prompt": None, "notification": None}
    stock_code = state["stock_code"]
    year = state["year"]
    quarter = state["quarter"]
    try:
        with sync_playwright() as p:
            # Truy cập vietstock
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            url = f"https://finance.vietstock.vn/{stock_code.upper()}/tai-tai-lieu.htm?doctype=1"
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            # Chọn năm
            year_selector = 'select.dropdown-year'
            page.wait_for_selector(year_selector, timeout=15000) 
            page.select_option(year_selector, str(year))
            # Vị trí file báo cáo pdf
            page.wait_for_selector("div.p-t-xs p.i-b-d", timeout=10000)
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

    if USE_OLLAMA:
        print("Sử dụng mô hình Ollama (local)")
        # Output JSON
        llm = ChatOllama(model="llama3", format="json", temperature=0)
    else:
        print("Sử dụng mô hình Google Gemini (API)")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        # Output JSON
        llm = llm.with_structured_output(ReportSelection)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
    Bạn là một trợ lý AI phân tích tài chính chính xác. Nhiệm vụ của bạn là tìm báo cáo phù hợp nhất trong một danh sách dựa trên Tiêu chí tìm kiếm (Quý và Năm).
    Bạn PHẢI tuân theo quy trình lọc nghiêm ngặt sau:

    BƯỚC 1: LỌC THEO THỜI GIAN
    - Đầu tiên, chỉ tìm các báo cáo khớp chính xác với Quý và Năm trong Tiêu chí tìm kiếm.
    - Ví dụ: Nếu tiêu chí là "Quý 1 Năm 2023", bạn chỉ được phép tìm trong các báo cáo có chứa "Quý 1" và "2023".
    - Nếu không tìm thấy báo cáo nào khớp chính xác (ví dụ: tìm Quý 2 nhưng chỉ có báo cáo 6 tháng), lúc đó bạn mới được xem xét báo cáo thay thế.

    BƯỚC 2: ƯU TIÊN LOẠI BÁO CÁO (CHỈ áp dụng cho các báo cáo đã tìm thấy ở Bước 1)
    Sau khi đã có danh sách lọc ở Bước 1, hãy chọn báo cáo DUY NHẤT theo thứ tự ưu tiên sau:
    1. Ưu tiên 1: Báo cáo có chữ "Soát xét".
    2. Ưu tiên 2: Báo cáo có chữ "Kiểm toán".
    3. Ưu tiên 3: Báo cáo thường (không có "Soát xét" hay "Kiểm toán").

    (Lưu ý: Nếu ở cùng một mức ưu tiên mà có cả "Hợp nhất" và "Công ty mẹ", hãy luôn ưu tiên chọn "Hợp nhất".)

    Nếu không có gì phù hợp, hãy chỉ ra là không tìm thấy.
    """),
        ("human", """
    Dựa vào Tiêu chí tìm kiếm sau: {search_criteria}
    Và danh sách các báo cáo có sẵn:
    {report_list_str}

    Hãy phân tích và chọn ra báo cáo phù hợp nhất theo quy trình đã hướng dẫn.
    """),
    ])
    
    chain = prompt_template | llm
    quarter = state["quarter"]
    search_criteria = f"Quý {quarter} Năm {year}."
    report_list_str = "\n".join([f'"{r["title"]}" | "{r["link"]}"' for r in scraped_reports])
    
    try:
        response = chain.invoke({
            "search_criteria": search_criteria,
            "report_list_str": report_list_str
        })

        if USE_OLLAMA:
            result_data = json.loads(response.content)
            result = ReportSelection(**result_data)
        else:
            result = response

        if result.match_type == "exact":
            output_state["report_link"] = result.selected_link
            if result.selected_title:
                if "soát xét" in result.selected_title.lower():
                    found_type = "Soát xét"
                elif "kiểm toán" in result.selected_title.lower():
                    found_type = "Kiểm toán"
                else:
                    found_type = "Không xác định"
                output_state["notification"] = f"Đã tìm thấy báo cáo phù hợp (Loại: {found_type})."
        # Đề xuất lại người dùng báo cáo khác
        elif result.match_type == "alternative":
            output_state["confirmation_prompt"] = (
                f"Không tìm thấy báo cáo bạn yêu cầu. Tuy nhiên, có một báo cáo khác là: '{result.selected_title}'. "
                f"Bạn có muốn lấy báo cáo này không?"
            )
        else:
            output_state["error_message"] = "Không tìm thấy báo cáo nào phù hợp với yêu cầu."
            
    except Exception as e:
        error_content = ""
        if 'response' in locals():
            error_content = f"Phản hồi nhận được: {response}"
        output_state["error_message"] = f"Lỗi khi xử lý phản hồi từ LLM: {e}. {error_content}"

    return {**state, **output_state}
# Graph Agent
graph = StateGraph(StockReportState)
graph.add_node("extract_report_link", extract_report_link_node)
graph.add_edge(START, "extract_report_link")
graph.add_edge("extract_report_link", END)
agent = graph.compile()
# Ví dụ
result = agent.invoke({"stock_code": "FPT",
              "year":2025, 
              "quarter":3})
# Lưu kết quả Node
with open("result.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
# Lưu graph
with open("graph.png", "wb") as f:
    f.write(agent.get_graph().draw_mermaid_png())

# Soát xét, kiểm toán, thường
# Công ty mẹ, Hợp nhất

# Cần thêm các conditional Edge:
# Nếu report_link -> Đi đến Node tiếp theo (Xử lý PDF)
# Nếu confirmation_prompt -> Đi đến Node xác nhận lại với người dùng
# Nếu error_message -> Kết thúc luồng và báo lỗi cho người dùng