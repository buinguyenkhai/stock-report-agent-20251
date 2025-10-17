import json
from typing import TypedDict, Optional, Literal
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from playwright.sync_api import sync_playwright

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
    report_type: Optional[str] 
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
    """
    Node để trích xuất link PDF
    """
    print(f"Bắt đầu Node: Trích xuất link cho {state['stock_code']}")
    output_state = {"report_link": None, "error_message": None, "confirmation_prompt": None, "notification": None}
    stock_code = state["stock_code"]
    year = state["year"]
    quarter = state["quarter"]
    user_report_type = state.get("report_type")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            url = f"https://finance.vietstock.vn/{stock_code.upper()}/tai-tai-lieu.htm?doctype=1"
            page.goto(url, wait_until="domcontentloaded", timeout=60000)

            year_selector = 'select.dropdown-year'
            page.wait_for_selector(year_selector, timeout=15000) 
            page.select_option(year_selector, str(year))

            page.wait_for_selector("div.p-t-xs p.i-b-d", timeout=10000)

            # Extract all report elements
            reports_data = page.query_selector_all("div.p-t-xs p.i-b-d")

            scraped_reports = []
            for row in reports_data:
                title_element = row.query_selector("a")
                if title_element:
                    title = title_element.inner_text().strip()
                    link = title_element.get_attribute('href')
                    if link and not link.startswith('http'):
                        link = "https://finance.vietstock.vn" + link
                    scraped_reports.append({"title": title, "link": link})
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
        ("system", "Bạn là một trợ lý AI phân tích tài chính chính xác. Nhiệm vụ của bạn là tìm báo cáo phù hợp nhất trong một danh sách dựa trên các tiêu chí cho trước."),
        ("human", """
Dựa vào tiêu chí tìm kiếm sau: {search_criteria}
Và danh sách các báo cáo có sẵn: {report_list_str}
Hãy phân tích và chọn ra báo cáo phù hợp nhất.
- Nếu tìm thấy báo cáo khớp chính xác yêu cầu, hãy chọn nó.
- Nếu không có báo cáo chính xác nhưng có báo cáo thay thế (ví dụ: hỏi quý 2 có báo cáo 6 tháng), hãy chọn báo cáo thay thế đó.
- Nếu không có gì phù hợp, hãy chỉ ra là không tìm thấy.
"""),
    ])
    
    chain = prompt_template | llm

    quarter = state["quarter"]
    user_report_type = state.get("report_type")
    search_criteria = f"Quý {quarter}, Năm {year}."
    if user_report_type:
        search_criteria += f" Yêu cầu loại báo cáo cụ thể: '{user_report_type}'."
    else:
        search_criteria += " Ưu tiên tìm theo thứ tự: Soát xét -> Kiểm toán -> Báo cáo thường."
    
    report_list_str = "\n".join([f'"{r["title"]}" | "{r["link"]}"' for r in scraped_reports])
    
    try:
        response = chain.invoke({
            "search_criteria": search_criteria,
            "report_list_str": report_list_str
        })

        if USE_OLLAMA:
            result_data = json.loads(response.content)
            result = ReportSelection(**result_data) # Validate output
        else:
            result = response

        if result.match_type == "exact":
            output_state["report_link"] = result.selected_link
            if not user_report_type and result.selected_title:
                found_type = "Không xác định"
                if "soát xét" in result.selected_title.lower():
                    found_type = "Soát xét"
                elif "kiểm toán" in result.selected_title.lower():
                    found_type = "Kiểm toán"
                output_state["notification"] = f"Đã tìm thấy báo cáo phù hợp (Loại: {found_type})."
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


graph = StateGraph(StockReportState)
graph.add_node("extract_report_link", extract_report_link_node)
graph.add_edge(START, "extract_report_link")
graph.add_edge("extract_report_link", END)
agent = graph.compile()
result = agent.invoke({"stock_code": "VIC",
              "year":2023, 
              "quarter":1})

with open("result.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
    

# Cần thêm các conditional Edge:
# Nếu report_link -> Đi đến Node tiếp theo (Xử lý PDF)
# Nếu confirmation_prompt -> Đi đến Node xác nhận lại với người dùng
# Nếu error_message -> Kết thúc luồng và báo lỗi cho người dùng