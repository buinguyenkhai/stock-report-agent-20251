from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from state import StockReportState
import regex as re

def prepare_next_extraction_node(state: StockReportState) -> StockReportState:
    """Lấy yêu cầu tiếp theo từ danh sách chờ và cập nhật State."""
    print("Bắt đầu Node: Chuẩn bị Trích xuất")
    pending = state["pending_requests"]
    if not pending:
        return {**state, "error_message": "Không có yêu cầu nào đang chờ."}
    next_request = pending.pop(0) # Lấy yêu cầu đầu tiên
    print(f"Đang xử lý yêu cầu: {next_request.request_id} - {next_request.stock_code} {next_request.period} {next_request.quarter}/{next_request.year}")

    return {
        **state,
        "pending_requests": pending,
        "current_request_id": next_request.request_id,
        "stock_code": next_request.stock_code,
        "year": next_request.year,
        "period": next_request.period,
        "quarter": next_request.quarter,
        "consolidation_status": next_request.consolidation_status,
        # Reset
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
    period = state["period"]
    user_consol_status = state.get("consolidation_status")
    output_state = { "report_link": None, "error_message": None, "clarification_prompt": None, "notification": None }
    try:
        with sync_playwright() as p:
            # Truy cập vietstock
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            url = f"https://finance.vietstock.vn/{stock_code.upper()}/tai-tai-lieu.htm?doctype=1"
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            if period != "Mới nhất" and year:
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
    except PlaywrightTimeoutError:
        output_state["error_message"] = f"Không tìm thấy thông tin cho mã chứng khoán '{stock_code}'. Vui lòng kiểm tra lại mã."
        return {**state, **output_state}
    except Exception as e:
        output_state["error_message"] = f"Lỗi khi scraping web: {str(e)}"
        return {**state, **output_state}
    
    if period == "Mới nhất":
        if not scraped_reports:
            output_state["error_message"] = f"Không tìm thấy báo cáo nào cho mã {stock_code}."
            return {**state, **output_state}
            
        # Lọc theo Hợp nhất/Công ty mẹ nếu có yêu cầu
        if user_consol_status:
            for report in scraped_reports:
                if user_consol_status.lower() in report["title"].lower():
                    output_state["report_link"] = report["link"]
                    output_state["notification"] = f"Đã tìm thấy báo cáo mới nhất theo yêu cầu: '{report['title']}'."
                    return {**state, **output_state}

            output_state["error_message"] = f"Không tìm thấy báo cáo '{user_consol_status}' mới nhất cho mã {stock_code}."
            return {**state, **output_state}
        else:
            # Nếu không yêu cầu, lấy cái đầu tiên (mới nhất)
            selected_report = scraped_reports[0]
            output_state["report_link"] = selected_report["link"]
            output_state["notification"] = f"Đã tìm thấy báo cáo mới nhất: '{selected_report['title']}'. (Mặc định lấy báo cáo đầu tiên trong danh sách)."
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

    user_quarter = state.get("quarter")

    # Trường hợp 1: Người dùng đã cung cấp đủ thông tin
    if period and user_consol_status:
        found_reports = []
        if period == "Quý" and user_quarter:
            found_reports = available_reports["Quý"][user_quarter][user_consol_status]
        elif period == "6 tháng":
            found_reports = available_reports["6 tháng"][user_consol_status]
        elif period == "Cả năm":
            found_reports = available_reports["Cả năm"][user_consol_status]

        if found_reports:
            selected_report = found_reports[0]
            output_state["report_link"] = selected_report["link"]
            output_state["notification"] = f"Đã tìm thấy báo cáo '{selected_report['title']}' theo yêu cầu."
            return {**state, **output_state}
        else:
            req_str = f"{period} Quý {user_quarter}" if period == "Quý" else period
            output_state["error_message"] = f"Không tìm thấy báo cáo '{req_str} - {user_consol_status}' bạn yêu cầu."
            return {**state, **output_state}

    # Trường hợp 2: Agent tự tìm và hỏi lại
    possible_choices = []
    requested_quarter_failed = False 

    # Tìm chính xác quý người dùng yêu cầu (nếu họ yêu cầu quý)
    if period == "Quý" and user_quarter:
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
                if period != "Cả năm" and available_reports["6 tháng"][cons_stat]:
                    report_item = available_reports["6 tháng"][cons_stat][0]
                    possible_choices.append({"period": "6 tháng", "consolidation_status": cons_stat, **report_item})

                if period != "6 tháng" and available_reports["Cả năm"][cons_stat]:
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