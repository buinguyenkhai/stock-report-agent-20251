from state import StockReportState
from config import settings
from logger import get_logger
import regex as re
from typing import Any, Dict, List, Optional, cast

import requests

logger = get_logger(__name__)

def prepare_next_extraction_node(state: StockReportState) -> StockReportState:
    """Lấy yêu cầu tiếp theo từ danh sách chờ và cập nhật State."""
    logger.info("Bắt đầu Node: Chuẩn bị Trích xuất")
    pending = list(state.get("pending_requests", []))
    if not pending:
        logger.warning("No pending requests found")
        return cast(StockReportState, {**state, "error_message": "Không có yêu cầu nào đang chờ."})
    next_request = pending.pop(0)
    logger.info(f"Đang xử lý yêu cầu: {next_request.request_id} - {next_request.stock_code} {next_request.period} {next_request.quarter}/{next_request.year}")

    return cast(StockReportState, {
        **state,
        "pending_requests": pending,
        "current_request_id": next_request.request_id,
        "stock_code": next_request.stock_code,
        "year": next_request.year,
        "period": next_request.period,
        "quarter": next_request.quarter,
        "consolidation_status": next_request.consolidation_status,
        "report_link": None,
        "error_message": None,
        "clarification_prompt": None,
        "notification": None,
    })

def extract_report_link_node(state: StockReportState) -> StockReportState:
    """Node để trích xuất link PDF."""
    stock_code = state.get("stock_code", "")
    logger.info(f"Bắt đầu Node: Trích xuất link cho {stock_code}")
    
    if not stock_code:
        logger.error("Missing stock_code in state")
        return cast(StockReportState, {**state, "error_message": "Thiếu mã chứng khoán."})
    
    # Khởi tạo
    year = state.get("year")
    period = state.get("period")
    user_consol_status = state.get("consolidation_status")
    output_state: Dict[str, Any] = {"report_link": None, "error_message": None, "clarification_prompt": None, "notification": None}

    def _merge(updates: Dict[str, Any]) -> StockReportState:
        return cast(StockReportState, {**state, **updates})

    def _get_token(session: requests.Session, page_url: str) -> str:
        html = session.get(
            page_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=max(10, int(settings.scraper_timeout / 1000)),
        ).text
        m = re.search(r"name=__RequestVerificationToken[^>]*value=([^\s>]+)", html)
        if not m:
            raise ValueError("Không lấy được __RequestVerificationToken từ trang Vietstock")
        return m.group(1).strip("\"'")

    def _fetch_documents(
        session: requests.Session,
        *,
        code: str,
        year: Optional[int],
        doc_type: int = 1,
        max_pages: int = 30,
    ) -> List[Dict[str, Any]]:
        page_url = f"{settings.vietstock_base_url}/{code.upper()}/tai-tai-lieu.htm?doctype={doc_type}"
        token = _get_token(session, page_url)

        headers = {
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": page_url,
        }

        all_items: List[Dict[str, Any]] = []
        seen_ids = set()
        total_row: Optional[int] = None
        page_size: Optional[int] = None

        for page in range(1, max_pages + 1):
            payload: Dict[str, Any] = {
                "code": code.upper(),
                "page": page,
                "type": doc_type,
                "__RequestVerificationToken": token,
            }
            if year:
                payload["year"] = int(year)

            resp = session.post(
                f"{settings.vietstock_base_url}/data/getdocument",
                data=payload,
                headers=headers,
                timeout=max(10, int(settings.scraper_wait_timeout / 1000)),
            )
            ct = (resp.headers.get("content-type") or "").lower()
            if "application/json" not in ct:
                raise ValueError(f"Vietstock API trả về content-type không hợp lệ: {ct}")

            items = resp.json()
            if not isinstance(items, list) or not items:
                break

            if page_size is None:
                page_size = len(items)

            # TotalRow is repeated on each item
            if total_row is None:
                try:
                    total_row = int(items[0].get("TotalRow") or 0)
                except Exception:
                    total_row = 0

            for it in items:
                file_id = it.get("FileInfoID")
                if file_id in seen_ids:
                    continue
                seen_ids.add(file_id)
                all_items.append(it)

            if total_row and len(all_items) >= total_row:
                break

            if page_size and len(items) < page_size:
                break

        return all_items

    try:
        session = requests.Session()
        # If year not specified ("Mới nhất"), fetch without year filter.
        api_items = _fetch_documents(session, code=stock_code, year=year if (period != "Mới nhất") else None)

        scraped_reports = []
        for it in api_items:
            title = (it.get("Title") or "").strip()
            link = (it.get("Url") or "").strip()
            if not title or not link:
                continue
            cleaned_title = re.sub(r"\s*\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}\s*$", "", title)
            scraped_reports.append({"title": cleaned_title, "link": link, "_sort": it.get("LastUpdate") or it.get("FileInfoID")})

        # Prefer newest first
        scraped_reports.sort(key=lambda r: str(r.get("_sort") or ""), reverse=True)
        for r in scraped_reports:
            r.pop("_sort", None)

        logger.debug(f"Found {len(scraped_reports)} reports")
        if not scraped_reports:
            output_state["error_message"] = f"Không tìm thấy báo cáo nào cho mã {stock_code}{' năm ' + str(year) if year else ''}."
            return _merge(output_state)
    except Exception as e:
        logger.error(f"Scraping error: {e}", exc_info=True)
        output_state["error_message"] = f"Lỗi khi scraping web: {str(e)}"
        return _merge(output_state)
    
    if period == "Mới nhất":
        if not scraped_reports:
            output_state["error_message"] = f"Không tìm thấy báo cáo nào cho mã {stock_code}."
            return _merge(output_state)
            
        # Lọc theo Hợp nhất/Công ty mẹ nếu có yêu cầu
        if user_consol_status:
            for report in scraped_reports:
                if user_consol_status.lower() in report["title"].lower():
                    output_state["report_link"] = report["link"]
                    output_state["notification"] = f"Đã tìm thấy báo cáo mới nhất theo yêu cầu: '{report['title']}'."
                    return _merge(output_state)

            output_state["error_message"] = f"Không tìm thấy báo cáo '{user_consol_status}' mới nhất cho mã {stock_code}."
            return _merge(output_state)
        else:
            # Nếu không yêu cầu, lấy cái đầu tiên (mới nhất)
            selected_report = scraped_reports[0]
            output_state["report_link"] = selected_report["link"]
            output_state["notification"] = f"Đã tìm thấy báo cáo mới nhất: '{selected_report['title']}'. (Mặc định lấy báo cáo đầu tiên trong danh sách)."
            return _merge(output_state)

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
            return _merge(output_state)
        else:
            req_str = f"{period} Quý {user_quarter}" if period == "Quý" else period
            output_state["error_message"] = f"Không tìm thấy báo cáo '{req_str} - {user_consol_status}' bạn yêu cầu."
            return _merge(output_state)

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
        output_state["consolidation_status"] = selected.get("consolidation_status")
        output_state["quarter"] = selected.get("quarter")
    else:
        # Streamlit UI cannot block on console input; auto-pick a reasonable default.
        preferred = None
        for c in possible_choices:
            if c.get("consolidation_status") == "Hợp nhất":
                preferred = c
                break
        selected = preferred or possible_choices[0]

        output_state["report_link"] = selected.get("link")
        output_state["consolidation_status"] = selected.get("consolidation_status")
        output_state["quarter"] = selected.get("quarter")

        titles = ", ".join([c.get("title", "") for c in possible_choices[:3] if c.get("title")])
        more = "" if len(possible_choices) <= 3 else f" (+{len(possible_choices) - 3} khác)"
        output_state["notification"] = (
            f"Tìm thấy nhiều báo cáo phù hợp ({len(possible_choices)}). "
            f"Mặc định chọn: '{selected.get('title', '')}'. "
            f"Các lựa chọn khác: {titles}{more}. "
            "Nếu muốn chọn khác, hãy ghi rõ 'Hợp nhất' hoặc 'Công ty mẹ' trong truy vấn."
        )
    return _merge(output_state)