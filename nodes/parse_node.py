from state import StockReportState
from services.parser_service import FinancialParser
from services.validator_service import FinancialValidator
from config import settings
from logger import get_logger
import json
import os

logger = get_logger(__name__)

def parse_report_node(state: StockReportState) -> StockReportState:
    """
    Node to parse the OCR Markdown content into structured data and validate it.
    """
    logger.info("Bắt đầu Node: Parse & Validate Báo cáo")
    
    markdown_content = state.get("ocr_markdown_content")
    stock_code = state.get("stock_code", "UNKNOWN")
    year = state.get("year", "NA")
    period = state.get("period", "NA")
    
    if not markdown_content:
        logger.warning("Không có nội dung Markdown để parse.")
        return {**state, "error_message": "Không có nội dung Markdown để parse."}

    try:
        # Parse
        parser = FinancialParser()
        logger.info("Đang trích xuất dữ liệu cấu trúc (Parsing)...")
        parsed_data = parser.parse(markdown_content)
        
        if not parsed_data or (not parsed_data.get("BS") and not parsed_data.get("PL") and not parsed_data.get("CF")):
            logger.warning("Parser returned empty or invalid data")
        
        # Save parsed data
        output_dir = settings.parsed_output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        safe_stock = str(stock_code).replace(" ", "_") if stock_code else "UNKNOWN"
        safe_year = str(year) if year else "NA"
        safe_period = str(period).replace(" ", "_") if period else "NA"
        filename = f"{safe_stock}_{safe_year}_{safe_period}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Đã lưu dữ liệu cấu trúc tại: {filepath}")

        # 2. Validate
        validator = FinancialValidator()
        logger.info("Đang kiểm tra tính hợp lệ (Validating)...")
        validation_errors = validator.validate(parsed_data)
        
        notification = state.get("notification") or ""
        notification += "\nĐã trích xuất dữ liệu thành công."
        
        if validation_errors:
            error_msg = "\nCảnh báo dữ liệu:\n- " + "\n- ".join(validation_errors)
            logger.warning(error_msg)
            notification += error_msg
        else:
            logger.info("Dữ liệu hợp lệ (thỏa mãn các phương trình kế toán cơ bản).")
            notification += "\nDữ liệu hợp lệ."

        return {
            **state,
            "notification": notification
        }

    except Exception as e:
        error_msg = f"Lỗi Parse/Validate: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "error_message": error_msg
        }
