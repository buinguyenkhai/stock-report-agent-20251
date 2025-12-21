from state import StockReportState
from services.pipeline import create_pipeline
from config import settings
from logger import get_logger
import json
import os

logger = get_logger(__name__)

def parse_report_node(state: StockReportState) -> StockReportState:
    """
    Node to parse the OCR Markdown content into structured data.
    Uses the new LLM-based extraction pipeline.
    """
    logger.info("Bắt đầu Node: Parse Báo cáo")
    
    markdown_content = state.get("ocr_markdown_content")
    stock_code = state.get("stock_code", "UNKNOWN")
    year = state.get("year", "NA")
    period = state.get("period", "NA")
    
    if not markdown_content:
        logger.warning("Không có nội dung Markdown để parse.")
        return {**state, "error_message": "Không có nội dung Markdown để parse."}

    try:
        # Create pipeline and process
        pipeline = create_pipeline(mode="separate", extract_notes=False, extract_metadata=True)
        logger.info("Đang trích xuất dữ liệu cấu trúc (Parsing with LLM pipeline)...")
        parsed_report = pipeline.process(markdown_content)
        
        # Convert to dict format
        parsed_data = pipeline.to_dict(parsed_report)
        
        if not parsed_data.get("balance_sheet") and not parsed_data.get("income_statement") and not parsed_data.get("cash_flow"):
            logger.warning("Parser returned empty data")
        
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

        notification = state.get("notification") or ""
        bs_count = len(parsed_data.get("balance_sheet", {}).get("items", []))
        pl_count = len(parsed_data.get("income_statement", {}).get("items", []))
        cf_count = len(parsed_data.get("cash_flow", {}).get("items", []))
        notification += f"\nĐã trích xuất dữ liệu thành công: BS={bs_count}, PL={pl_count}, CF={cf_count} items."

        return {
            **state,
            "notification": notification
        }

    except Exception as e:
        error_msg = f"Lỗi Parse: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "error_message": error_msg
        }
