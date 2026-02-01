from state import StockReportState
from services.ocr import get_ocr_service
from config import settings
from logger import get_logger
import os

logger = get_logger(__name__)

def ocr_report_node(state: StockReportState) -> StockReportState:
    """
    Node to perform OCR on the found report link.
    """
    logger.info("Bắt đầu Node: OCR Báo cáo")
    
    report_link = state.get("report_link")
    stock_code = state.get("stock_code", "UNKNOWN")
    year = state.get("year", "NA")
    period = state.get("period", "NA")
    
    if not report_link:
        logger.warning("Không có link báo cáo để OCR.")
        return {**state, "error_message": "Không có link báo cáo để OCR."}

    try:
        # Initialize OCR Service
        ocr_engine = state.get("ocr_engine") or settings.default_ocr_service
        ocr_service = get_ocr_service(str(ocr_engine))
        logger.info(f"Đang gửi yêu cầu OCR cho: {report_link}")
        markdown_content = ocr_service.process_pdf(pdf_url=report_link)
        
        if not markdown_content:
            logger.error("OCR returned empty content")
            return {**state, "error_message": "OCR trả về nội dung trống."}
        
        # Save file
        output_dir = settings.reports_output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Safe filename construction
        safe_stock = str(stock_code).replace(" ", "_") if stock_code else "UNKNOWN"
        safe_year = str(year) if year else "NA"
        safe_period = str(period).replace(" ", "_") if period else "NA"
        filename = f"{safe_stock}_{safe_year}_{safe_period}.md"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        logger.info(f"OCR hoàn tất. Đã lưu tại: {filepath}")
        
        preview_limit = 20_000
        preview = markdown_content[:preview_limit]

        # Avoid storing the full OCR text in state/session to reduce memory.
        return {
            **state,
            "ocr_markdown_path": filepath,
            "ocr_markdown_preview": preview,
            "ocr_engine": str(ocr_engine),
            "notification": (state.get("notification") or "") + "\nĐã xử lý OCR thành công."
        }
        
    except Exception as e:
        error_msg = f"Lỗi OCR: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            **state,
            "error_message": error_msg
        }
