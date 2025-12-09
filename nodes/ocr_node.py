from state import StockReportState
from services.ocr import get_ocr_service
import os

def ocr_report_node(state: StockReportState) -> StockReportState:
    """
    Node to perform OCR on the found report link.
    """
    print("Bắt đầu Node: OCR Báo cáo")
    
    report_link = state.get("report_link")
    stock_code = state.get("stock_code")
    year = state.get("year")
    period = state.get("period")
    
    if not report_link:
        print("Không có link báo cáo để OCR.")
        return state

    try:
        # Initialize OCR Service (default to Marker)
        ocr_service = get_ocr_service("marker")    
        print(f"Đang gửi yêu cầu OCR cho: {report_link}")
        markdown_content = ocr_service.process_pdf(pdf_url=report_link)
        # Save file
        output_dir = "data/reports"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{stock_code}_{year}_{period}.md".replace(" ", "_")
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        print(f"OCR hoàn tất. Đã lưu tại: {filepath}")
        
        return {
            **state,
            "ocr_markdown_content": markdown_content,
            "notification": (state.get("notification") or "") + "\nĐã xử lý OCR thành công."
        }
        
    except Exception as e:
        error_msg = f"Lỗi OCR: {str(e)}"
        print(error_msg)
        return {
            **state,
            "error_message": error_msg
        }
