from state import StockReportState
from services.parser_service import FinancialParser
from services.validator_service import FinancialValidator
import json
import os

def parse_report_node(state: StockReportState) -> StockReportState:
    """
    Node to parse the OCR Markdown content into structured data and validate it.
    """
    print("Bắt đầu Node: Parse & Validate Báo cáo")
    
    markdown_content = state.get("ocr_markdown_content")
    stock_code = state.get("stock_code")
    year = state.get("year")
    period = state.get("period")
    
    if not markdown_content:
        print("Không có nội dung Markdown để parse.")
        return state

    try:
        # Parse
        parser = FinancialParser()
        print("Đang trích xuất dữ liệu cấu trúc (Parsing)...")
        parsed_data = parser.parse(markdown_content)
        # Save parsed data
        output_dir = "data/parsed"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{stock_code}_{year}_{period}.json".replace(" ", "_")
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)
        print(f"Đã lưu dữ liệu cấu trúc tại: {filepath}")

        # 2. Validate
        validator = FinancialValidator()
        print("Đang kiểm tra tính hợp lệ (Validating)...")
        validation_errors = validator.validate(parsed_data)
        
        notification = state.get("notification") or ""
        notification += "\nĐã trích xuất dữ liệu thành công."
        
        if validation_errors:
            error_msg = "\nCảnh báo dữ liệu:\n- " + "\n- ".join(validation_errors)
            print(error_msg)
            notification += error_msg
        else:
            print("Dữ liệu hợp lệ (thỏa mãn các phương trình kế toán cơ bản).")
            notification += "\nDữ liệu hợp lệ."

        return {
            **state,
            "notification": notification
        }

    except Exception as e:
        error_msg = f"Lỗi Parse/Validate: {str(e)}"
        print(error_msg)
        return {
            **state,
            "error_message": error_msg
        }
