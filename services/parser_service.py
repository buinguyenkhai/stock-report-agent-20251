from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.prompts import ChatPromptTemplate
from pydantic_models import FinancialReportData
from config import settings
from logger import get_logger
from services.llm_factory import create_llm_for_task

logger = get_logger(__name__)


class FinancialParser:
    """
    Parses raw Markdown content into structured financial items using LLM.
    """
    def __init__(self):
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        
        self.llm = create_llm_for_task("parsing", model=settings.llm_model)
        self.structured_llm = self.llm.with_structured_output(FinancialReportData)

    @retry(
        stop=stop_after_attempt(settings.retry_max_attempts),
        wait=wait_exponential(multiplier=1, min=settings.retry_min_wait, max=settings.retry_max_wait),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying LLM parse call (attempt {retry_state.attempt_number})..."
        )
    )
    def parse(self, markdown_content: str) -> dict:
        """
        Returns a dictionary with keys 'BS' (Balance Sheet), 'PL' (Profit Loss), 'CF' (Cash Flow).
        Each value is a list of items.
        """
        system_prompt = """Bạn là một chuyên gia kế toán và phân tích dữ liệu tài chính.
        Nhiệm vụ của bạn là trích xuất dữ liệu từ văn bản Markdown của Báo cáo tài chính (Việt Nam) thành cấu trúc JSON.

        HÃY TRÍCH XUẤT 3 BẢNG CHÍNH VÀ THUYẾT MINH:
        1. Bảng Cân đối kế toán (Balance Sheet)
        2. Báo cáo Kết quả hoạt động kinh doanh (Income Statement)
        3. Báo cáo Lưu chuyển tiền tệ (Cash Flow)
        4. Thuyết minh (Notes): Trích xuất danh sách các mục thuyết minh chính (Số hiệu, Tiêu đề, Nội dung tóm tắt).

        === XÁC ĐỊNH PHẠM VI BÁO CÁO (report_scope) ===
        - Xem tiêu đề báo cáo để xác định:
          * "Hợp nhất" / "Consolidated" -> report_scope = "consolidated"
          * "Công ty mẹ" / "Riêng lẻ" / "Parent" / "Separate" -> report_scope = "parent"
        - Nếu không rõ, mặc định là "consolidated".

        === XÁC ĐỊNH LOẠI KỲ BÁO CÁO (period_type) ===
        - Báo cáo có thể có nhiều cột số liệu. Hãy xác định loại kỳ từ header cột:
          * "Quý X" / "Số quý này" / "Kỳ này" / "Năm nay" -> period_type = "quarterly" (số liệu của riêng quý đó)
          * "Lũy kế" / "Từ đầu năm" / "Accumulated" / "YTD" / "Cộng dồn" -> period_type = "cumulative" (lũy kế từ đầu năm)
        - **ƯU TIÊN**: Nếu có nhiều cột, hãy ưu tiên lấy cột "quarterly" (số quý) thay vì "cumulative" (lũy kế).
        - Nếu chỉ có một cột hoặc không rõ, mặc định là "quarterly".

        === CHỌN CỘT SỐ LIỆU ĐÚNG ===
        - Bảng có thể có 2-4 cột số liệu với các tổ hợp:
          * Cột 1: Quý X năm nay | Cột 2: Quý X năm trước | Cột 3: Lũy kế năm nay | Cột 4: Lũy kế năm trước
          * Hoặc: Cuối kỳ | Đầu năm (cho Balance Sheet)
        - **CHỈ LẤY** cột số liệu của kỳ hiện tại (Quý X năm nay hoặc Cuối kỳ). KHÔNG lấy:
          * Số liệu năm trước / kỳ trước
          * Số liệu đầu năm
          * Số liệu lũy kế (trừ khi báo cáo chỉ có cột lũy kế)

        QUY TẮC QUAN TRỌNG:
        - **ĐƠN VỊ TIỀN TỆ**: Tìm dòng "Đơn vị tính:" hoặc cột header có chứa đơn vị. Có thể là:
          * "VND" hoặc "VNĐ" hoặc "đồng" -> unit = "VND"
          * "triệu VND" hoặc "Triệu VND" hoặc "Triệu đồng" -> unit = "triệu VND"
          * "tỷ VND" hoặc "Tỷ VND" hoặc "Tỷ đồng" hoặc "Bn. VND" -> unit = "tỷ VND"
          * "nghìn VND" hoặc "Nghìn đồng" -> unit = "nghìn VND"
        - "item_code" (Mã số) là RẤT QUAN TRỌNG. Hãy cố gắng lấy chính xác. Nếu không có, hãy để trống.
        - "value" (Giá trị) phải là số thực (float). Hãy xử lý các dấu phân cách hàng nghìn (dấu chấm hoặc phẩy tùy báo cáo) để chuyển thành số đúng. Ví dụ: "1.000.000" -> 1000000.
        - Nếu giá trị nằm trong ngoặc đơn `(100)`, đó là số âm -> -100.
        - Bỏ qua các dòng tiêu đề không có số liệu (ví dụ: "I. TÀI SẢN NGẮN HẠN"). Chỉ lấy các dòng có giá trị cụ thể.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Đây là nội dung báo cáo:\n\n{content}")
        ])

        chain = prompt | self.structured_llm

        try:
            logger.info("Invoking LLM for structured parsing...")
            result: FinancialReportData = chain.invoke({"content": markdown_content})
            
            parsed = {
                "unit": result.unit or "VND",  # Default to VND if not detected
                "report_scope": result.report_scope or "consolidated",
                "period_type": result.period_type or "quarterly",
                "BS": [item.model_dump() for item in result.balance_sheet],
                "PL": [item.model_dump() for item in result.income_statement],
                "CF": [item.model_dump() for item in result.cash_flow],
                "Notes": [item.model_dump() for item in result.notes]
            }
            logger.info(f"Successfully parsed: Unit={parsed['unit']}, Scope={parsed['report_scope']}, Period={parsed['period_type']}, BS={len(parsed['BS'])}, PL={len(parsed['PL'])}, CF={len(parsed['CF'])} items")
            return parsed
        except Exception as e:
            logger.error(f"Parser Error: {str(e)}", exc_info=True)
            # Return empty structure on failure
            return {
                "unit": "VND",
                "report_scope": "consolidated",
                "period_type": "quarterly",
                "BS": [],
                "PL": [],
                "CF": [],
                "Notes": []
            }
