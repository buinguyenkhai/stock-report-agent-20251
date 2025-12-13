from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic_models import FinancialReportData
from config import settings
from logger import get_logger

logger = get_logger(__name__)


class FinancialParser:
    """
    Parses raw Markdown content into structured financial items using LLM.
    """
    def __init__(self):
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key
        )
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

        QUY TẮC QUAN TRỌNG:
        - Chỉ lấy số liệu của "Kỳ này" hoặc "Cuối kỳ" (Cột số liệu mới nhất). KHÔNG lấy số liệu "Kỳ trước" hay "Đầu năm".
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
                "BS": [item.model_dump() for item in result.balance_sheet],
                "PL": [item.model_dump() for item in result.income_statement],
                "CF": [item.model_dump() for item in result.cash_flow],
                "Notes": [item.model_dump() for item in result.notes]
            }
            logger.info(f"Successfully parsed: BS={len(parsed['BS'])}, PL={len(parsed['PL'])}, CF={len(parsed['CF'])} items")
            return parsed
        except Exception as e:
            logger.error(f"Parser Error: {str(e)}", exc_info=True)
            # Return empty structure on failure
            return {
                "BS": [],
                "PL": [],
                "CF": [],
                "Notes": []
            }
