from datetime import datetime
from langchain_core.tools import tool

@tool
def get_current_time() -> str:
    """Trả về ngày và tháng hiện tại để giúp LLM suy luận về sự tồn tại của các báo cáo theo quý."""
    return datetime.now().strftime("Hôm nay là ngày %d tháng %m năm %Y")