from typing import TypedDict, List, Dict, Optional, Literal
from pydantic_models import ReportRequest

# Agent State
class StockReportState(TypedDict):
    query: str
    pending_requests: List[ReportRequest]
    collected_links: Dict[str, str]
    comparison_context: str
    current_request_id: Optional[str]
    stock_code: Optional[str]
    year: Optional[int]
    period: Optional[Literal["Quý", "6 tháng", "Cả năm"]]
    quarter: Optional[int]
    consolidation_status: Optional[Literal["Hợp nhất", "Công ty mẹ"]]
    report_link: Optional[str]
    error_message: Optional[str]
    clarification_prompt: Optional[str]
    possible_choices: Optional[List[dict]]
    notification: Optional[str]
    final_response: Optional[str]