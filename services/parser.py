"""Aggregated Financial Report Parser."""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator
import re
import traceback
import math
import unicodedata

from config import settings
from logger import get_logger
from services.llm_factory import create_llm_for_task, create_structured_llm_for_task

logger = get_logger(__name__)

# Pydantic Models for Structured Output
class FinancialItem(BaseModel):
    """Single financial line item."""
    item_code: Optional[str] = Field(default=None, description="Mã số chỉ tiêu")
    item_name: str = Field(description="Tên chỉ tiêu")
    value: Optional[float] = Field(default=None, description="Giá trị số (đã chuyển về VND)")
    notes_ref: Optional[str] = Field(default=None, description="Thuyết minh")
    original_name: Optional[str] = Field(default=None, description="Tên gốc trong báo cáo nếu khác")
    row_identity: Optional[str] = Field(
        default=None,
        description="Định danh ổn định cho cùng một chỉ tiêu kinh doanh, dùng khi có lặp tên hoặc lặp qua nhiều trang",
    )
    column_label: Optional[str] = Field(
        default=None,
        description="Nhãn cột giá trị gốc, ví dụ 'Quý III 2024', 'Năm nay', '2023'",
    )
    period_key: Optional[str] = Field(
        default=None,
        description="Khóa kỳ chuẩn hóa cho cột giá trị, ví dụ 2024Q3, 2023Q3_YTD, 2024FY",
    )


class ParsedStatement(BaseModel):
    """Parsed financial statement wrapper."""
    items: List[FinancialItem] = Field(default_factory=list)


class ParsedReport(BaseModel):
    """Complete parsed financial report."""
    # Metadata
    company_name: Optional[str] = Field(default=None)
    stock_ticker: Optional[str] = Field(default=None)
    year: Optional[int] = Field(default=None)
    quarter: Optional[int] = Field(default=None)
    unit: str = Field(default="VND", description="Original source unit shown in report")
    source_unit_label: str = Field(default="VND", description="Detected unit label from source report")
    source_unit_multiplier_to_vnd: float = Field(
        default=1.0,
        description="Multiplier used to convert printed source values to canonical VND",
    )
    value_unit: str = Field(default="VND", description="Canonical unit of parsed numeric values")
    is_ytd: bool = Field(default=False, description="True if values are year-to-date cumulative")
    
    # Financial statements
    balance_sheet: ParsedStatement = Field(default_factory=ParsedStatement)
    income_statement: ParsedStatement = Field(default_factory=ParsedStatement)
    cash_flow: ParsedStatement = Field(default_factory=ParsedStatement)
    
    # Parsing status
    bs_found: bool = Field(default=False)
    pl_found: bool = Field(default=False)
    cf_found: bool = Field(default=False)
    warnings: List[str] = Field(default_factory=list)
    parse_audit: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('quarter', mode='before')
    @classmethod
    def parse_quarter(cls, v):
        """Convert quarter strings like 'Q4', 'Q1', 'Quý 3' to int."""
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            # Try to extract digit from strings like "Q4", "Q1", "Quý 3"
            match = re.search(r'(\d+)', v)
            if match:
                return int(match.group(1))
            # Try direct conversion
            try:
                return int(v)
            except ValueError:
                return None
        return None


# Extraction Results Container
@dataclass
class ExtractionBundle:
    """Bundle of all extraction results to pass to parser."""
    balance_sheet: str = ""
    income_statement: str = ""
    cash_flow: str = ""
    notes_text: str = ""
    notes_tables: str = ""
    other_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_tables_content(self) -> str:
        """Get combined financial tables content."""
        parts = []
        if self.balance_sheet:
            parts.append("## BẢNG CÂN ĐỐI KẾ TOÁN\n" + self.balance_sheet)
        if self.income_statement:
            parts.append("## BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH\n" + self.income_statement)
        if self.cash_flow:
            parts.append("## BÁO CÁO LƯU CHUYỂN TIỀN TỆ\n" + self.cash_flow)
        return "\n\n".join(parts)
    
    def has_content(self) -> bool:
        """Check if any content was extracted."""
        return bool(self.balance_sheet or self.income_statement or self.cash_flow)


# Parser Implementation

class AggregatedParser:
    """
    Smart parser that processes extracted content and produces structured output with proper normalization.
    """
    
    def __init__(self, model: Optional[str] = None):
        """Initialize parser with LLM model."""
        self.model = model or settings.llm_model
        self._llm = None
    
    @property
    def llm(self):
        """Lazy-load structured LLM."""
        if self._llm is None:
            # Use task config (larger max_tokens + longer timeout) to avoid
            # truncated/invalid structured outputs on long documents.
            self._llm = create_structured_llm_for_task(
                task="parsing",
                model=self.model,
                schema=ParsedReport,
            )
        return self._llm
    
    def parse(self, bundle: ExtractionBundle) -> ParsedReport:
        """
        Parse extraction bundle into structured report.
        """
        system_prompt: str = ""
        user_prompt: str = ""

        if not bundle.has_content():
            logger.warning("No content to parse")
            return ParsedReport(warnings=["No financial tables found in extraction"])
        
        # Build the prompt
        system_prompt = self._get_system_prompt()
        user_prompt = self._get_user_prompt(bundle)

        try:
            
            messages = [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
            
            logger.info(f"Parsing {len(bundle.get_tables_content()):,} chars of extracted content")
            
            # Invoke LLM
            result: ParsedReport = self.llm.invoke(messages) # type: ignore
            
            result = self._finalize_report(result, bundle)
            
            logger.info(
                f"Parsed: BS={len(result.balance_sheet.items)}, "
                f"PL={len(result.income_statement.items)}, "
                f"CF={len(result.cash_flow.items)} items"
            )
            
            return result

        except Exception as e:
            err_type = type(e).__name__
            err_msg = str(e) or repr(e)
            logger.exception(f"Parsing failed ({err_type}): {err_msg}")

            # Fallback: ask the model for raw JSON and parse manually.
            try:
                raw_llm = create_llm_for_task("parsing", model=self.model)
                fallback_messages = [
                    (
                        "system",
                        system_prompt
                        + "\n\nBẠN PHẢI trả về DUY NHẤT một JSON object hợp lệ, không markdown, không giải thích.",
                    ),
                    ("human", user_prompt + "\n\nChỉ trả về JSON."),
                ]
                resp = raw_llm.invoke(fallback_messages)
                content = (getattr(resp, "content", None) or "").strip()

                # Try to extract JSON from code fences or surrounding text.
                import re
                json_text = ""
                m = re.search(r"```json\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
                if m:
                    json_text = m.group(1)
                else:
                    start = content.find("{")
                    end = content.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        json_text = content[start : end + 1]
                    else:
                        json_text = content

                data = ParsedReport.model_validate_json(json_text)
                data = self._finalize_report(data, bundle)
                data.warnings.append(
                    f"Parser fallback used due to structured parse error ({err_type}): {err_msg}"
                )
                return data
            except Exception as e2:
                err2_type = type(e2).__name__
                err2_msg = str(e2) or repr(e2)
                return ParsedReport(
                    warnings=[
                        f"Parsing error ({err_type}): {err_msg}",
                        f"Fallback parsing error ({err2_type}): {err2_msg}",
                        f"Parsing traceback: {traceback.format_exc()}",
                    ]
                )
    
    def _get_system_prompt(self) -> str:
        """Generate system prompt for financial report parsing."""
        
        return """Bạn là chuyên gia phân tích báo cáo tài chính Việt Nam.
Nhiệm vụ: Trích xuất dữ liệu từ 3 báo cáo tài chính chính sang định dạng cấu trúc.

## QUY TẮC QUAN TRỌNG:

### 1. Trích xuất chỉ tiêu
- Trích xuất trung thực tên chỉ tiêu, mã số (nếu có) và thuyết minh (nếu có) từ báo cáo.
- Không cần chuẩn hóa tên theo danh mục bên thứ ba, ưu tiên giữ đúng tên gốc trong báo cáo.

### 2. Xử lý số liệu
- Chuyển TẤT CẢ giá trị về đơn vị VND (đồng).
- Nếu đơn vị là "triệu VND": nhân 1,000,000.
- Nếu đơn vị là "tỷ VND": nhân 1,000,000,000.
- Nếu đơn vị là "nghìn VND": nhân 1,000.
- Số âm: giữ nguyên dấu (chi phí, chi tiền thường là số âm).
- Số trong ngoặc đơn (1,234) = số âm -1234.
- Toàn bộ output phải dùng CÙNG một chuẩn giá trị: `value_unit = "VND"`.
- Phải điền `source_unit_label` và `source_unit_multiplier_to_vnd` nhất quán cho cả báo cáo.

### 3. Xử lý định dạng số Việt Nam
- Dấu chấm (.) thường là phân cách hàng nghìn trong báo cáo VN: 1.234.567 = 1234567.
- Dấu phẩy (,) thường là phân cách thập phân: 1.234,56 = 1234.56.
- Lưu ý: Một số báo cáo theo chuẩn quốc tế có thể dùng ngược lại. Hãy kiểm tra ngữ cảnh.

### 4. Xác định YTD (lũy kế)
- Nếu thấy "Lũy kế từ đầu năm" hoặc giá trị lớn bất thường cho báo cáo quý → is_ytd = true.
- Nếu là Q4 và có cột "Quý 4" riêng → lấy số quý, không lấy lũy kế.

### 5. Quy ước dấu (Sign convention)
- Chi phí, chi tiền: SỐ ÂM.
- Doanh thu, thu nhập, thu tiền: SỐ DƯƠNG.
- Giữ nguyên dấu logic như trong báo cáo gốc.

### 6. Định danh dòng và cột
- Nếu cùng một `item_name` lặp lại ở các cột thời gian khác nhau hoặc các mục con khác nhau, hãy điền `row_identity` ổn định.
- Nếu trang có nhiều cột giá trị, hãy điền `column_label` và `period_key` cho từng dòng.
- `period_key` nên theo dạng chuẩn như `2024Q3`, `2023Q3`, `2024Q3_YTD`, `2024FY` khi suy ra được."""

    def _get_user_prompt(self, bundle: ExtractionBundle) -> str:
        """Generate user prompt with extracted content."""
        content = bundle.get_tables_content()
        
        # Add metadata context if available
        metadata_context = ""
        if bundle.metadata:
            meta_parts = []
            if "unit" in bundle.metadata:
                meta_parts.append(f"- Đơn vị: {bundle.metadata['unit']}")
            if "year" in bundle.metadata:
                meta_parts.append(f"- Năm: {bundle.metadata['year']}")
            if "quarter" in bundle.metadata:
                meta_parts.append(f"- Quý: {bundle.metadata['quarter']}")
            if "is_ytd" in bundle.metadata:
                meta_parts.append(f"- Lũy kế: {bundle.metadata['is_ytd']}")
            if meta_parts:
                metadata_context = "## THÔNG TIN ĐÃ BIẾT:\n" + "\n".join(meta_parts) + "\n\n"
        
        return f"""{metadata_context}## NỘI DUNG TRÍCH XUẤT:

{content}

## YÊU CẦU:
1. Trích xuất TẤT CẢ chỉ tiêu từ 3 báo cáo.
2. Với mỗi chỉ tiêu, lấy: item_name, item_code (mã số), value (giá trị VND), notes_ref (thuyết minh), original_name, row_identity, column_label, period_key.
3. Chuyển đổi giá trị về VND dựa trên đơn vị tính của báo cáo.
4. Điền metadata đơn vị nguồn:
   - source_unit_label
   - source_unit_multiplier_to_vnd
   - value_unit = VND
5. Không được trộn nhiều thang đơn vị khác nhau trong cùng output."""

    def _normalize_text_ascii(self, value: Any) -> str:
        text = unicodedata.normalize("NFKD", str(value or ""))
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"\s+", " ", text.strip().lower())
        text = text.replace("đ", "d").replace("Đ", "D")
        return text

    def _detect_source_unit(self, bundle: ExtractionBundle) -> Tuple[str, float]:
        raw_unit = str(bundle.metadata.get("unit") or "").strip()
        content = "\n\n".join(
            part
            for part in [
                raw_unit,
                bundle.balance_sheet,
                bundle.income_statement,
                bundle.cash_flow,
            ]
            if part
        )
        norm = self._normalize_text_ascii(content)
        patterns = [
            (("ty vnd", "ty dong", "ty vietnam dong", "ty d"), 1_000_000_000.0),
            (("trieu vnd", "trieu dong", "trieu vietnam dong", "trieu d"), 1_000_000.0),
            (("nghin vnd", "nghin dong", "nghin vietnam dong", "nghin d"), 1_000.0),
            (("ngan vnd", "ngan dong", "ngan vietnam dong", "ngan d"), 1_000.0),
            (("vnd", "dong", "viet nam dong", "vietnam dong"), 1.0),
        ]
        for aliases, scale in patterns:
            for alias in aliases:
                if alias in norm:
                    return alias, scale
        if raw_unit:
            return raw_unit, 1.0
        return "VND", 1.0

    def _build_default_row_identity(self, statement: str, item: FinancialItem, index: int) -> str:
        if item.row_identity:
            return str(item.row_identity).strip()
        if item.item_code and str(item.item_code).strip():
            base = f"{statement}|code:{str(item.item_code).strip()}"
        elif item.notes_ref and str(item.notes_ref).strip():
            base = f"{statement}|name:{item.item_name.strip()}|note:{str(item.notes_ref).strip()}"
        else:
            base = f"{statement}|name:{item.item_name.strip()}"
        if item.period_key and str(item.period_key).strip():
            return f"{base}|period:{str(item.period_key).strip()}"
        if item.column_label and str(item.column_label).strip():
            return f"{base}|column:{str(item.column_label).strip()}"
        return base if base.strip() else f"{statement}|idx:{index}"

    def _build_parse_audit(self, report: ParsedReport) -> Dict[str, Any]:
        repeated: Dict[str, int] = {}
        scale_conflicts: List[Dict[str, Any]] = []
        sign_conflicts: List[Dict[str, Any]] = []
        values_by_identity: Dict[str, List[float]] = {}

        for statement_name, statement in (
            ("balance_sheet", report.balance_sheet),
            ("income_statement", report.income_statement),
            ("cash_flow", report.cash_flow),
        ):
            for idx, item in enumerate(statement.items):
                row_identity = self._build_default_row_identity(statement_name, item, idx)
                item.row_identity = row_identity
                repeated[row_identity] = repeated.get(row_identity, 0) + 1
                if item.value is not None:
                    values_by_identity.setdefault(row_identity, []).append(float(item.value))

        for row_identity, values in values_by_identity.items():
            if len(values) < 2:
                continue
            base = values[0]
            for other in values[1:]:
                if base == 0.0 or other == 0.0:
                    continue
                if (base < 0 < other) or (other < 0 < base):
                    sign_conflicts.append(
                        {"row_identity": row_identity, "values": [base, other]}
                    )
                ratio = abs(other / base)
                for factor in (1e-6, 1e-3, 1e3, 1e6):
                    if math.isclose(ratio, factor, rel_tol=1e-9, abs_tol=1e-12):
                        scale_conflicts.append(
                            {"row_identity": row_identity, "factor": factor, "values": [base, other]}
                        )
                        break

        return {
            "repeated_row_identity_count": int(sum(1 for count in repeated.values() if count > 1)),
            "repeated_row_identities": {k: v for k, v in repeated.items() if v > 1},
            "possible_scale_conflicts": scale_conflicts[:100],
            "possible_sign_conflicts": sign_conflicts[:100],
        }

    def _finalize_report(self, result: ParsedReport, bundle: ExtractionBundle) -> ParsedReport:
        if bundle.metadata:
            if not result.company_name and "company_name" in bundle.metadata:
                result.company_name = bundle.metadata["company_name"]
            if not result.stock_ticker and "stock_ticker" in bundle.metadata:
                result.stock_ticker = bundle.metadata["stock_ticker"]
            if not result.year and "year" in bundle.metadata:
                result.year = bundle.metadata["year"]
            if not result.quarter and "quarter" in bundle.metadata:
                result.quarter = bundle.metadata["quarter"]
            if "is_ytd" in bundle.metadata:
                result.is_ytd = bool(bundle.metadata["is_ytd"])

        unit_label, unit_multiplier = self._detect_source_unit(bundle)
        result.unit = str(bundle.metadata.get("unit") or unit_label or result.unit or "VND")
        result.source_unit_label = str(unit_label or result.source_unit_label or "VND")
        result.source_unit_multiplier_to_vnd = float(unit_multiplier or 1.0)
        result.value_unit = "VND"

        result.bs_found = len(result.balance_sheet.items) > 0
        result.pl_found = len(result.income_statement.items) > 0
        result.cf_found = len(result.cash_flow.items) > 0
        result.parse_audit = self._build_parse_audit(result)

        if result.source_unit_multiplier_to_vnd <= 0:
            result.warnings.append("Invalid source_unit_multiplier_to_vnd detected; forcing 1.0")
            result.source_unit_multiplier_to_vnd = 1.0
        if result.value_unit != "VND":
            result.warnings.append("Parser value_unit was not VND; forcing canonical VND")
            result.value_unit = "VND"
        if result.parse_audit.get("possible_scale_conflicts"):
            result.warnings.append("Possible mixed-scale row identities detected in parsed output")
        if result.parse_audit.get("possible_sign_conflicts"):
            result.warnings.append("Possible sign conflicts detected in parsed output")

        return result

    def to_dict(self, report: ParsedReport) -> Dict[str, Any]:
        """
        Convert ParsedReport to dictionary format compatible with UI and Evaluation.
        """
        return {
            "metadata": {
                "company_name": report.company_name,
                "stock_ticker": report.stock_ticker,
                "year": report.year,
                "quarter": report.quarter,
                "unit": report.unit,
                "source_unit_label": report.source_unit_label,
                "source_unit_multiplier_to_vnd": report.source_unit_multiplier_to_vnd,
                "value_unit": report.value_unit,
                "is_ytd": report.is_ytd,
            },
            "balance_sheet": {
                "items": [
                    {
                        "item_code": item.item_code,
                        "item_name": item.item_name,
                        "value": item.value,
                        "notes_ref": item.notes_ref,
                        "original_name": item.original_name,
                        "row_identity": item.row_identity,
                        "column_label": item.column_label,
                        "period_key": item.period_key,
                    }
                    for item in report.balance_sheet.items
                ]
            },
            "income_statement": {
                "items": [
                    {
                        "item_code": item.item_code,
                        "item_name": item.item_name,
                        "value": item.value,
                        "notes_ref": item.notes_ref,
                        "original_name": item.original_name,
                        "row_identity": item.row_identity,
                        "column_label": item.column_label,
                        "period_key": item.period_key,
                    }
                    for item in report.income_statement.items
                ]
            },
            "cash_flow": {
                "items": [
                    {
                        "item_code": item.item_code,
                        "item_name": item.item_name,
                        "value": item.value,
                        "notes_ref": item.notes_ref,
                        "original_name": item.original_name,
                        "row_identity": item.row_identity,
                        "column_label": item.column_label,
                        "period_key": item.period_key,
                    }
                    for item in report.cash_flow.items
                ]
            },
            "status": {
                "bs_found": report.bs_found,
                "pl_found": report.pl_found,
                "cf_found": report.cf_found,
                "warnings": report.warnings,
                "parse_audit": report.parse_audit,
            }
        }
