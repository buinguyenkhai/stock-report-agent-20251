import pandas as pd
import re
from typing import List, Dict, Optional
from datetime import datetime

from .canonical_format import (
    FinancialItem,
    FinancialStatement,
    FinancialReport,
    normalize_to_billions
)
from logger import get_logger

logger = get_logger(__name__)

EXPENSE_PATTERNS = [
    # Cost of goods sold / Giá vốn
    r'giá vốn',
    r'cost of (?:goods )?sold',
    
    # Financial costs / Chi phí tài chính
    r'chi phí tài chính',
    r'chi phí lãi vay',
    r'lãi vay',
    r'financial (?:cost|expense)',
    r'interest expense',
    
    # Operating expenses / Chi phí hoạt động
    r'chi phí bán hàng',
    r'chi phí quản lý',
    r'chi phí quản lý doanh nghiệp',
    r'selling expense',
    r'admin(?:istrative)? expense',
    r'general .* admin',
    
    # Tax expenses / Chi phí thuế
    r'chi phí thuế',
    r'thuế thu nhập (?:doanh nghiệp|dn)',
    r'income tax expense',
    
    # Depreciation and amortization
    r'khấu hao',
    r'depreciation',
    r'amortization',
    
    # Provisions / Dự phòng (as expense, not reserve)
    r'trích lập dự phòng',
    r'chi phí dự phòng',
    
    # Other costs
    r'chi phí khác',
    r'other (?:cost|expense)',
]

# Compile patterns for efficiency
_EXPENSE_REGEX = re.compile('|'.join(EXPENSE_PATTERNS), re.IGNORECASE)


def is_expense_item(item_name: str) -> bool:
    """
    Check if an item is an expense that should typically be negative.
    
    Args:
        item_name: The name of the financial item
        
    Returns:
        True if the item is an expense that should be negative
    """
    return bool(_EXPENSE_REGEX.search(item_name))


# ============================================================================
# CASH FLOW SIGN NORMALIZATION: Outflow items that should be negative
# ============================================================================
# Cash outflows are typically reported as negative in vnstock.
# OCR may extract them as positive, so we need to flip the sign.
CASH_OUTFLOW_PATTERNS = [
    # Cash paid for purchases / Tiền chi mua hàng
    r'tiền chi (?:mua|trả)',
    r'tiền chi cho',
    r'chi trả cho',
    r'tiền trả cho',
    
    # Cash paid for wages, salaries / Tiền chi trả nhân viên
    r'tiền chi trả (?:cho )?(?:người )?(?:lao động|nhân viên|cán bộ)',
    
    # Cash paid for interest / Tiền lãi vay đã trả
    r'tiền lãi (?:vay )?đã trả',
    r'lãi vay đã trả',
    r'interest paid',
    
    # Cash paid for taxes / Tiền nộp thuế
    r'tiền (?:đã )?nộp thuế',
    r'thuế (?:tndn )?đã (?:nộp|trả)',
    r'tax(?:es)? paid',
    
    # Investment outflows / Chi đầu tư
    r'tiền chi (?:đầu tư|mua)',
    r'tiền chi (?:để )?mua sắm',
    r'tiền chi xây dựng',
    r'tiền mua (?:tài sản|cổ phiếu|trái phiếu)',
    
    # Loan repayments / Tiền trả nợ
    r'tiền trả nợ',
    r'tiền chi trả nợ',
    r'repayment',
    
    # Dividends paid / Cổ tức đã trả
    r'cổ tức (?:đã )?(?:trả|chi)',
    r'tiền chi trả cổ tức',
    r'dividend(?:s)? paid',
    
    # Other cash outflows
    r'tiền chi khác',
    r'chi phí .* đã trả',
]

# Compile patterns for efficiency
_CASH_OUTFLOW_REGEX = re.compile('|'.join(CASH_OUTFLOW_PATTERNS), re.IGNORECASE)


def is_cash_outflow(item_name: str) -> bool:
    """
    Check if a cash flow item is an outflow that should typically be negative.
    
    Args:
        item_name: The name of the cash flow item
        
    Returns:
        True if the item is a cash outflow that should be negative
    """
    return bool(_CASH_OUTFLOW_REGEX.search(item_name))


# ============================================================================
# HEADER ROW FILTERING: Patterns to identify section headers (not data rows)
# ============================================================================
# These patterns identify rows that are section headers, not actual financial items.
# Headers should be filtered out because they don't contain meaningful values.
HEADER_PATTERNS = [
    # Roman numeral prefixes (I., II., III., IV., V., etc.)
    r'^[IVX]+\.\s',
    
    # Letter prefixes (A., B., C., etc.)
    r'^[A-Z]\.\s',
    
    # All caps section headers (common in Vietnamese reports)
    r'^TÀI SẢN NGẮN HẠN$',
    r'^TÀI SẢN DÀI HẠN$',
    r'^TỔNG CỘNG TÀI SẢN$',
    r'^NỢ PHẢI TRẢ$',
    r'^NỢ NGẮN HẠN$',
    r'^NỢ DÀI HẠN$',
    r'^VỐN CHỦ SỞ HỮU$',
    r'^TỔNG CỘNG NGUỒN VỐN$',
    
    # English equivalents
    r'^CURRENT ASSETS$',
    r'^NON-?CURRENT ASSETS$',
    r'^TOTAL ASSETS$',
    r'^LIABILITIES$',
    r'^CURRENT LIABILITIES$',
    r'^NON-?CURRENT LIABILITIES$',
    r'^(?:SHAREHOLDERS?\'? )?EQUITY$',
    
    # Generic total/subtotal patterns
    r'^TỔNG\s',
    r'^CỘNG\s',
    r'^TOTAL\s',
    r'^SUBTOTAL\s',
]

# Compile patterns for efficiency
_HEADER_REGEX = re.compile('|'.join(HEADER_PATTERNS), re.IGNORECASE)


def is_header_row(item_name: str, item_code: Optional[str] = None) -> bool:
    """
    Check if a row is a section header that should be filtered out.
    
    A row is considered a header if:
    1. It matches known header patterns, OR
    2. It has no valid item_code and starts with a letter/roman numeral prefix
    
    Args:
        item_name: The name of the item
        item_code: The item code (e.g., "110", "01")
        
    Returns:
        True if this is a header row that should be filtered out
    """
    if not item_name:
        return True
    
    name_stripped = item_name.strip()
    
    # Check against known header patterns
    if _HEADER_REGEX.search(name_stripped):
        return True
    
    # If no item_code, check for letter/roman prefix patterns
    if not item_code or item_code.strip() == '':
        # Check for roman numeral or letter prefix
        if re.match(r'^[IVX]+\.\s', name_stripped) or re.match(r'^[A-Z]\.\s', name_stripped):
            return True
        # Check for numbered prefix without proper code (e.g., "1. Tiền" without item_code "110")
        if re.match(r'^\d+\.\s', name_stripped):
            return True
    
    return False


def has_valid_item_code(item_code: Optional[str]) -> bool:
    """
    Check if an item has a valid financial item code.
    
    Valid codes are typically:
    - 2-3 digit numbers (100, 110, 200, etc. for BS)
    - 2 digit numbers (01, 02, 10, 20, etc. for PL/CF)
    
    Args:
        item_code: The item code to validate
        
    Returns:
        True if the code appears to be a valid financial item code
    """
    if not item_code:
        return False
    
    code = str(item_code).strip()
    if not code:
        return False
    
    # Must be numeric
    if not re.match(r'^\d+$', code):
        return False
    
    # Reasonable length (1-4 digits)
    if len(code) < 1 or len(code) > 4:
        return False
    
    return True


class VnstockTransformer:
    """
    Transform vnstock DataFrame output to canonical format.
    """
    
    # Columns to skip (metadata, not financial items)
    SKIP_COLUMNS_YEARLY_VI = {'CP', 'Năm'}
    SKIP_COLUMNS_QUARTERLY_EN = {'ticker', 'yearReport', 'lengthReport'}
    SKIP_COLUMNS_CASHFLOW = {'ticker', 'yearReport'}
    
    def transform_yearly_report(
        self,
        stock_code: str,
        year: int,
        balance_sheet_df: pd.DataFrame,
        income_statement_df: pd.DataFrame,
        cash_flow_df: pd.DataFrame,
        lang: str = 'vi'
    ) -> Optional[FinancialReport]:
        """
        Transform yearly vnstock data to canonical format.
        """
        try:
            report = FinancialReport(
                stock_code=stock_code,
                year=year,
                quarter=None,
                source="vnstock",
                collected_at=datetime.now().isoformat()
            )
            
            # Transform each statement - detect year column per DataFrame
            report.balance_sheet = self._transform_statement(
                balance_sheet_df, year, self._detect_year_column(balance_sheet_df), "BS", lang
            )
            report.income_statement = self._transform_statement(
                income_statement_df, year, self._detect_year_column(income_statement_df), "PL", lang
            )
            report.cash_flow = self._transform_statement(
                cash_flow_df, year, self._detect_year_column(cash_flow_df), "CF", lang
            )
            
            logger.info(f"Transformed vnstock yearly report: {report.report_id}")
            logger.debug(f"  BS: {len(report.balance_sheet.items)} items")
            logger.debug(f"  PL: {len(report.income_statement.items)} items")
            logger.debug(f"  CF: {len(report.cash_flow.items)} items")
            
            return report
            
        except Exception as e:
            logger.error(f"Error transforming yearly report {stock_code} {year}: {e}")
            return None
    
    def transform_quarterly_report(
        self,
        stock_code: str,
        year: int,
        quarter: int,
        balance_sheet_df: pd.DataFrame,
        income_statement_df: pd.DataFrame = None,
        cash_flow_df: pd.DataFrame = None,
        lang: str = 'en'
    ) -> Optional[FinancialReport]:
        """
        Transform quarterly vnstock data to canonical format.
        """
        try:
            report = FinancialReport(
                stock_code=stock_code,
                year=year,
                quarter=quarter,
                source="vnstock",
                collected_at=datetime.now().isoformat()
            )
            
            # For quarterly data, we need to filter by yearReport and lengthReport
            report.balance_sheet = self._transform_quarterly_statement(
                balance_sheet_df, year, quarter, "BS", lang
            )
            
            if income_statement_df is not None:
                report.income_statement = self._transform_quarterly_statement(
                    income_statement_df, year, quarter, "PL", lang
                )
            
            if cash_flow_df is not None:
                report.cash_flow = self._transform_quarterly_statement(
                    cash_flow_df, year, quarter, "CF", lang
                )
            
            logger.info(f"Transformed vnstock quarterly report: {report.report_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error transforming quarterly report {stock_code} {year}Q{quarter}: {e}")
            return None
    
    def _detect_year_column(self, df: pd.DataFrame) -> str:
        """
        Detect the year column in a vnstock DataFrame.
        """
        if df is None or df.empty:
            return 'Năm'  # default
        
        if 'Năm' in df.columns:
            return 'Năm'
        elif 'yearReport' in df.columns:
            return 'yearReport'
        else:
            # Fallback: look for any column containing 'year' or 'năm'
            for col in df.columns:
                col_lower = col.lower()
                if 'year' in col_lower or 'năm' in col_lower:
                    return col
            logger.warning(f"Could not detect year column in DataFrame. Columns: {df.columns.tolist()[:5]}")
            return 'Năm'  # default fallback
    
    def _detect_skip_columns(self, df: pd.DataFrame) -> set:
        """
        Detect which metadata columns to skip based on what's in the DataFrame.
        Returns set of column names that are metadata (not financial items).
        """
        skip = set()
        
        # Add year columns
        if 'Năm' in df.columns:
            skip.add('Năm')
        if 'yearReport' in df.columns:
            skip.add('yearReport')
        
        # Add ticker/stock code columns
        if 'CP' in df.columns:
            skip.add('CP')
        if 'ticker' in df.columns:
            skip.add('ticker')
        
        # Add period/quarter columns
        if 'lengthReport' in df.columns:
            skip.add('lengthReport')
        if 'Kỳ' in df.columns:
            skip.add('Kỳ')
        
        return skip
    
    def _transform_statement(
        self,
        df: pd.DataFrame,
        year: int,
        year_col: str,
        statement_type: str,
        lang: str
    ) -> FinancialStatement:
        """Transform a single statement DataFrame to canonical format."""
        statement = FinancialStatement(statement_type)
        
        if df is None or df.empty:
            return statement
        
        # Find the row for the specific year
        year_mask = df[year_col] == year
        if not year_mask.any():
            logger.warning(f"Year {year} not found in {statement_type} DataFrame")
            return statement
        
        row = df[year_mask].iloc[0]
        
        # Determine which columns to skip
        skip_cols = self._detect_skip_columns(df)
        
        # Extract each financial item
        for col_name in df.columns:
            if col_name in skip_cols:
                continue
            
            value = row[col_name]
            
            # Skip NaN values
            if pd.isna(value):
                continue
            
            try:
                value_float = float(value)
            except (ValueError, TypeError):
                continue
            
            # Clean item name
            item_name = self._clean_item_name(col_name)
            
            # Determine unit from column name
            unit = self._detect_unit(col_name)
            
            # Convert to billions
            value_billions = normalize_to_billions(value_float, unit)
            
            statement.items.append(FinancialItem(
                item_name=item_name,
                value=value_billions,
                original_value=value_float,
                original_unit=unit
            ))
        
        return statement
    
    def _transform_quarterly_statement(
        self,
        df: pd.DataFrame,
        year: int,
        quarter: int,
        statement_type: str,
        lang: str
    ) -> FinancialStatement:
        """Transform quarterly statement DataFrame."""
        statement = FinancialStatement(statement_type)
        
        if df is None or df.empty:
            return statement
        
        # Detect column names
        year_col = 'Năm' if 'Năm' in df.columns else 'yearReport'
        quarter_col = 'Kỳ' if 'Kỳ' in df.columns else 'lengthReport'
        
        # Filter by year and quarter
        year_mask = df[year_col] == year
        quarter_mask = df[quarter_col] == quarter
        combined_mask = year_mask & quarter_mask
        
        if not combined_mask.any():
            logger.warning(f"Period {year}Q{quarter} not found in {statement_type}. "
                          f"Available: {df[[year_col, quarter_col]].drop_duplicates().head(5).to_dict('records')}")
            return statement
        
        row = df[combined_mask].iloc[0]
        
        # Skip metadata columns
        skip_cols = self._detect_skip_columns(df)
        
        for col_name in df.columns:
            if col_name in skip_cols:
                continue
            
            value = row[col_name]
            if pd.isna(value):
                continue
            
            try:
                value_float = float(value)
            except (ValueError, TypeError):
                continue
            
            item_name = self._clean_item_name(col_name)
            unit = self._detect_unit(col_name)

            value_billions = normalize_to_billions(value_float, "VND")
            
            statement.items.append(FinancialItem(
                item_name=item_name,
                value=value_billions,
                original_value=value_float,
                original_unit=unit
            ))
        
        return statement
    
    def _clean_item_name(self, col_name: str) -> str:
        """
        Clean column name to get item name.
        """
        import re
        
        # Remove unit indicators in parentheses
        cleaned = re.sub(r'\s*\([^)]*\)\s*$', '', col_name)
        
        return cleaned.strip()
    
    def _detect_unit(self, col_name: str) -> str:
        """Detect unit from column name."""
        col_lower = col_name.lower()
        
        if '(đồng)' in col_lower or '(dong)' in col_lower:
            return 'VND'
        elif '(bn. vnd)' in col_lower or '(bn.vnd)' in col_lower:
            return 'VND'
        elif '(tỷ)' in col_lower or '(ty)' in col_lower:
            return 'billions'
        elif '(%)' in col_lower:
            return 'percent'
        else:
            return 'VND'


def detect_unit_from_markdown(content: str) -> str:
    """
    Detect currency unit from markdown content.
    Finds the FIRST unit declaration in the document.
    """
    import re
    
    # All patterns with their unit mappings
    unit_patterns = [
        (r'[Đđ]ơn\s*vị\s*(tính)?\s*:?\s*[Tt]riệu\s*(VNĐ|VND|đồng)', "triệu VND"),
        (r'[Đđ]ơn\s*vị\s*(tính)?\s*:?\s*[Tt]ỷ\s*(VNĐ|VND|đồng)', "tỷ VND"),
        (r'[Đđ]ơn\s*vị\s*(tính)?\s*:?\s*[Nn]ghìn\s*(VNĐ|VND|đồng)', "nghìn VND"),
        (r'[Đđ]ơn\s*vị\s*(tính)?\s*:?\s*(VNĐ|VND|đồng)', "VND"),
        (r'ĐVT\s*:?\s*[Tt]riệu\s*(VNĐ|VND|đồng)', "triệu VND"),
        (r'ĐVT\s*:?\s*[Tt]ỷ\s*(VNĐ|VND|đồng)', "tỷ VND"),
        (r'ĐVT\s*:?\s*(VNĐ|VND|đồng)', "VND"),
    ]
    
    # Find all matches with their positions
    matches = []
    for pattern, unit in unit_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            matches.append((match.start(), unit, match.group(0)))
    
    # Return the unit from the earliest match (first in document)
    if matches:
        matches.sort(key=lambda x: x[0])  # Sort by position
        first_match = matches[0]
        logger.debug(f"Detected unit '{first_match[1]}' from: '{first_match[2]}' at position {first_match[0]}")
        return first_match[1]
    
    # Table headers as fallback
    header_patterns = [
        (r'\|\s*[^|]*Triệu\s*VN[DĐ][^|]*\|', "triệu VND"),
        (r'\|\s*[^|]*Tỷ\s*VN[DĐ][^|]*\|', "tỷ VND"),
        (r'\|\s*[^|]*Nghìn\s*VN[DĐ][^|]*\|', "nghìn VND"),
    ]
    
    for pattern, unit in header_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            logger.debug(f"Detected unit from table header: {unit}")
            return unit
    
    logger.debug("No unit detected, defaulting to VND")
    return "VND"  # Default


class OCRTransformer:
    """
    Transform OCR JSON output to canonical format.
    """
    
    @staticmethod
    def detect_unit(markdown_content: str) -> str:
        """Detect unit from markdown content."""
        return detect_unit_from_markdown(markdown_content)
    
    def transform_report(
        self,
        ocr_output: Dict,
        stock_code: str,
        year: int,
        quarter: Optional[int] = None,
        unit_override: Optional[str] = None
    ) -> FinancialReport:
        """
        Transform OCR JSON output to canonical format.
        """
        report = FinancialReport(
            stock_code=stock_code,
            year=year,
            quarter=quarter,
            source="ocr",
            collected_at=datetime.now().isoformat()
        )
        
        unit = unit_override or ocr_output.get("unit", "VND")
        logger.info(f"OCR Transformer using unit: {unit}")
        
        # Transform each statement with the detected unit
        report.balance_sheet = self._transform_statement(
            ocr_output.get("BS", []), "BS", unit
        )
        report.income_statement = self._transform_statement(
            ocr_output.get("PL", []), "PL", unit
        )
        report.cash_flow = self._transform_statement(
            ocr_output.get("CF", []), "CF", unit
        )
        
        logger.info(f"Transformed OCR report: {report.report_id}")
        logger.debug(f"  BS: {len(report.balance_sheet.items)} items")
        logger.debug(f"  PL: {len(report.income_statement.items)} items")
        logger.debug(f"  CF: {len(report.cash_flow.items)} items")
        
        return report
    
    def _transform_statement(
        self,
        items_list: List[Dict],
        statement_type: str,
        unit: str = "VND"
    ) -> FinancialStatement:
        """
        Transform OCR item list to FinancialStatement.
        
        Applies:
        1. Header row filtering - removes section headers
        2. Sign normalization - flips sign for expense items (PL only)
        3. Unit conversion - normalizes to billions
        """
        statement = FinancialStatement(statement_type)
        filtered_count = 0
        sign_flipped_count = 0
        
        for item in items_list:
            item_name = item.get("item_name", "")
            item_code = item.get("item_code")
            value = item.get("value")
            
            # Skip items without value
            if value is None:
                continue
            
            # Apply header row filtering
            if is_header_row(item_name, item_code):
                filtered_count += 1
                logger.debug(f"Filtered header row: {item_name}")
                continue
            
            try:
                value_float = float(value)
            except (ValueError, TypeError):
                continue
            
            # Convert to billions using the detected/specified unit
            value_billions = normalize_to_billions(value_float, unit)
            
            # Apply sign normalization for Income Statement expense items
            # VNStock reports expenses as negative, but OCR may extract as positive
            if statement_type == "PL" and is_expense_item(item_name):
                # Only flip if value is positive (likely extracted incorrectly)
                if value_billions > 0:
                    value_billions = -value_billions
                    sign_flipped_count += 1
                    logger.debug(f"Flipped sign for expense: {item_name} -> {value_billions}")
            
            # Apply sign normalization for Cash Flow outflow items
            # VNStock reports cash outflows as negative, but OCR may extract as positive
            if statement_type == "CF" and is_cash_outflow(item_name):
                # Only flip if value is positive (likely extracted incorrectly)
                if value_billions > 0:
                    value_billions = -value_billions
                    sign_flipped_count += 1
                    logger.debug(f"Flipped sign for cash outflow: {item_name} -> {value_billions}")
            
            statement.items.append(FinancialItem(
                item_name=item_name,
                value=value_billions,
                item_code=item_code,
                notes_ref=item.get("notes_ref"),
                original_value=value_float,
                original_unit=unit
            ))
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} header rows from {statement_type}")
        if sign_flipped_count > 0:
            logger.info(f"Flipped sign for {sign_flipped_count} items in {statement_type}")
        
        return statement


def detect_ytd_mismatch(
    ocr_statement: FinancialStatement,
    vnstock_statement: FinancialStatement,
    key_items: List[str] = None
) -> Dict:
    """
    Detect if OCR extracted YTD (cumulative) data instead of quarterly data.
    """
    if key_items is None:
        # Default key items that are likely to show YTD vs quarterly difference
        key_items = [
            r'doanh thu',
            r'revenue',
            r'lợi nhuận',
            r'profit',
            r'chi phí',
            r'cost',
            r'expense',
        ]
    
    key_pattern = re.compile('|'.join(key_items), re.IGNORECASE)
    
    ratios = []
    for ocr_item in ocr_statement.items:
        if not key_pattern.search(ocr_item.item_name):
            continue
        
        # Find matching vnstock item
        for vn_item in vnstock_statement.items:
            if key_pattern.search(vn_item.item_name):
                # Check if names are similar enough
                ocr_norm = ocr_item.item_name.lower()
                vn_norm = vn_item.item_name.lower()
                if any(kw in ocr_norm and kw in vn_norm for kw in ['doanh thu', 'lợi nhuận', 'chi phí', 'revenue', 'profit']):
                    if vn_item.value != 0 and ocr_item.value != 0:
                        ratio = abs(ocr_item.value / vn_item.value)
                        if 0.1 < ratio < 20:  # Filter out extreme outliers
                            ratios.append({
                                'ocr_name': ocr_item.item_name,
                                'vn_name': vn_item.item_name,
                                'ocr_value': ocr_item.value,
                                'vn_value': vn_item.value,
                                'ratio': ratio
                            })
                    break
    
    if not ratios:
        return {
            'is_ytd_mismatch': False,
            'avg_ratio': 1.0,
            'ratios': [],
            'quarter_estimate': None,
            'message': 'No matching items found for YTD detection'
        }
    
    avg_ratio = sum(r['ratio'] for r in ratios) / len(ratios)
    
    # Determine if it's a YTD mismatch
    # Q1: ratio ~1, Q2: ratio ~2, Q3: ratio ~3, Q4: ratio ~4
    is_ytd_mismatch = avg_ratio > 1.5  # If ratio > 1.5, likely YTD
    quarter_estimate = None
    
    if 1.8 <= avg_ratio <= 2.2:
        quarter_estimate = 2
    elif 2.8 <= avg_ratio <= 3.2:
        quarter_estimate = 3
    elif 3.5 <= avg_ratio <= 4.5:
        quarter_estimate = 4
    elif avg_ratio < 1.5:
        quarter_estimate = 1
    
    message = f"Average ratio: {avg_ratio:.2f}"
    if is_ytd_mismatch:
        message += f" - Likely YTD data (Q{quarter_estimate} cumulative)" if quarter_estimate else " - Possible YTD mismatch"
    else:
        message += " - Data appears to be quarterly"
    
    return {
        'is_ytd_mismatch': is_ytd_mismatch,
        'avg_ratio': avg_ratio,
        'ratios': ratios,
        'quarter_estimate': quarter_estimate,
        'message': message
    }


def load_vnstock_from_csv(
    balance_sheet_path: str,
    income_statement_path: str = None,
    cash_flow_path: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Load vnstock data from CSV files.
    """
    result = {}
    
    result['balance_sheet'] = pd.read_csv(balance_sheet_path)
    
    if income_statement_path:
        result['income_statement'] = pd.read_csv(income_statement_path)
    
    if cash_flow_path:
        result['cash_flow'] = pd.read_csv(cash_flow_path)
    
    return result
