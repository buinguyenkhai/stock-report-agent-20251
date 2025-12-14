import pandas as pd
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
    """
    import re
    
    patterns = [
        # "Đơn vị tính: triệu VNĐ"
        (r'[Đđ]ơn\s*vị\s*(tính)?\s*:?\s*[Tt]riệu\s*(VNĐ|VND|đồng)', "triệu VND"),
        (r'[Đđ]ơn\s*vị\s*(tính)?\s*:?\s*[Tt]ỷ\s*(VNĐ|VND|đồng)', "tỷ VND"),
        (r'[Đđ]ơn\s*vị\s*(tính)?\s*:?\s*[Nn]ghìn\s*(VNĐ|VND|đồng)', "nghìn VND"),
        (r'[Đđ]ơn\s*vị\s*(tính)?\s*:?\s*(VNĐ|VND|đồng)', "VND"),
    ]
    
    for pattern, unit in patterns:
        if re.search(pattern, content):
            logger.debug(f"Detected unit from explicit declaration: {unit}")
            return unit
    
    # Table headers
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
        """Transform OCR item list to FinancialStatement."""
        statement = FinancialStatement(statement_type)
        
        for item in items_list:
            value = item.get("value")
            
            if value is None:
                continue
            
            try:
                value_float = float(value)
            except (ValueError, TypeError):
                continue
            
            # Convert to billions using the detected/specified unit
            value_billions = normalize_to_billions(value_float, unit)
            
            statement.items.append(FinancialItem(
                item_name=item.get("item_name", ""),
                value=value_billions,
                item_code=item.get("item_code"),
                notes_ref=item.get("notes_ref"),
                original_value=value_float,
                original_unit=unit
            ))
        
        return statement


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
