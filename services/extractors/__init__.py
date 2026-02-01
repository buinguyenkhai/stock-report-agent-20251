"""
Financial Report Extractors

Parallel extractors for different components of financial reports.
All extractors use cheap, long-context LLMs for focused extraction tasks.
"""

from .base import BaseExtractor, ExtractionResult
from .financial_tables import FinancialTablesExtractor
from .balance_sheet import BalanceSheetExtractor
from .income_statement import IncomeStatementExtractor
from .cash_flow import CashFlowExtractor
from .other_text import OtherTextExtractor
from .metadata import MetadataExtractor

__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "FinancialTablesExtractor",
    "BalanceSheetExtractor",
    "IncomeStatementExtractor",
    "CashFlowExtractor",
    "OtherTextExtractor",
    "MetadataExtractor",
]
