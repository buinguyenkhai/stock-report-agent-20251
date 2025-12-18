from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class FinancialItem:
    """
    Canonical format for a single financial line item.
    Both vnstock and OCR data are transformed to this format.
    Values are stored in billions (tỷ đồng) for consistency.
    """
    item_name: str              # "Tổng tài sản", "Total Assets"
    value: float                # In billions (tỷ đồng)
    item_code: Optional[str] = None    # "270"
    notes_ref: Optional[str] = None    # Reference to notes (OCR only)
    original_value: Optional[float] = None  # Original value before normalization
    original_unit: Optional[str] = None     # "VND", "billions", etc.

    def __repr__(self):
        return f"FinancialItem({self.item_name}: {self.value:.2f}B)"


@dataclass
class FinancialStatement:
    """A single financial statement (BS, PL, or CF)."""
    statement_type: str         # "BS", "PL", "CF"
    items: List[FinancialItem] = field(default_factory=list)
    
    def get_item_by_name(self, name: str) -> Optional[FinancialItem]:
        """Find item by exact name match."""
        for item in self.items:
            if item.item_name == name:
                return item
        return None
    
    def get_item_by_code(self, code: str) -> Optional[FinancialItem]:
        """Find item by code (OCR data only)."""
        for item in self.items:
            if item.item_code == code:
                return item
        return None


@dataclass
class FinancialReport:
    """
    Canonical format for a complete financial report.
    Contains all three statements for a specific period.
    """
    stock_code: str             # "FPT", "VCI"
    year: int                   # 2024
    quarter: Optional[int] = None  # 1-4 for quarterly, None for yearly
    report_scope: str = "consolidated"  # "consolidated" or "parent"
    period_type: str = "quarterly"      # "quarterly" or "cumulative"
    
    balance_sheet: FinancialStatement = field(
        default_factory=lambda: FinancialStatement("BS")
    )
    income_statement: FinancialStatement = field(
        default_factory=lambda: FinancialStatement("PL")
    )
    cash_flow: FinancialStatement = field(
        default_factory=lambda: FinancialStatement("CF")
    )
    
    source: str = ""            # "vnstock" or "ocr"
    collected_at: str = ""      # ISO timestamp
    
    @property
    def period(self) -> str:
        """Return period string like '2024Q3' or '2024'."""
        if self.quarter:
            return f"{self.year}Q{self.quarter}"
        return str(self.year)
    
    @property
    def report_id(self) -> str:
        """Unique identifier for this report."""
        return f"{self.stock_code}_{self.period}"
    
    def get_statement(self, statement_type: str) -> FinancialStatement:
        """Get statement by type ('BS', 'PL', 'CF')."""
        mapping = {
            "BS": self.balance_sheet,
            "PL": self.income_statement,
            "CF": self.cash_flow
        }
        return mapping.get(statement_type.upper(), FinancialStatement(statement_type))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "stock_code": self.stock_code,
            "year": self.year,
            "quarter": self.quarter,
            "period": self.period,
            "report_scope": self.report_scope,
            "period_type": self.period_type,
            "source": self.source,
            "collected_at": self.collected_at,
            "balance_sheet": [
                {
                    "item_name": item.item_name,
                    "value": item.value,
                    "item_code": item.item_code,
                    "notes_ref": item.notes_ref
                }
                for item in self.balance_sheet.items
            ],
            "income_statement": [
                {
                    "item_name": item.item_name,
                    "value": item.value,
                    "item_code": item.item_code,
                    "notes_ref": item.notes_ref
                }
                for item in self.income_statement.items
            ],
            "cash_flow": [
                {
                    "item_name": item.item_name,
                    "value": item.value,
                    "item_code": item.item_code,
                    "notes_ref": item.notes_ref
                }
                for item in self.cash_flow.items
            ]
        }

UNIT_MULTIPLIERS = {
    # Full VND/đồng -> billions (divide by 1 billion = 1e9)
    "VND": 1e-9,
    "dong": 1e-9,
    "đồng": 1e-9,
    
    # Thousands -> billions (divide by 1 million = 1e6)
    "nghìn VND": 1e-6,
    "thousands": 1e-6,
    
    # Millions -> billions (divide by 1 thousand = 1e3)
    "triệu VND": 1e-3,
    "millions": 1e-3,
    "Triệu VND": 1e-3,
    
    # Already in billions
    "tỷ VND": 1.0,
    "billions": 1.0,
    "Tỷ VND": 1.0,
    "Bn. VND": 1.0,
}


def normalize_to_billions(value: float, unit: str = "VND") -> float:
    """
    Normalize value to billions (tỷ đồng).
    """
    multiplier = UNIT_MULTIPLIERS.get(unit, 1e-12)  # Default to full VND
    return value * multiplier
