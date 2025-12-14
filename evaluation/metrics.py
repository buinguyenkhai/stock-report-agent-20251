from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .canonical_format import FinancialStatement, FinancialReport, FinancialItem
from .matchers import LLMBasedMatcher, normalize_name
from logger import get_logger

logger = get_logger(__name__)

@dataclass
class ValueComparison:
    """Result of comparing two values."""
    ocr_item_name: str
    vnstock_item_name: str
    ocr_value: float
    vnstock_value: float
    absolute_error: float
    relative_error_percent: float
    match_similarity: float
    is_correct: bool


@dataclass
class SectionEvaluation:
    """Evaluation results for one financial statement section."""
    section_type: str               # "BS", "PL", "CF"
    
    # Counts
    total_ocr_items: int
    total_vnstock_items: int
    matched_items: int
    correct_values: int
    
    # Rates (0.0 to 1.0)
    match_rate: float
    value_accuracy: float
    
    # Error metrics
    mape: float                     # Mean Absolute Percentage Error (%)
    max_error_percent: float
    
    # Details
    comparisons: List[ValueComparison] = field(default_factory=list)
    unmatched_ocr: List[str] = field(default_factory=list)
    missing_vnstock: List[str] = field(default_factory=list)
    
    @property
    def missing_rate(self) -> float:
        """Percentage of vnstock items not found in OCR."""
        if self.total_vnstock_items == 0:
            return 0.0
        return len(self.missing_vnstock) / self.total_vnstock_items
    
    @property
    def hallucination_rate(self) -> float:
        """Percentage of OCR items not in vnstock."""
        if self.total_ocr_items == 0:
            return 0.0
        return len(self.unmatched_ocr) / self.total_ocr_items
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "section_type": self.section_type,
            "total_ocr_items": self.total_ocr_items,
            "total_vnstock_items": self.total_vnstock_items,
            "matched_items": self.matched_items,
            "correct_values": self.correct_values,
            "match_rate": round(self.match_rate, 4),
            "value_accuracy": round(self.value_accuracy, 4),
            "mape": round(self.mape, 4),
            "max_error_percent": round(self.max_error_percent, 4),
            "missing_rate": round(self.missing_rate, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "unmatched_ocr": self.unmatched_ocr,
            "missing_vnstock": self.missing_vnstock,
        }


@dataclass
class ReportEvaluation:
    """Complete evaluation results for one financial report."""
    report_id: str
    stock_code: str
    period: str
    
    # Per-section results
    balance_sheet: Optional[SectionEvaluation] = None
    income_statement: Optional[SectionEvaluation] = None
    cash_flow: Optional[SectionEvaluation] = None
    
    # Overall metrics (weighted by item count)
    overall_match_rate: float = 0.0
    overall_value_accuracy: float = 0.0
    overall_mape: float = 0.0
    
    def get_section(self, section_type: str) -> Optional[SectionEvaluation]:
        """Get section by type."""
        mapping = {
            "BS": self.balance_sheet,
            "PL": self.income_statement,
            "CF": self.cash_flow,
        }
        return mapping.get(section_type.upper())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "report_id": self.report_id,
            "stock_code": self.stock_code,
            "period": self.period,
            "overall": {
                "match_rate": round(self.overall_match_rate, 4),
                "value_accuracy": round(self.overall_value_accuracy, 4),
                "mape": round(self.overall_mape, 4),
            },
            "sections": {}
        }
        
        if self.balance_sheet:
            result["sections"]["BS"] = self.balance_sheet.to_dict()
        if self.income_statement:
            result["sections"]["PL"] = self.income_statement.to_dict()
        if self.cash_flow:
            result["sections"]["CF"] = self.cash_flow.to_dict()
        
        return result

# Items to skip from comparison (per-share values, ratios, etc.)
# These items are in different units (VND/share instead of document unit)
SKIP_ITEMS_PATTERNS = [
    "lãi cơ bản trên cổ phiếu",  # EPS - Basic earnings per share
    "lãi suy giảm trên cổ phiếu",  # Diluted EPS
    "lãi trên cổ phiếu",  # Earnings per share
    "eps",
    "tăng trưởng",  # Growth rates (%)
]

def should_skip_item(item_name: str) -> bool:
    """Check if an item should be skipped from comparison."""
    name_lower = item_name.lower()
    for pattern in SKIP_ITEMS_PATTERNS:
        if pattern in name_lower:
            return True
    return False

def deduplicate_items(items: List[FinancialItem]) -> List[FinancialItem]:
    """
    Remove duplicate items from list based on normalized name similarity.
    """
    seen_normalized = {}
    unique_items = []
    
    for item in items:
        normalized = normalize_name(item.item_name)
        
        # Check if we've seen a very similar item
        is_duplicate = False
        for seen_name in seen_normalized:
            # Use high threshold to detect near-duplicates
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, normalized, seen_name).ratio()
            if similarity >= 0.95:  # Very high threshold for duplicates
                is_duplicate = True
                logger.debug(f"Removing duplicate: '{item.item_name}' similar to '{seen_normalized[seen_name].item_name}'")
                break
        
        if not is_duplicate:
            seen_normalized[normalized] = item
            unique_items.append(item)
    
    if len(items) != len(unique_items):
        logger.debug(f"Deduplicated {len(items)} items to {len(unique_items)}")
    
    return unique_items


def evaluate_section(
    ocr_statement: FinancialStatement,
    vnstock_statement: FinancialStatement,
    matcher,
    tolerance: float = 0.05,
    use_absolute_values: bool = True,
    section_type: str = "BS"
) -> SectionEvaluation:
    """
    Evaluate one section (BS, PL, or CF).
    """
    ocr_items = ocr_statement.items
    vnstock_items = vnstock_statement.items
    
    # Deduplicate OCR items to avoid wrong matching when same item extracted twice
    ocr_items = deduplicate_items(ocr_items)
    
    # Filter out items that should be skipped (EPS, ratios, etc.)
    vnstock_items_filtered = [
        item for item in vnstock_items 
        if not should_skip_item(item.item_name)
    ]
    
    skipped_count = len(vnstock_items) - len(vnstock_items_filtered)
    if skipped_count > 0:
        logger.debug(f"Skipped {skipped_count} items (EPS/ratios) from {ocr_statement.statement_type}")
    
    # Match items by name using the LLM matcher
    match_result = matcher.match_all(ocr_items, vnstock_items_filtered, section=section_type)
    matched_pairs = match_result["matched"]
    unmatched_ocr = [item.item_name for item in match_result["unmatched_ocr"]]
    missing_vnstock = [item.item_name for item in match_result["unmatched_vnstock"]]
    
    # Compare values for matched pairs
    comparisons = []
    correct_count = 0
    relative_errors = []
    max_error = 0.0
    
    for ocr_item, vn_item, similarity in matched_pairs:
        # Use absolute values for comparison to handle sign convention differences
        # vnstock stores costs as negative, OCR typically extracts positive values
        if use_absolute_values:
            ocr_val = abs(ocr_item.value)
            vn_val = abs(vn_item.value)
        else:
            ocr_val = ocr_item.value
            vn_val = vn_item.value
        
        # Calculate error
        if vn_val == 0:
            # Avoid division by zero
            rel_error = 0.0 if ocr_val == 0 else 100.0
        else:
            rel_error = abs(ocr_val - vn_val) / abs(vn_val) * 100
        
        abs_error = abs(ocr_val - vn_val)
        is_correct = rel_error <= (tolerance * 100)
        
        if is_correct:
            correct_count += 1
        
        relative_errors.append(rel_error)
        max_error = max(max_error, rel_error)
        
        comparisons.append(ValueComparison(
            ocr_item_name=ocr_item.item_name,
            vnstock_item_name=vn_item.item_name,
            ocr_value=ocr_item.value,
            vnstock_value=vn_item.value,
            absolute_error=abs_error,
            relative_error_percent=rel_error,
            match_similarity=similarity,
            is_correct=is_correct
        ))
    
    # Calculate rates
    total_vn = len(vnstock_items)
    matched_count = len(matched_pairs)
    
    match_rate = matched_count / total_vn if total_vn > 0 else 0.0
    value_accuracy = correct_count / matched_count if matched_count > 0 else 0.0
    mape = sum(relative_errors) / len(relative_errors) if relative_errors else 0.0
    
    return SectionEvaluation(
        section_type=ocr_statement.statement_type,
        total_ocr_items=len(ocr_items),
        total_vnstock_items=total_vn,
        matched_items=matched_count,
        correct_values=correct_count,
        match_rate=match_rate,
        value_accuracy=value_accuracy,
        mape=mape,
        max_error_percent=max_error,
        comparisons=comparisons,
        unmatched_ocr=unmatched_ocr,
        missing_vnstock=missing_vnstock
    )


def evaluate_report(
    ocr_report: FinancialReport,
    vnstock_report: FinancialReport,
    tolerance: float = 0.05,
) -> ReportEvaluation:
    """
    Evaluate complete financial report.
    """
    matcher = LLMBasedMatcher()
    
    # Evaluate each section
    bs_eval = evaluate_section(
        ocr_report.balance_sheet,
        vnstock_report.balance_sheet,
        matcher,
        tolerance,
        section_type="BS"
    )
    
    pl_eval = evaluate_section(
        ocr_report.income_statement,
        vnstock_report.income_statement,
        matcher,
        tolerance,
        section_type="PL"
    )
    
    cf_eval = evaluate_section(
        ocr_report.cash_flow,
        vnstock_report.cash_flow,
        matcher,
        tolerance,
        section_type="CF"
    )
    
    # Calculate overall metrics (weighted by vnstock item count)
    sections = [bs_eval, pl_eval, cf_eval]
    total_vn_items = sum(s.total_vnstock_items for s in sections)
    total_matched = sum(s.matched_items for s in sections)
    total_correct = sum(s.correct_values for s in sections)
    
    overall_match_rate = total_matched / total_vn_items if total_vn_items > 0 else 0.0
    overall_value_accuracy = total_correct / total_matched if total_matched > 0 else 0.0
    
    # Weighted MAPE
    weighted_mape = 0.0
    if total_vn_items > 0:
        for s in sections:
            weight = s.total_vnstock_items / total_vn_items
            weighted_mape += s.mape * weight
    
    return ReportEvaluation(
        report_id=ocr_report.report_id,
        stock_code=ocr_report.stock_code,
        period=ocr_report.period,
        balance_sheet=bs_eval,
        income_statement=pl_eval,
        cash_flow=cf_eval,
        overall_match_rate=overall_match_rate,
        overall_value_accuracy=overall_value_accuracy,
        overall_mape=weighted_mape
    )

def print_evaluation_summary(evaluation: ReportEvaluation):
    """Print formatted evaluation summary to console."""
    print("\n" + "=" * 70)
    print(f"EVALUATION: {evaluation.report_id}")
    print("=" * 70)
    
    print("\nOverall Metrics:")
    print(f"  Match Rate:      {evaluation.overall_match_rate:.1%}")
    print(f"  Value Accuracy:  {evaluation.overall_value_accuracy:.1%}")
    print(f"  MAPE:            {evaluation.overall_mape:.2f}%")
    
    print("\nPer-Section Breakdown:")
    print("-" * 70)
    print(f"{'Section':<15} {'Match Rate':<12} {'Val Acc':<12} {'MAPE':<10} {'Items':<15}")
    print("-" * 70)
    
    for section_type, section in [
        ("Balance Sheet", evaluation.balance_sheet),
        ("Income Stmt", evaluation.income_statement),
        ("Cash Flow", evaluation.cash_flow)
    ]:
        if section:
            items_str = f"{section.matched_items}/{section.total_vnstock_items}"
            print(f"{section_type:<15} {section.match_rate:<12.1%} "
                  f"{section.value_accuracy:<12.1%} {section.mape:<10.2f}% {items_str:<15}")
    
    print("-" * 70)
    
    # Show top errors
    all_errors = []
    for section in [evaluation.balance_sheet, evaluation.income_statement, evaluation.cash_flow]:
        if section:
            for comp in section.comparisons:
                if not comp.is_correct:
                    all_errors.append(comp)
    
    if all_errors:
        print("\nTop Value Errors (showing first 5):")
        sorted_errors = sorted(all_errors, key=lambda x: x.relative_error_percent, reverse=True)[:5]
        for err in sorted_errors:
            print(f"  • {err.ocr_item_name[:40]:<40}")
            print(f"    OCR: {err.ocr_value:,.2f}B | VNStock: {err.vnstock_value:,.2f}B | Error: {err.relative_error_percent:.1f}%")
    
    print("=" * 70 + "\n")
