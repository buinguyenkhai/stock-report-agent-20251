"""
Simple Evaluator

Direct evaluation against vnstock ground truth.
Since the parser outputs vnstock-aligned names, matching is trivial.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from logger import get_logger
from services.parser import ParsedReport, FinancialItem


logger = get_logger(__name__)


@dataclass
class ItemMatch:
    """Single item comparison result."""
    name: str  # vnstock canonical name
    ocr_value: Optional[float]
    ground_truth_value: Optional[float]
    matched: bool
    value_match: bool = False
    error_pct: Optional[float] = None
    
    def __post_init__(self):
        if self.matched and self.ocr_value is not None and self.ground_truth_value is not None:
            if self.ground_truth_value != 0:
                self.error_pct = abs(self.ocr_value - self.ground_truth_value) / abs(self.ground_truth_value) * 100
                self.value_match = self.error_pct < 1.0  # 1% tolerance
            else:
                self.value_match = abs(self.ocr_value) < 1000  # Near zero


@dataclass
class StatementEvaluation:
    """Evaluation results for one financial statement."""
    statement_type: str
    total_ground_truth: int
    total_ocr: int
    matched_count: int
    value_match_count: int
    matches: List[ItemMatch] = field(default_factory=list)
    
    @property
    def match_rate(self) -> float:
        """Percentage of ground truth items that were matched."""
        if self.total_ground_truth == 0:
            return 0.0
        return self.matched_count / self.total_ground_truth * 100
    
    @property
    def value_accuracy(self) -> float:
        """Percentage of matched items with correct values."""
        if self.matched_count == 0:
            return 0.0
        return self.value_match_count / self.matched_count * 100
    
    @property
    def mape(self) -> float:
        """Mean Absolute Percentage Error for matched items."""
        errors = [m.error_pct for m in self.matches if m.error_pct is not None]
        if not errors:
            return 0.0
        return sum(errors) / len(errors)


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    report_id: str
    balance_sheet: StatementEvaluation
    income_statement: StatementEvaluation
    cash_flow: StatementEvaluation
    warnings: List[str] = field(default_factory=list)
    
    @property
    def overall_match_rate(self) -> float:
        """Overall match rate across all statements."""
        total_gt = (
            self.balance_sheet.total_ground_truth + 
            self.income_statement.total_ground_truth + 
            self.cash_flow.total_ground_truth
        )
        total_matched = (
            self.balance_sheet.matched_count + 
            self.income_statement.matched_count + 
            self.cash_flow.matched_count
        )
        if total_gt == 0:
            return 0.0
        return total_matched / total_gt * 100
    
    @property
    def overall_value_accuracy(self) -> float:
        """Overall value accuracy across all statements."""
        total_matched = (
            self.balance_sheet.matched_count + 
            self.income_statement.matched_count + 
            self.cash_flow.matched_count
        )
        total_value_match = (
            self.balance_sheet.value_match_count + 
            self.income_statement.value_match_count + 
            self.cash_flow.value_match_count
        )
        if total_matched == 0:
            return 0.0
        return total_value_match / total_matched * 100
    
    @property
    def overall_mape(self) -> float:
        """Overall MAPE across all statements."""
        all_errors = []
        for stmt in [self.balance_sheet, self.income_statement, self.cash_flow]:
            for m in stmt.matches:
                if m.error_pct is not None:
                    all_errors.append(m.error_pct)
        if not all_errors:
            return 0.0
        return sum(all_errors) / len(all_errors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "report_id": self.report_id,
            "overall": {
                "match_rate": round(self.overall_match_rate, 2),
                "value_accuracy": round(self.overall_value_accuracy, 2),
                "mape": round(self.overall_mape, 2),
            },
            "balance_sheet": {
                "total_gt": self.balance_sheet.total_ground_truth,
                "matched": self.balance_sheet.matched_count,
                "match_rate": round(self.balance_sheet.match_rate, 2),
                "value_accuracy": round(self.balance_sheet.value_accuracy, 2),
                "mape": round(self.balance_sheet.mape, 2),
            },
            "income_statement": {
                "total_gt": self.income_statement.total_ground_truth,
                "matched": self.income_statement.matched_count,
                "match_rate": round(self.income_statement.match_rate, 2),
                "value_accuracy": round(self.income_statement.value_accuracy, 2),
                "mape": round(self.income_statement.mape, 2),
            },
            "cash_flow": {
                "total_gt": self.cash_flow.total_ground_truth,
                "matched": self.cash_flow.matched_count,
                "match_rate": round(self.cash_flow.match_rate, 2),
                "value_accuracy": round(self.cash_flow.value_accuracy, 2),
                "mape": round(self.cash_flow.mape, 2),
            },
            "warnings": self.warnings,
        }


class SimpleEvaluator:
    """
    Simple evaluator that compares parsed report against vnstock ground truth.
    
    Uses direct name matching since parser normalizes to vnstock vocabulary.
    """
    
    def __init__(self, ground_truth_dir: str = "data/ground_truth"):
        """
        Initialize evaluator.
        
        Args:
            ground_truth_dir: Directory containing ground truth CSV files.
        """
        self.ground_truth_dir = Path(ground_truth_dir)
    
    def load_ground_truth(
        self, 
        stock_code: str, 
        year: int, 
        quarter: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Load ground truth data from CSV files.
        
        Args:
            stock_code: Stock ticker (e.g., "FPT")
            year: Report year
            quarter: Report quarter (1-4)
            
        Returns:
            Dictionary with BS, PL, CF data as {item_name: value}.
        """
        gt_dir = self.ground_truth_dir / f"{stock_code}_{year}_Q{quarter}"
        
        if not gt_dir.exists():
            raise FileNotFoundError(f"Ground truth not found: {gt_dir}")
        
        result = {
            "balance_sheet": {},
            "income_statement": {},
            "cash_flow": {},
        }
        
        # Load each statement
        for stmt_name, filename in [
            ("balance_sheet", "balance_sheet.csv"),
            ("income_statement", "income_statement.csv"),
            ("cash_flow", "cash_flow.csv"),
        ]:
            path = gt_dir / filename
            if path.exists():
                df = pd.read_csv(path)
                
                # Filter to the specific quarter if multiple rows
                if "Kỳ" in df.columns and len(df) > 1:
                    df = df[df["Kỳ"] == quarter]
                
                if len(df) > 0:
                    row = df.iloc[0]
                    
                    # Extract values, skipping metadata columns
                    skip_cols = {"CP", "Năm", "Kỳ"}
                    for col in df.columns:
                        if col not in skip_cols:
                            # Clean column name (remove unit suffix)
                            clean_name = self._clean_column_name(col)
                            value = row[col]
                            if pd.notna(value) and value != 0:
                                result[stmt_name][clean_name] = float(value)
        
        logger.info(
            f"Loaded ground truth for {stock_code} {year} Q{quarter}: "
            f"BS={len(result['balance_sheet'])}, "
            f"PL={len(result['income_statement'])}, "
            f"CF={len(result['cash_flow'])}"
        )
        
        return result
    
    def _clean_column_name(self, col: str) -> str:
        """Remove unit suffix from column name."""
        # Remove common suffixes
        suffixes = [" (đồng)", " (%)", " (VND)"]
        for suffix in suffixes:
            if col.endswith(suffix):
                col = col[:-len(suffix)]
        return col.strip()
    
    def evaluate(
        self,
        parsed: ParsedReport,
        stock_code: str,
        year: int,
        quarter: int,
    ) -> EvaluationResult:
        """
        Evaluate parsed report against ground truth.
        
        Args:
            parsed: ParsedReport from the pipeline.
            stock_code: Stock ticker.
            year: Report year.
            quarter: Report quarter.
            
        Returns:
            EvaluationResult with detailed metrics.
        """
        report_id = f"{stock_code}_{year}_Q{quarter}"
        
        # Load ground truth
        try:
            ground_truth = self.load_ground_truth(stock_code, year, quarter)
        except FileNotFoundError as e:
            logger.error(f"Ground truth not found: {e}")
            return EvaluationResult(
                report_id=report_id,
                balance_sheet=StatementEvaluation("BS", 0, 0, 0, 0),
                income_statement=StatementEvaluation("PL", 0, 0, 0, 0),
                cash_flow=StatementEvaluation("CF", 0, 0, 0, 0),
                warnings=[str(e)]
            )
        
        # Evaluate each statement
        bs_eval = self._evaluate_statement(
            "BS",
            parsed.balance_sheet.items,
            ground_truth["balance_sheet"]
        )
        
        pl_eval = self._evaluate_statement(
            "PL",
            parsed.income_statement.items,
            ground_truth["income_statement"]
        )
        
        cf_eval = self._evaluate_statement(
            "CF",
            parsed.cash_flow.items,
            ground_truth["cash_flow"]
        )
        
        return EvaluationResult(
            report_id=report_id,
            balance_sheet=bs_eval,
            income_statement=pl_eval,
            cash_flow=cf_eval,
        )
    
    def _evaluate_statement(
        self,
        statement_type: str,
        ocr_items: List[FinancialItem],
        ground_truth: Dict[str, float],
    ) -> StatementEvaluation:
        """Evaluate a single financial statement."""
        matches = []
        matched_count = 0
        value_match_count = 0
        
        # Build OCR lookup (normalized name -> value)
        ocr_lookup = {}
        for item in ocr_items:
            if item.name and item.value is not None:
                ocr_lookup[item.name] = item.value
        
        # Match against ground truth
        for gt_name, gt_value in ground_truth.items():
            # Try exact match first
            ocr_value = ocr_lookup.get(gt_name)
            
            # Try fuzzy match if no exact match
            if ocr_value is None:
                ocr_value = self._fuzzy_lookup(gt_name, ocr_lookup)
            
            matched = ocr_value is not None
            
            match = ItemMatch(
                name=gt_name,
                ocr_value=ocr_value,
                ground_truth_value=gt_value,
                matched=matched,
            )
            
            if matched:
                matched_count += 1
                if match.value_match:
                    value_match_count += 1
            
            matches.append(match)
        
        return StatementEvaluation(
            statement_type=statement_type,
            total_ground_truth=len(ground_truth),
            total_ocr=len(ocr_items),
            matched_count=matched_count,
            value_match_count=value_match_count,
            matches=matches,
        )
    
    def _fuzzy_lookup(
        self, 
        target: str, 
        lookup: Dict[str, float]
    ) -> Optional[float]:
        """
        Try to find a fuzzy match for the target name.
        
        Uses simple token overlap for now.
        """
        target_lower = target.lower()
        target_tokens = set(target_lower.split())
        
        best_match = None
        best_score = 0.0
        
        for name, value in lookup.items():
            name_lower = name.lower()
            name_tokens = set(name_lower.split())
            
            # Jaccard similarity
            intersection = len(target_tokens & name_tokens)
            union = len(target_tokens | name_tokens)
            
            if union > 0:
                score = intersection / union
                if score > best_score and score >= 0.5:  # 50% threshold
                    best_score = score
                    best_match = value
        
        return best_match
    
    def print_report(self, result: EvaluationResult):
        """Print formatted evaluation report."""
        print(f"\n{'='*60}")
        print(f"EVALUATION: {result.report_id}")
        print('='*60)
        
        print("\nOverall Metrics:")
        print(f"  Match Rate:      {result.overall_match_rate:.1f}%")
        print(f"  Value Accuracy:  {result.overall_value_accuracy:.1f}%")
        print(f"  MAPE:            {result.overall_mape:.2f}%")
        
        print("\nPer-Statement Breakdown:")
        print("-"*60)
        print(f"{'Statement':<15} {'Match Rate':>12} {'Val Acc':>10} {'MAPE':>10} {'Items':>10}")
        print("-"*60)
        
        for name, stmt in [
            ("Balance Sheet", result.balance_sheet),
            ("Income Stmt", result.income_statement),
            ("Cash Flow", result.cash_flow),
        ]:
            print(
                f"{name:<15} "
                f"{stmt.match_rate:>10.1f}% "
                f"{stmt.value_accuracy:>10.1f}% "
                f"{stmt.mape:>9.2f}% "
                f"{stmt.matched_count:>3}/{stmt.total_ground_truth:<3}"
            )
        print("-"*60)
        
        # Show top errors
        all_errors = []
        for stmt in [result.balance_sheet, result.income_statement, result.cash_flow]:
            for m in stmt.matches:
                if m.matched and m.error_pct is not None and m.error_pct > 1.0:
                    all_errors.append(m)
        
        if all_errors:
            print("\nTop Value Errors (showing first 5):")
            all_errors.sort(key=lambda x: x.error_pct or 0, reverse=True)
            for m in all_errors[:5]:
                print(f"  • {m.name[:40]}")
                print(f"    OCR: {m.ocr_value/1e9:.2f}B | GT: {m.ground_truth_value/1e9:.2f}B | Error: {m.error_pct:.1f}%")


# Convenience Functions

def evaluate_report(
    parsed: ParsedReport,
    stock_code: str,
    year: int,
    quarter: int,
) -> EvaluationResult:
    """
    Evaluate a parsed report against vnstock ground truth.
    """
    evaluator = SimpleEvaluator()
    return evaluator.evaluate(parsed, stock_code, year, quarter)
