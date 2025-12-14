import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

from .canonical_format import FinancialReport
from .data_transformers import VnstockTransformer, OCRTransformer
from .metrics import evaluate_report, ReportEvaluation
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""
    vnstock_dir: str                # Directory with vnstock CSV exports
    ocr_dir: str                    # Directory with OCR JSON outputs
    output_dir: str                 # Directory to save results
    
    # Evaluation parameters
    tolerance: float = 0.05         # Value tolerance
    similarity_threshold: float = 0.75  # Name matching threshold
    
    # Filters (optional)
    stock_codes: Optional[List[str]] = None
    years: Optional[List[int]] = None
    quarters: Optional[List[int]] = None


@dataclass
class AggregateResults:
    """Aggregate results across all evaluated reports."""
    total_reports: int = 0
    successful_reports: int = 0
    failed_reports: int = 0
    
    # Overall averages
    avg_match_rate: float = 0.0
    avg_value_accuracy: float = 0.0
    avg_mape: float = 0.0
    
    # Per-section averages
    bs_avg_match_rate: float = 0.0
    bs_avg_value_accuracy: float = 0.0
    pl_avg_match_rate: float = 0.0
    pl_avg_value_accuracy: float = 0.0
    cf_avg_match_rate: float = 0.0
    cf_avg_value_accuracy: float = 0.0
    
    # Best/worst
    best_report_id: str = ""
    best_value_accuracy: float = 0.0
    worst_report_id: str = ""
    worst_value_accuracy: float = 1.0
    
    # Individual results
    report_evaluations: List[ReportEvaluation] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON."""
        return {
            "summary": {
                "total_reports": self.total_reports,
                "successful_reports": self.successful_reports,
                "failed_reports": self.failed_reports,
                "avg_match_rate": round(self.avg_match_rate, 4),
                "avg_value_accuracy": round(self.avg_value_accuracy, 4),
                "avg_mape": round(self.avg_mape, 4),
            },
            "per_section": {
                "BS": {"match_rate": round(self.bs_avg_match_rate, 4), "value_accuracy": round(self.bs_avg_value_accuracy, 4)},
                "PL": {"match_rate": round(self.pl_avg_match_rate, 4), "value_accuracy": round(self.pl_avg_value_accuracy, 4)},
                "CF": {"match_rate": round(self.cf_avg_match_rate, 4), "value_accuracy": round(self.cf_avg_value_accuracy, 4)},
            },
            "best_report": {"id": self.best_report_id, "value_accuracy": round(self.best_value_accuracy, 4)},
            "worst_report": {"id": self.worst_report_id, "value_accuracy": round(self.worst_value_accuracy, 4)},
            "reports": [r.to_dict() for r in self.report_evaluations]
        }


class OCRPipelineEvaluator:
    """
    Main evaluator class for OCR pipeline.
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.vnstock_transformer = VnstockTransformer()
        self.ocr_transformer = OCRTransformer()
    
    def discover_reports(self) -> List[Dict]:
        """
        Discover available reports from vnstock directory.
        Expects files named: {stock_code}_balance_sheet_{period}.csv
        """
        vnstock_path = Path(self.config.vnstock_dir)
        reports = []
        seen = set()
        
        for csv_file in vnstock_path.glob("*_balance_sheet_*.csv"):
            parts = csv_file.stem.split("_")
            if len(parts) < 3:
                continue
            
            stock_code = parts[0].upper()
            period_type = parts[-1]  # "year" or "quarter"

            try:
                df = pd.read_csv(csv_file)
                
                if period_type == "year":
                    year_col = 'Năm' if 'Năm' in df.columns else 'yearReport'
                    if year_col in df.columns:
                        for year in df[year_col].unique():
                            key = f"{stock_code}_{year}"
                            if key not in seen:
                                seen.add(key)
                                reports.append({
                                    "stock_code": stock_code,
                                    "year": int(year),
                                    "quarter": None,
                                    "period_type": "year"
                                })
                
                elif period_type == "quarter":
                    if 'yearReport' in df.columns and 'lengthReport' in df.columns:
                        for _, row in df[['yearReport', 'lengthReport']].drop_duplicates().iterrows():
                            year = int(row['yearReport'])
                            quarter = int(row['lengthReport'])
                            key = f"{stock_code}_{year}Q{quarter}"
                            if key not in seen:
                                seen.add(key)
                                reports.append({
                                    "stock_code": stock_code,
                                    "year": year,
                                    "quarter": quarter,
                                    "period_type": "quarter"
                                })
            
            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")
        
        # Apply filters
        if self.config.stock_codes:
            reports = [r for r in reports if r["stock_code"] in self.config.stock_codes]
        if self.config.years:
            reports = [r for r in reports if r["year"] in self.config.years]
        if self.config.quarters:
            reports = [r for r in reports if r.get("quarter") in self.config.quarters]
        
        logger.info(f"Discovered {len(reports)} reports for evaluation")
        return reports
    
    def load_vnstock_report(
        self,
        stock_code: str,
        year: int,
        quarter: Optional[int] = None
    ) -> Optional[FinancialReport]:
        """Load and transform vnstock data."""
        vnstock_path = Path(self.config.vnstock_dir)
        period_type = "quarter" if quarter else "year"
        
        # Load CSVs
        bs_path = vnstock_path / f"{stock_code.lower()}_balance_sheet_{period_type}.csv"
        is_path = vnstock_path / f"{stock_code.lower()}_income_statement_{period_type}.csv"
        cf_path = vnstock_path / f"{stock_code.lower()}_cash_flow_{period_type}.csv"
        
        try:
            bs_df = pd.read_csv(bs_path) if bs_path.exists() else None
            is_df = pd.read_csv(is_path) if is_path.exists() else None
            cf_df = pd.read_csv(cf_path) if cf_path.exists() else None
            
            if bs_df is None:
                logger.warning(f"Balance sheet not found: {bs_path}")
                return None
            
            if quarter:
                return self.vnstock_transformer.transform_quarterly_report(
                    stock_code=stock_code,
                    year=year,
                    quarter=quarter,
                    balance_sheet_df=bs_df,
                    income_statement_df=is_df,
                    cash_flow_df=cf_df,
                    lang='en'
                )
            else:
                return self.vnstock_transformer.transform_yearly_report(
                    stock_code=stock_code,
                    year=year,
                    balance_sheet_df=bs_df,
                    income_statement_df=is_df,
                    cash_flow_df=cf_df,
                    lang='vi'
                )
        
        except Exception as e:
            logger.error(f"Error loading vnstock data: {e}")
            return None
    
    def load_ocr_report(
        self,
        stock_code: str,
        year: int,
        quarter: Optional[int] = None
    ) -> Optional[FinancialReport]:
        """Load and transform OCR prediction."""
        ocr_path = Path(self.config.ocr_dir)
        
        # Try different filename patterns
        if quarter:
            patterns = [
                f"{stock_code}_{year}Q{quarter}.json",
                f"{stock_code}_{year}_Q{quarter}.json",
                f"{stock_code.lower()}_{year}Q{quarter}.json",
            ]
        else:
            patterns = [
                f"{stock_code}_{year}.json",
                f"{stock_code.lower()}_{year}.json",
            ]
        
        json_path = None
        for pattern in patterns:
            candidate = ocr_path / pattern
            if candidate.exists():
                json_path = candidate
                break
        
        if not json_path:
            logger.warning(f"OCR prediction not found for {stock_code} {year} Q{quarter}")
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            
            return self.ocr_transformer.transform_report(
                ocr_output=ocr_data,
                stock_code=stock_code,
                year=year,
                quarter=quarter
            )
        
        except Exception as e:
            logger.error(f"Error loading OCR data from {json_path}: {e}")
            return None
    
    def evaluate_single(
        self,
        stock_code: str,
        year: int,
        quarter: Optional[int] = None
    ) -> Optional[ReportEvaluation]:
        """Evaluate a single report."""
        # Load both reports
        vnstock_report = self.load_vnstock_report(stock_code, year, quarter)
        ocr_report = self.load_ocr_report(stock_code, year, quarter)
        
        if not vnstock_report:
            logger.warning(f"Missing vnstock data for {stock_code} {year} Q{quarter}")
            return None
        
        if not ocr_report:
            logger.warning(f"Missing OCR data for {stock_code} {year} Q{quarter}")
            return None
        
        # Evaluate
        evaluation = evaluate_report(
            ocr_report=ocr_report,
            vnstock_report=vnstock_report,
            tolerance=self.config.tolerance,
            similarity_threshold=self.config.similarity_threshold
        )
        
        return evaluation
    
    def run(self) -> AggregateResults:
        """Run evaluation on all discovered reports."""
        reports = self.discover_reports()
        results = AggregateResults(total_reports=len(reports))
        
        # Track section-level metrics for averaging
        bs_match_rates, bs_value_accs = [], []
        pl_match_rates, pl_value_accs = [], []
        cf_match_rates, cf_value_accs = [], []
        
        for report_info in reports:
            stock_code = report_info["stock_code"]
            year = report_info["year"]
            quarter = report_info.get("quarter")
            
            period_str = f"{year}Q{quarter}" if quarter else str(year)
            logger.info(f"Evaluating {stock_code} {period_str}...")
            
            evaluation = self.evaluate_single(stock_code, year, quarter)
            
            if evaluation is None:
                results.failed_reports += 1
                continue
            
            results.successful_reports += 1
            results.report_evaluations.append(evaluation)
            
            # Track for aggregation
            if evaluation.balance_sheet:
                bs_match_rates.append(evaluation.balance_sheet.match_rate)
                bs_value_accs.append(evaluation.balance_sheet.value_accuracy)
            if evaluation.income_statement:
                pl_match_rates.append(evaluation.income_statement.match_rate)
                pl_value_accs.append(evaluation.income_statement.value_accuracy)
            if evaluation.cash_flow:
                cf_match_rates.append(evaluation.cash_flow.match_rate)
                cf_value_accs.append(evaluation.cash_flow.value_accuracy)
            
            # Track best/worst
            if evaluation.overall_value_accuracy > results.best_value_accuracy:
                results.best_value_accuracy = evaluation.overall_value_accuracy
                results.best_report_id = evaluation.report_id
            if evaluation.overall_value_accuracy < results.worst_value_accuracy:
                results.worst_value_accuracy = evaluation.overall_value_accuracy
                results.worst_report_id = evaluation.report_id
        
        # Calculate averages
        if results.report_evaluations:
            evals = results.report_evaluations
            results.avg_match_rate = sum(e.overall_match_rate for e in evals) / len(evals)
            results.avg_value_accuracy = sum(e.overall_value_accuracy for e in evals) / len(evals)
            results.avg_mape = sum(e.overall_mape for e in evals) / len(evals)
        
        if bs_match_rates:
            results.bs_avg_match_rate = sum(bs_match_rates) / len(bs_match_rates)
            results.bs_avg_value_accuracy = sum(bs_value_accs) / len(bs_value_accs)
        if pl_match_rates:
            results.pl_avg_match_rate = sum(pl_match_rates) / len(pl_match_rates)
            results.pl_avg_value_accuracy = sum(pl_value_accs) / len(pl_value_accs)
        if cf_match_rates:
            results.cf_avg_match_rate = sum(cf_match_rates) / len(cf_match_rates)
            results.cf_avg_value_accuracy = sum(cf_value_accs) / len(cf_value_accs)
        
        logger.info(f"Evaluation complete: {results.successful_reports}/{results.total_reports} reports")
        return results
    
    def save_results(self, results: AggregateResults, filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{timestamp}.json"
        
        output_path = Path(self.config.output_dir) / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def print_summary(self, results: AggregateResults):
        """Print aggregate summary."""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        print(f"\nReports Evaluated: {results.successful_reports}/{results.total_reports}")
        print(f"Failed: {results.failed_reports}")
        
        print("\nOverall Averages:")
        print(f"  Match Rate:      {results.avg_match_rate:.1%}")
        print(f"  Value Accuracy:  {results.avg_value_accuracy:.1%}")
        print(f"  MAPE:            {results.avg_mape:.2f}%")
        
        print("\nPer-Section Averages:")
        print(f"  Balance Sheet:   Match {results.bs_avg_match_rate:.1%} | Accuracy {results.bs_avg_value_accuracy:.1%}")
        print(f"  Income Stmt:     Match {results.pl_avg_match_rate:.1%} | Accuracy {results.pl_avg_value_accuracy:.1%}")
        print(f"  Cash Flow:       Match {results.cf_avg_match_rate:.1%} | Accuracy {results.cf_avg_value_accuracy:.1%}")
        
        print(f"\nBest Report:  {results.best_report_id} ({results.best_value_accuracy:.1%})")
        print(f"Worst Report: {results.worst_report_id} ({results.worst_value_accuracy:.1%})")
        print("=" * 70 + "\n")


def quick_evaluate(
    vnstock_dir: str,
    ocr_dir: str,
    output_dir: str = "evaluation_output",
    tolerance: float = 0.05,
    similarity_threshold: float = 0.75
) -> AggregateResults:
    """
    Quick evaluation function for simple usage.
    """
    config = EvaluationConfig(
        vnstock_dir=vnstock_dir,
        ocr_dir=ocr_dir,
        output_dir=output_dir,
        tolerance=tolerance,
        similarity_threshold=similarity_threshold
    )
    
    evaluator = OCRPipelineEvaluator(config)
    results = evaluator.run()
    evaluator.save_results(results)
    evaluator.print_summary(results)
    
    return results
