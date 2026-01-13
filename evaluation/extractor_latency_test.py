"""
Extractor Latency Comparison Test

DEPRECATED: This file is currently broken due to dataset changes from vnstock to vnpdf.
Kept as reference for future reimplementation of latency benchmarking.
   
TODO: Reimplement to work with PageLevelBenchmark and VnPdfDataset

Original purpose:
Compares latency of:
1. 3 separate extractors (BalanceSheet + IncomeStatement + CashFlow)
2. 1 combined FinancialTablesExtractor

Run with: python -m evaluation.extractor_latency_test
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
import argparse

from logger import get_logger
from services.extractors import (
    BalanceSheetExtractor,
    IncomeStatementExtractor,
    CashFlowExtractor,
    FinancialTablesExtractor,
)

logger = get_logger(__name__)


@dataclass
class LatencyResult:
    """Result of a latency test."""
    approach: str  # "separate" or "combined"
    report_id: str
    total_latency_ms: float
    success: bool
    bs_found: bool = False
    pl_found: bool = False
    cf_found: bool = False
    error: Optional[str] = None
    
    # For separate approach
    bs_latency_ms: float = 0.0
    pl_latency_ms: float = 0.0
    cf_latency_ms: float = 0.0


@dataclass
class ComparisonSummary:
    """Summary comparing both approaches."""
    report_id: str
    separate_latency_ms: float
    combined_latency_ms: float
    speedup: float  # combined is X times faster
    separate_success: bool
    combined_success: bool


class ExtractorLatencyTest:
    """
    Tests latency of separate vs combined extractors.
    """
    
    def __init__(
        self,
        model: str = "mistralai/devstral-2512:free",
        ocr_cache_dir: str = "evaluation_results_pipeline",
    ):
        self.model = model
        self.ocr_cache_dir = Path(ocr_cache_dir)
        
        # Initialize extractors
        self.bs_extractor = BalanceSheetExtractor(model=model)
        self.pl_extractor = IncomeStatementExtractor(model=model)
        self.cf_extractor = CashFlowExtractor(model=model)
        self.combined_extractor = FinancialTablesExtractor(model=model)
    
    def get_available_reports(self) -> List[str]:
        """Get list of reports with OCR output available."""
        reports = []
        if self.ocr_cache_dir.exists():
            for report_dir in self.ocr_cache_dir.iterdir():
                if report_dir.is_dir():
                    ocr_file = report_dir / "ocr_output.md"
                    if ocr_file.exists():
                        reports.append(report_dir.name)
        return sorted(reports)
    
    def load_ocr_content(self, report_id: str) -> str:
        """Load OCR content for a report."""
        ocr_file = self.ocr_cache_dir / report_id / "ocr_output.md"
        return ocr_file.read_text(encoding="utf-8")
    
    async def test_separate_async(self, report_id: str, markdown: str) -> LatencyResult:
        """Test 3 separate extractors running in parallel."""
        result = LatencyResult(
            approach="separate",
            report_id=report_id,
            total_latency_ms=0,
            success=False,
        )
        
        try:
            start = time.perf_counter()
            
            # Run all 3 in parallel
            bs_task = self.bs_extractor.extract_async(markdown)
            pl_task = self.pl_extractor.extract_async(markdown)
            cf_task = self.cf_extractor.extract_async(markdown)
            
            # Also measure individual latencies
            bs_start = time.perf_counter()
            bs_result = await bs_task
            result.bs_latency_ms = (time.perf_counter() - bs_start) * 1000
            
            pl_start = time.perf_counter()
            pl_result = await pl_task
            result.pl_latency_ms = (time.perf_counter() - pl_start) * 1000
            
            cf_start = time.perf_counter()
            cf_result = await cf_task
            result.cf_latency_ms = (time.perf_counter() - cf_start) * 1000
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.total_latency_ms = elapsed_ms
            
            result.bs_found = bool(bs_result and bs_result.content and bs_result.success)
            result.pl_found = bool(pl_result and pl_result.content and pl_result.success)
            result.cf_found = bool(cf_result and cf_result.content and cf_result.success)
            result.success = result.bs_found or result.pl_found or result.cf_found
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    async def test_separate_parallel_async(self, report_id: str, markdown: str) -> LatencyResult:
        """Test 3 separate extractors running truly in parallel with asyncio.gather."""
        result = LatencyResult(
            approach="separate_parallel",
            report_id=report_id,
            total_latency_ms=0,
            success=False,
        )
        
        try:
            start = time.perf_counter()
            
            # Run all 3 truly in parallel
            results = await asyncio.gather(
                self.bs_extractor.extract_async(markdown),
                self.pl_extractor.extract_async(markdown),
                self.cf_extractor.extract_async(markdown),
                return_exceptions=True
            )
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.total_latency_ms = elapsed_ms
            
            bs_result, pl_result, cf_result = results
            
            if not isinstance(bs_result, Exception):
                result.bs_found = bool(bs_result and bs_result.content and bs_result.success)
            if not isinstance(pl_result, Exception):
                result.pl_found = bool(pl_result and pl_result.content and pl_result.success)
            if not isinstance(cf_result, Exception):
                result.cf_found = bool(cf_result and cf_result.content and cf_result.success)
            
            result.success = result.bs_found or result.pl_found or result.cf_found
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    async def test_combined_async(self, report_id: str, markdown: str) -> LatencyResult:
        """Test 1 combined extractor."""
        result = LatencyResult(
            approach="combined",
            report_id=report_id,
            total_latency_ms=0,
            success=False,
        )
        
        try:
            start = time.perf_counter()
            
            combined_result = await self.combined_extractor.extract_combined_async(markdown)
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.total_latency_ms = elapsed_ms
            
            result.bs_found = bool(combined_result.balance_sheet)
            result.pl_found = bool(combined_result.income_statement)
            result.cf_found = bool(combined_result.cash_flow)
            result.success = combined_result.success
            
            if not combined_result.success:
                result.error = combined_result.error
                
        except Exception as e:
            result.error = str(e)
            
        return result
    
    async def run_comparison_async(
        self,
        report_ids: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, ComparisonSummary]:
        """
        Run comparison between separate and combined approaches.
        """
        if report_ids is None:
            report_ids = self.get_available_reports()
        
        if not report_ids:
            logger.error("No reports available for testing")
            return {}
        
        summaries = {}
        
        for report_id in report_ids:
            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"Testing: {report_id}")
                logger.info(f"{'='*60}")
            
            markdown = self.load_ocr_content(report_id)
            if verbose:
                logger.info(f"OCR content: {len(markdown):,} chars")
            
            # Test separate (parallel)
            if verbose:
                logger.info("\n--- Testing SEPARATE extractors (3 parallel calls) ---")
            separate_result = await self.test_separate_parallel_async(report_id, markdown)
            if verbose:
                logger.info(f"Latency: {separate_result.total_latency_ms:.2f}ms")
                logger.info(f"Found: BS={separate_result.bs_found}, PL={separate_result.pl_found}, CF={separate_result.cf_found}")
            
            # Test combined
            if verbose:
                logger.info("\n--- Testing COMBINED extractor (1 call) ---")
            combined_result = await self.test_combined_async(report_id, markdown)
            if verbose:
                logger.info(f"Latency: {combined_result.total_latency_ms:.2f}ms")
                logger.info(f"Found: BS={combined_result.bs_found}, PL={combined_result.pl_found}, CF={combined_result.cf_found}")
            
            # Calculate speedup
            if separate_result.total_latency_ms > 0:
                speedup = separate_result.total_latency_ms / combined_result.total_latency_ms
            else:
                speedup = 0
            
            summary = ComparisonSummary(
                report_id=report_id,
                separate_latency_ms=separate_result.total_latency_ms,
                combined_latency_ms=combined_result.total_latency_ms,
                speedup=speedup,
                separate_success=separate_result.success,
                combined_success=combined_result.success,
            )
            summaries[report_id] = summary
            
            if verbose:
                logger.info(f"\n>>> Speedup: {speedup:.2f}x {'(combined faster)' if speedup > 1 else '(separate faster)'}")
        
        return summaries
    
    def run_comparison(
        self,
        report_ids: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, ComparisonSummary]:
        """Synchronous wrapper for run_comparison_async."""
        return asyncio.run(self.run_comparison_async(report_ids, verbose))
    
    def print_summary(self, summaries: Dict[str, ComparisonSummary]):
        """Print summary table of results."""
        if not summaries:
            print("No results to display")
            return
        
        print("LATENCY COMPARISON SUMMARY")
        print(f"{'Report ID':<20} {'Separate (ms)':<15} {'Combined (ms)':<15} {'Speedup':<12} {'Status':<15}")
        
        total_separate = 0
        total_combined = 0
        
        for report_id, summary in summaries.items():
            status = "Both OK" if summary.separate_success and summary.combined_success else "Check"
            print(f"{report_id:<20} {summary.separate_latency_ms:<15.2f} {summary.combined_latency_ms:<15.2f} {summary.speedup:<12.2f}x {status:<15}")
            total_separate += summary.separate_latency_ms
            total_combined += summary.combined_latency_ms
        
        avg_separate = total_separate / len(summaries)
        avg_combined = total_combined / len(summaries)
        avg_speedup = avg_separate / avg_combined if avg_combined > 0 else 0
        
        print(f"{'AVERAGE':<20} {avg_separate:<15.2f} {avg_combined:<15.2f} {avg_speedup:<12.2f}x")
        print(f"{'TOTAL':<20} {total_separate:<15.2f} {total_combined:<15.2f}")
        
        if avg_speedup > 1:
            print(f"\nCombined extractor is {avg_speedup:.2f}x FASTER on average")
            savings = total_separate - total_combined
            print(f"Total time saved: {savings:.2f}ms ({savings/1000:.2f}s)")
        else:
            print(f"\nSeparate extractors are {1/avg_speedup:.2f}x FASTER on average")


def main():
    parser = argparse.ArgumentParser(description="Compare extractor latencies")
    parser.add_argument(
        "--model",
        default="mistralai/devstral-2512:free",
        help="Model to use for extraction"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Specific report ID to test (default: all available)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    tester = ExtractorLatencyTest(model=args.model)
    
    # Show available reports
    available = tester.get_available_reports()
    print(f"Available reports: {available}")
    
    if not available:
        print("No reports with OCR output found. Run benchmark to generate OCR first.")
        return
    
    report_ids = [args.report] if args.report else None
    
    summaries = tester.run_comparison(
        report_ids=report_ids,
        verbose=not args.quiet
    )
    
    tester.print_summary(summaries)


if __name__ == "__main__":
    main()
