"""
Pipeline Benchmark

Tests LLM models on actual pipeline tasks:
1. Extraction: Can the model correctly extract financial tables from OCR markdown?
2. Parsing: Can the model correctly parse and normalize financial data?

Uses real ground truth data for rigorous evaluation.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from logger import get_logger
from services.extractors import (
    BalanceSheetExtractor,
    IncomeStatementExtractor,
    CashFlowExtractor,
)
from services.parser import AggregatedParser, ExtractionBundle, ParsedReport

logger = get_logger(__name__)


class BenchmarkTask(str, Enum):
    """Pipeline benchmark tasks."""
    EXTRACTION = "extraction"
    PARSING = "parsing"
    FULL_PIPELINE = "full_pipeline"
    ALL = "all"


@dataclass
class ExtractionTestCase:
    """Test case for extraction benchmark."""
    report_id: str
    ocr_content: str
    expected_bs_items: int  # Minimum expected items
    expected_pl_items: int
    expected_cf_items: int
    # Key items that MUST be present
    required_bs_items: List[str] = field(default_factory=list)
    required_pl_items: List[str] = field(default_factory=list)
    required_cf_items: List[str] = field(default_factory=list)


@dataclass
class ParsingTestCase:
    """Test case for parsing benchmark."""
    report_id: str
    extracted_content: Dict[str, str]  # {bs, pl, cf}
    ground_truth: Dict[str, Dict[str, float]]  # {statement: {item_name: value}}
    unit: str = "VND"


@dataclass
class TaskResult:
    """Result of a single test case."""
    test_id: str
    success: bool
    latency_ms: float
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelBenchmarkResult:
    """Benchmark results for one model on one task."""
    model: str
    task: str
    
    # Counts
    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    
    # Latency (ms)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # Accuracy metrics (task-specific)
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Per-case results
    case_results: List[TaskResult] = field(default_factory=list)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    def add_result(self, result: TaskResult):
        """Add a test case result."""
        self.case_results.append(result)
        self.total_cases += 1
        
        if result.success:
            self.successful_cases += 1
            self.total_latency_ms += result.latency_ms
            self.min_latency_ms = min(self.min_latency_ms, result.latency_ms)
            self.max_latency_ms = max(self.max_latency_ms, result.latency_ms)
        else:
            self.failed_cases += 1
            if result.error:
                self.errors.append(f"{result.test_id}: {result.error}")
    
    def finalize(self):
        """Calculate final metrics."""
        if self.successful_cases > 0:
            self.avg_latency_ms = self.total_latency_ms / self.successful_cases
        if self.min_latency_ms == float('inf'):
            self.min_latency_ms = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.successful_cases / self.total_cases
    
    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "task": self.task,
            "total_cases": self.total_cases,
            "successful_cases": self.successful_cases,
            "failed_cases": self.failed_cases,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "accuracy_metrics": {k: round(v, 4) for k, v in self.accuracy_metrics.items()},
            "errors": self.errors[:5],
            "case_details": [
                {
                    "test_id": r.test_id,
                    "success": r.success,
                    "latency_ms": round(r.latency_ms, 2),
                    "error": r.error,
                    "metrics": r.metrics,
                }
                for r in self.case_results
            ],
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    models_tested: List[str]
    tasks_benchmarked: List[str]
    results: Dict[str, List[ModelBenchmarkResult]] = field(default_factory=dict)
    leaderboard: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "models_tested": self.models_tested,
            "tasks_benchmarked": self.tasks_benchmarked,
            "results": {
                task: [r.to_dict() for r in results]
                for task, results in self.results.items()
            },
            "leaderboard": self.leaderboard,
        }


class PipelineBenchmark:
    """
    Benchmark runner for pipeline LLM tasks.
    
    Tests models on:
    1. Extraction accuracy - Can the model find and extract correct tables?
    2. Parsing accuracy - Can the model produce correct structured output?
    3. Full pipeline - End-to-end extraction + parsing
    """
    
    def __init__(
        self,
        models: List[str],
        ground_truth_dir: str = "data/ground_truth",
        ocr_cache_dir: str = "evaluation_results_pipeline",
        pdf_dir: str = "data/pdfs",
        output_dir: str = "benchmark_results",
    ):
        self.models = models
        self.ground_truth_dir = Path(ground_truth_dir)
        self.ocr_cache_dir = Path(ocr_cache_dir)
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache
        self._extraction_cases: Optional[List[ExtractionTestCase]] = None
        self._parsing_cases: Optional[List[ParsingTestCase]] = None
        self._ocr_service = None
    
    def _get_ocr_service(self):
        """Lazy load OCR service."""
        if self._ocr_service is None:
            from services.ocr import get_ocr_service
            self._ocr_service = get_ocr_service("marker")
        return self._ocr_service
    
    def prepare_test_data(self) -> int:
        """
        Prepare OCR test data from PDFs.
        Returns number of test cases prepared.
        """
        count = 0
        
        if not self.pdf_dir.exists():
            logger.warning(f"PDF directory not found: {self.pdf_dir}")
            return 0
        
        for pdf_file in self.pdf_dir.glob("*.pdf"):
            report_id = pdf_file.stem  # e.g., "DBC_2022_Q1"
            
            # Check if ground truth exists
            gt_dir = self.ground_truth_dir / report_id
            if not gt_dir.exists():
                logger.warning(f"No ground truth for {report_id}, skipping")
                continue
            
            # Check if OCR already done
            cache_dir = self.ocr_cache_dir / report_id
            ocr_file = cache_dir / "ocr_output.md"
            
            if ocr_file.exists():
                logger.info(f"OCR already exists for {report_id}")
                count += 1
                continue
            
            # Run OCR
            logger.info(f"Running OCR for {report_id}...")
            try:
                ocr_service = self._get_ocr_service()
                content = ocr_service.process_pdf(str(pdf_file))
                
                cache_dir.mkdir(parents=True, exist_ok=True)
                ocr_file.write_text(content, encoding="utf-8")
                logger.info(f"OCR complete: {len(content):,} chars")
                count += 1
            except Exception as e:
                logger.error(f"OCR failed for {report_id}: {e}")
        
        return count
    
    def run(
        self,
        tasks: List[BenchmarkTask] = None,
    ) -> BenchmarkReport:
        """Run benchmark for specified tasks."""
        if tasks is None:
            tasks = [BenchmarkTask.EXTRACTION, BenchmarkTask.PARSING]
        elif BenchmarkTask.ALL in tasks:
            tasks = [BenchmarkTask.EXTRACTION, BenchmarkTask.PARSING, BenchmarkTask.FULL_PIPELINE]
        
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            models_tested=self.models.copy(),
            tasks_benchmarked=[t.value for t in tasks],
        )
        
        for task in tasks:
            logger.info(f"=== Benchmarking: {task.value} ===")
            task_results = []
            
            for model in self.models:
                logger.info(f"Testing model: {model}")
                try:
                    if task == BenchmarkTask.EXTRACTION:
                        result = self._benchmark_extraction(model)
                    elif task == BenchmarkTask.PARSING:
                        result = self._benchmark_parsing(model)
                    elif task == BenchmarkTask.FULL_PIPELINE:
                        result = self._benchmark_full_pipeline(model)
                    else:
                        raise ValueError(f"Unknown task: {task}")
                    
                    task_results.append(result)
                except Exception as e:
                    logger.error(f"Benchmark failed for {model}: {e}")
                    result = ModelBenchmarkResult(model=model, task=task.value)
                    result.errors.append(str(e))
                    task_results.append(result)
            
            report.results[task.value] = task_results
            
            # Determine leaderboard
            if task_results:
                self._update_leaderboard(report, task.value, task_results)
        
        self._save_report(report)
        return report
    
    # Benchmark
    
    def _benchmark_extraction(self, model: str) -> ModelBenchmarkResult:
        """Benchmark extraction task."""
        result = ModelBenchmarkResult(model=model, task="extraction")
        test_cases = self._load_extraction_cases()
        
        if not test_cases:
            result.errors.append("No test cases available")
            return result
        
        # Create extractors with this model
        bs_extractor = BalanceSheetExtractor(model=model)
        pl_extractor = IncomeStatementExtractor(model=model)
        cf_extractor = CashFlowExtractor(model=model)
        
        total_items_found = 0
        total_items_expected = 0
        required_items_found = 0
        required_items_total = 0
        
        for case in test_cases:
            try:
                start = time.perf_counter()
                
                # Run extraction
                bs_result = bs_extractor.extract(case.ocr_content)
                pl_result = pl_extractor.extract(case.ocr_content)
                cf_result = cf_extractor.extract(case.ocr_content)
                
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                # Save extracted content for parsing benchmark to use
                report_dir = self.ocr_cache_dir / case.report_id
                if bs_result and bs_result.content:
                    (report_dir / "extracted_bs.md").write_text(bs_result.content, encoding="utf-8")
                if pl_result and pl_result.content:
                    (report_dir / "extracted_pl.md").write_text(pl_result.content, encoding="utf-8")
                if cf_result and cf_result.content:
                    (report_dir / "extracted_cf.md").write_text(cf_result.content, encoding="utf-8")
                
                # Check required items presence
                found_required = 0
                total_required = len(case.required_bs_items) + len(case.required_pl_items) + len(case.required_cf_items)
                
                for item in case.required_bs_items:
                    if bs_result and item.lower() in bs_result.content.lower():
                        found_required += 1
                for item in case.required_pl_items:
                    if pl_result and item.lower() in pl_result.content.lower():
                        found_required += 1
                for item in case.required_cf_items:
                    if cf_result and item.lower() in cf_result.content.lower():
                        found_required += 1
                
                # Accuracy is based on required items found
                accuracy = found_required / total_required if total_required > 0 else 0
                
                total_items_found += found_required
                total_items_expected += total_required
                required_items_found += found_required
                required_items_total += total_required
                
                # Success if we found at least 70% of required items
                success = accuracy >= 0.7
                
                task_result = TaskResult(
                    test_id=case.report_id,
                    success=success,
                    latency_ms=elapsed_ms,
                    metrics={
                        "required_found": found_required,
                        "required_total": total_required,
                        "accuracy": accuracy,
                    }
                )
                result.add_result(task_result)
                
            except Exception as e:
                result.add_result(TaskResult(
                    test_id=case.report_id,
                    success=False,
                    latency_ms=0,
                    error=str(e),
                ))
        
        # Calculate accuracy metrics (now based on required items)
        result.finalize()
        if total_items_expected > 0:
            result.accuracy_metrics["required_item_accuracy"] = total_items_found / total_items_expected
        
        return result
    
    # Parsing Benchmark     
    def _benchmark_parsing(self, model: str) -> ModelBenchmarkResult:
        """Benchmark parsing task."""
        result = ModelBenchmarkResult(model=model, task="parsing")
        test_cases = self._load_parsing_cases()
        
        if not test_cases:
            result.errors.append("No test cases available")
            return result
        
        parser = AggregatedParser(model=model)
        
        total_matched = 0
        total_gt_items = 0
        total_value_accurate = 0
        
        for case in test_cases:
            try:
                # Build extraction bundle
                bundle = ExtractionBundle(
                    balance_sheet=case.extracted_content.get("bs", ""),
                    income_statement=case.extracted_content.get("pl", ""),
                    cash_flow=case.extracted_content.get("cf", ""),
                    metadata={"unit": case.unit},
                )
                
                start = time.perf_counter()
                parsed: ParsedReport = parser.parse(bundle)
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                # Evaluate against ground truth
                metrics = self._evaluate_parsing(parsed, case.ground_truth)
                
                total_matched += metrics["matched_items"]
                total_gt_items += metrics["total_gt_items"]
                total_value_accurate += metrics["value_accurate_items"]
                
                success = metrics["match_rate"] >= 0.3  # At least 30% match rate
                
                task_result = TaskResult(
                    test_id=case.report_id,
                    success=success,
                    latency_ms=elapsed_ms,
                    metrics=metrics,
                )
                result.add_result(task_result)
                
            except Exception as e:
                result.add_result(TaskResult(
                    test_id=case.report_id,
                    success=False,
                    latency_ms=0,
                    error=str(e),
                ))
        
        # Calculate aggregate metrics
        result.finalize()
        if total_gt_items > 0:
            result.accuracy_metrics["match_rate"] = total_matched / total_gt_items
        if total_matched > 0:
            result.accuracy_metrics["value_accuracy"] = total_value_accurate / total_matched
        
        return result
    
    # Full Pipeline Benchmark
    
    def _benchmark_full_pipeline(self, model: str) -> ModelBenchmarkResult:
        """Benchmark full extraction + parsing pipeline."""
        result = ModelBenchmarkResult(model=model, task="full_pipeline")
        
        # Load test cases with both OCR and ground truth
        extraction_cases = self._load_extraction_cases()
        
        if not extraction_cases:
            result.errors.append("No test cases available")
            return result
        
        # Create components
        bs_extractor = BalanceSheetExtractor(model=model)
        pl_extractor = IncomeStatementExtractor(model=model)
        cf_extractor = CashFlowExtractor(model=model)
        parser = AggregatedParser(model=model)
        
        total_matched = 0
        total_gt_items = 0
        
        for case in extraction_cases:
            try:
                # Load ground truth for this report
                gt = self._load_ground_truth(case.report_id)
                if not gt:
                    continue
                
                start = time.perf_counter()
                
                # Extraction
                bs_result = bs_extractor.extract(case.ocr_content)
                pl_result = pl_extractor.extract(case.ocr_content)
                cf_result = cf_extractor.extract(case.ocr_content)
                
                # Parsing
                bundle = ExtractionBundle(
                    balance_sheet=bs_result.content if bs_result else "",
                    income_statement=pl_result.content if pl_result else "",
                    cash_flow=cf_result.content if cf_result else "",
                )
                parsed = parser.parse(bundle)
                
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                # Evaluate
                metrics = self._evaluate_parsing(parsed, gt)
                total_matched += metrics["matched_items"]
                total_gt_items += metrics["total_gt_items"]
                
                success = metrics["match_rate"] >= 0.25
                
                result.add_result(TaskResult(
                    test_id=case.report_id,
                    success=success,
                    latency_ms=elapsed_ms,
                    metrics=metrics,
                ))
                
            except Exception as e:
                result.add_result(TaskResult(
                    test_id=case.report_id,
                    success=False,
                    latency_ms=0,
                    error=str(e),
                ))
        
        result.finalize()
        if total_gt_items > 0:
            result.accuracy_metrics["end_to_end_match_rate"] = total_matched / total_gt_items
        
        return result
    
    # Test Case Loading
    
    def _load_extraction_cases(self) -> List[ExtractionTestCase]:
        """Load extraction test cases from cached OCR outputs."""
        if self._extraction_cases is not None:
            return self._extraction_cases
        
        cases = []
        
        # Check OCR cache directory
        for report_dir in self.ocr_cache_dir.iterdir():
            if not report_dir.is_dir():
                continue
            
            ocr_file = report_dir / "ocr_output.md"
            if not ocr_file.exists():
                continue
            
            content = ocr_file.read_text(encoding="utf-8")
            report_id = report_dir.name
            
            # Load ground truth to get expected counts
            gt_dir = self.ground_truth_dir / report_id
            if not gt_dir.exists():
                continue
            
            bs_count = self._count_csv_rows(gt_dir / "balance_sheet.csv")
            pl_count = self._count_csv_rows(gt_dir / "income_statement.csv")
            cf_count = self._count_csv_rows(gt_dir / "cash_flow.csv")
            
            # Detect if this is a bank (different accounting terminology)
            is_bank = self._detect_bank_report(content)
            
            # Required items that MUST be found (different for banks vs companies)
            if is_bank:
                # Banks use different balance sheet structure
                required_bs = [
                    "TÀI SẢN CÓ",  # Assets (bank terminology)
                    "NỢ PHẢI TRẢ",
                    "VỐN CHỦ SỞ HỮU",
                ]
                required_pl = [
                    "Thu nhập lãi",  # Interest income (bank)
                    "Lợi nhuận",
                ]
            else:
                # Standard company balance sheet
                required_bs = [
                    "TÀI SẢN NGẮN HẠN",
                    "TÀI SẢN DÀI HẠN", 
                    "TỔNG CỘNG TÀI SẢN",
                    "NỢ PHẢI TRẢ",
                    "VỐN CHỦ SỞ HỮU",
                ]
                required_pl = [
                    "Doanh thu",
                    "Lợi nhuận",
                ]
            
            required_cf = [
                "Lưu chuyển tiền",
            ]
            
            cases.append(ExtractionTestCase(
                report_id=report_id,
                ocr_content=content,
                expected_bs_items=min(bs_count, 35),
                expected_pl_items=min(pl_count, 26),
                expected_cf_items=min(cf_count, 35),
                required_bs_items=required_bs,
                required_pl_items=required_pl,
                required_cf_items=required_cf,
            ))
        
        self._extraction_cases = cases
        logger.info(f"Loaded {len(cases)} extraction test cases")
        return cases
    
    def _load_parsing_cases(self) -> List[ParsingTestCase]:
        """Load parsing test cases with pre-extracted content."""
        if self._parsing_cases is not None:
            return self._parsing_cases
        
        cases = []
        
        # Use cached pipeline results if available
        for report_dir in self.ocr_cache_dir.iterdir():
            if not report_dir.is_dir():
                continue
            
            report_id = report_dir.name
            
            # Check for extracted content
            bs_file = report_dir / "extracted_bs.md"
            pl_file = report_dir / "extracted_pl.md"
            cf_file = report_dir / "extracted_cf.md"
            
            # If no extracted files exist, run extraction first
            if not bs_file.exists():
                ocr_file = report_dir / "ocr_output.md"
                if not ocr_file.exists():
                    continue
                
                # Run extraction to create the files
                logger.info(f"Running extraction for {report_id} (no cached extraction files)")
                ocr_content = ocr_file.read_text(encoding="utf-8")
                
                from services.extractors import (
                    BalanceSheetExtractor, 
                    IncomeStatementExtractor, 
                    CashFlowExtractor
                )
                
                bs_extractor = BalanceSheetExtractor()
                pl_extractor = IncomeStatementExtractor()
                cf_extractor = CashFlowExtractor()
                
                try:
                    bs_result = bs_extractor.extract(ocr_content)
                    pl_result = pl_extractor.extract(ocr_content)
                    cf_result = cf_extractor.extract(ocr_content)
                    
                    # Save for future use
                    if bs_result and bs_result.content:
                        bs_file.write_text(bs_result.content, encoding="utf-8")
                    if pl_result and pl_result.content:
                        pl_file.write_text(pl_result.content, encoding="utf-8")
                    if cf_result and cf_result.content:
                        cf_file.write_text(cf_result.content, encoding="utf-8")
                        
                    extracted = {
                        "bs": bs_result.content if bs_result else "",
                        "pl": pl_result.content if pl_result else "",
                        "cf": cf_result.content if cf_result else "",
                    }
                except Exception as e:
                    logger.error(f"Extraction failed for {report_id}: {e}")
                    continue
            else:
                extracted = {
                    "bs": bs_file.read_text(encoding="utf-8") if bs_file.exists() else "",
                    "pl": pl_file.read_text(encoding="utf-8") if pl_file.exists() else "",
                    "cf": cf_file.read_text(encoding="utf-8") if cf_file.exists() else "",
                }
            
            # Load ground truth
            gt = self._load_ground_truth(report_id)
            if not gt:
                continue
            
            cases.append(ParsingTestCase(
                report_id=report_id,
                extracted_content=extracted,
                ground_truth=gt,
                unit="VND",
            ))
        
        self._parsing_cases = cases
        logger.info(f"Loaded {len(cases)} parsing test cases")
        return cases
    
    def _detect_bank_report(self, content: str) -> bool:
        """Detect if content is from a bank financial report."""
        content_lower = content.lower()
        
        # Strong bank-specific indicators (these only appear in bank reports)
        strong_indicators = [
            "tổng tài sản có",  # Total assets (bank-only terminology)
            "cho vay khách hàng",  # Customer loans (bank core business)
            "tiền gửi khách hàng",  # Customer deposits (bank core business)
            "tổ chức tín dụng",  # Credit institution
        ]
        
        # If any strong indicator is present, it's a bank
        for indicator in strong_indicators:
            if indicator in content_lower:
                return True
        
        return False
    
    def _load_ground_truth(self, report_id: str) -> Optional[Dict[str, Dict[str, float]]]:
        """Load ground truth for a report."""
        gt_dir = self.ground_truth_dir / report_id
        if not gt_dir.exists():
            return None
        
        result = {"bs": {}, "pl": {}, "cf": {}}
        
        # Balance sheet
        bs_file = gt_dir / "balance_sheet.csv"
        if bs_file.exists():
            result["bs"] = self._parse_ground_truth_csv(bs_file, report_id)
        
        # Income statement
        pl_file = gt_dir / "income_statement.csv"
        if pl_file.exists():
            result["pl"] = self._parse_ground_truth_csv(pl_file, report_id)
        
        # Cash flow
        cf_file = gt_dir / "cash_flow.csv"
        if cf_file.exists():
            result["cf"] = self._parse_ground_truth_csv(cf_file, report_id)
        
        return result
    
    def _parse_ground_truth_csv(self, filepath: Path, report_id: str) -> Dict[str, float]:
        """Parse ground truth CSV to {item_name: value} dict."""
        try:
            df = pd.read_csv(filepath)
            
            # Extract year and quarter from report_id (e.g., "DBC_2022_Q1")
            parts = report_id.split("_")
            if len(parts) >= 3:
                target_year = int(parts[1])
                target_quarter = int(parts[2].replace("Q", ""))
            else:
                return {}
            
            # Filter to matching row
            df_filtered = df[(df["Năm"] == target_year) & (df["Kỳ"] == target_quarter)]
            if df_filtered.empty:
                return {}
            
            row = df_filtered.iloc[0]
            
            # Get all numeric columns (item values)
            result = {}
            for col in df.columns:
                if col in ["CP", "Năm", "Kỳ"]:
                    continue
                try:
                    value = float(row[col])
                    # Clean column name
                    item_name = col.replace(" (đồng)", "").replace("(đồng)", "").strip()
                    result[item_name] = value
                except (ValueError, TypeError):
                    continue
            
            return result
        except Exception as e:
            logger.warning(f"Failed to parse {filepath}: {e}")
            return {}
    
    # Evaluation Helpers
    
    def _evaluate_parsing(
        self, 
        parsed: ParsedReport, 
        ground_truth: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Evaluate parsed output against ground truth."""
        matched = 0
        value_accurate = 0
        total_gt = 0
        
        # Safety check - handle malformed parsed output
        def get_items_safely(statement) -> Dict[str, float]:
            """Safely extract items from a ParsedStatement."""
            if statement is None:
                return {}
            items = getattr(statement, 'items', None)
            if items is None:
                return {}
            try:
                return {item.name: item.value for item in items if item is not None}
            except (TypeError, AttributeError):
                return {}
        
        # Evaluate each statement
        for stmt_key, gt_items in ground_truth.items():
            if stmt_key == "bs":
                parsed_items = get_items_safely(parsed.balance_sheet)
            elif stmt_key == "pl":
                parsed_items = get_items_safely(parsed.income_statement)
            elif stmt_key == "cf":
                parsed_items = get_items_safely(parsed.cash_flow)
            else:
                continue
            
            total_gt += len(gt_items)
            
            for gt_name, gt_value in gt_items.items():
                # Try to find matching parsed item
                match_name = self._find_matching_item(gt_name, parsed_items.keys())
                if match_name:
                    matched += 1
                    
                    # Check value accuracy with multiple tolerance levels
                    parsed_value = parsed_items.get(match_name)
                    if parsed_value is not None and gt_value != 0:
                        error_pct = abs(parsed_value - gt_value) / abs(gt_value)
                        # Use 5% tolerance (vnstock API data may differ from PDF)
                        if error_pct < 0.05:
                            value_accurate += 1
        
        return {
            "matched_items": matched,
            "total_gt_items": total_gt,
            "value_accurate_items": value_accurate,
            "match_rate": matched / total_gt if total_gt > 0 else 0,
            "value_accuracy": value_accurate / matched if matched > 0 else 0,
        }
    
    def _find_matching_item(self, gt_name: str, parsed_names: List[str]) -> Optional[str]:
        """Find matching item name using fuzzy matching."""
        gt_lower = gt_name.lower().strip()
        
        # Exact match
        for name in parsed_names:
            if name.lower().strip() == gt_lower:
                return name
        
        # Partial match (one contains the other)
        for name in parsed_names:
            name_lower = name.lower().strip()
            if gt_lower in name_lower or name_lower in gt_lower:
                return name
        
        # Jaccard similarity > 0.5
        gt_words = set(gt_lower.split())
        for name in parsed_names:
            name_words = set(name.lower().split())
            intersection = gt_words & name_words
            union = gt_words | name_words
            if union and len(intersection) / len(union) > 0.5:
                return name
        
        return None
    
    def _estimate_table_rows(self, content: str) -> int:
        """Estimate number of table rows in markdown content."""
        if not content:
            return 0
        # Count lines with pipe characters (markdown table format)
        lines = [line for line in content.split("\n") if "|" in line and not line.startswith("|---")]
        return max(0, len(lines) - 1)  # Subtract header row
    
    def _count_csv_rows(self, filepath: Path) -> int:
        """Count data rows in a CSV file."""
        if not filepath.exists():
            return 0
        try:
            df = pd.read_csv(filepath)
            return len(df.columns) - 3  # Subtract CP, Năm, Kỳ columns
        except:
            return 0
    
    # Leaderboard
    
    def _update_leaderboard(
        self, 
        report: BenchmarkReport, 
        task: str, 
        results: List[ModelBenchmarkResult]
    ):
        """Update leaderboard for a task."""
        if not results:
            return
        
        # Find best by success rate, then by primary accuracy metric
        valid_results = [r for r in results if r.successful_cases > 0]
        if not valid_results:
            return
        
        # Sort by success rate, then by accuracy
        def score(r: ModelBenchmarkResult) -> Tuple[float, float, float]:
            acc = list(r.accuracy_metrics.values())[0] if r.accuracy_metrics else 0
            return (r.success_rate, acc, -r.avg_latency_ms)
        
        best = max(valid_results, key=score)
        
        report.leaderboard[task] = {
            "model": best.model,
            "success_rate": round(best.success_rate, 4),
            "avg_latency_ms": round(best.avg_latency_ms, 2),
            "accuracy": {k: round(v, 4) for k, v in best.accuracy_metrics.items()},
        }
    
    # Report Saving
    
    def _save_report(self, report: BenchmarkReport) -> Path:
        """Save benchmark report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"pipeline_benchmark_{timestamp}.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Save latest
        latest = self.output_dir / "pipeline_benchmark_latest.json"
        with open(latest, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Report saved to {filepath}")
        return filepath


def print_benchmark_summary(report: BenchmarkReport):
    """Print formatted benchmark summary."""
    print("\n" + "=" * 70)
    print("PIPELINE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Timestamp: {report.timestamp}")
    print(f"Models: {len(report.models_tested)}")
    print(f"Tasks: {', '.join(report.tasks_benchmarked)}")
    
    for task, results in report.results.items():
        print(f"\n{'─' * 70}")
        print(f"{task.upper()}")
        print(f"{'─' * 70}")
        print(f"{'Model':<45} {'Success':<10} {'Latency':<12} {'Accuracy':<12}")
        print("-" * 70)
        
        for r in sorted(results, key=lambda x: (-x.success_rate, x.avg_latency_ms)):
            acc_str = ""
            if r.accuracy_metrics:
                acc_val = list(r.accuracy_metrics.values())[0] * 100
                acc_str = f"{acc_val:.1f}%"
            
            print(f"{r.model:<45} {r.success_rate*100:>5.1f}%    {r.avg_latency_ms:>8.0f}ms   {acc_str:>10}")
    
    print(f"\n{'=' * 70}")
    print("LEADERBOARD")
    print("=" * 70)
    for task, winner in report.leaderboard.items():
        if isinstance(winner, dict):
            acc_str = ""
            if winner.get("accuracy"):
                acc_val = list(winner["accuracy"].values())[0] * 100
                acc_str = f", acc={acc_val:.1f}%"
            print(f"  {task}: {winner['model']} (success={winner['success_rate']*100:.1f}%{acc_str})")
        else:
            print(f"  {task}: {winner}")
    print("=" * 70)
