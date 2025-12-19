import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

from logger import get_logger
from services.llm_factory import get_model_info, test_model_structured_output
from services.llm_utils import (
    LLMItemMatcher, 
    LLMUnitDetector,
    MatchResult,
    UnitDetectionResult
)

logger = get_logger(__name__)


class BenchmarkTask(str, Enum):
    """Benchmark task types."""
    ITEM_MATCHING = "item_matching"
    UNIT_DETECTION = "unit_detection"
    ALL = "all"


@dataclass
class TaskResult:
    """Result of a single task execution."""
    success: bool
    latency_ms: float
    error: Optional[str] = None
    output: Optional[Any] = None
    

@dataclass
class ModelTaskBenchmark:
    """Benchmark results for one model on one task."""
    model: str
    provider: str
    task: str
    
    # Aggregated metrics
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    
    # Latency stats (ms)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    
    # Task-specific accuracy metrics
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Errors encountered
    errors: List[str] = field(default_factory=list)
    
    # Individual run results
    run_results: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "task": self.task,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": self.successful_runs / self.total_runs if self.total_runs > 0 else 0,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2) if self.min_latency_ms != float('inf') else None,
            "max_latency_ms": round(self.max_latency_ms, 2),
            "accuracy_metrics": {k: round(v, 4) for k, v in self.accuracy_metrics.items()},
            "errors": self.errors[:5],
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report across all models and tasks."""
    timestamp: str
    tasks_benchmarked: List[str]
    models_tested: List[str]
    
    # Per-task results
    results: Dict[str, List[ModelTaskBenchmark]] = field(default_factory=dict)
    
    # Leaderboard (best model per task)
    leaderboard: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "tasks_benchmarked": self.tasks_benchmarked,
            "models_tested": self.models_tested,
            "results": {
                task: [r.to_dict() for r in results]
                for task, results in self.results.items()
            },
            "leaderboard": self.leaderboard
        }


class ModelBenchmark:
    """
    Main benchmark runner for testing LLM models on different tasks.
    """
    
    def __init__(
        self,
        models: List[str],
        ground_truth_dir: str = "data/ground_truth",
        evaluation_results_dir: str = "evaluation_results",
        output_dir: str = "benchmark_results"
    ):
        """
        Initialize benchmark runner.
        """
        self.models = models
        self.ground_truth_dir = Path(ground_truth_dir)
        self.evaluation_results_dir = Path(evaluation_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test data loaded on demand
        self._test_documents: Optional[List[Dict]] = None
        self._test_items: Optional[List[Dict]] = None
    
    def run(
        self, 
        tasks: List[BenchmarkTask] = None,
        validate_structured_output: bool = True
    ) -> BenchmarkReport:
        """
        Run benchmarks for specified tasks.
        """
        if tasks is None:
            tasks = [BenchmarkTask.ITEM_MATCHING, BenchmarkTask.UNIT_DETECTION]
        elif BenchmarkTask.ALL in tasks:
            tasks = [BenchmarkTask.ITEM_MATCHING, BenchmarkTask.UNIT_DETECTION]
        
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            tasks_benchmarked=[t.value for t in tasks],
            models_tested=self.models.copy(),
            results={},
            leaderboard={}
        )
        
        # Validate models if requested
        valid_models = self.models
        if validate_structured_output:
            valid_models = self._validate_models(tasks)
            report.models_tested = valid_models
        
        if not valid_models:
            logger.error("No valid models to benchmark!")
            return report
        
        # Run benchmarks per task
        for task in tasks:
            logger.info(f"=== Benchmarking task: {task.value} ===")
            task_results = []
            
            for model in valid_models:
                logger.info(f"Testing model: {model}")
                try:
                    result = self._benchmark_task(model, task)
                    task_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to benchmark {model} on {task.value}: {e}")
                    # Create failed result
                    model_info = get_model_info(model)
                    result = ModelTaskBenchmark(
                        model=model,
                        provider=model_info["provider"],
                        task=task.value,
                        total_runs=0,
                        errors=[str(e)]
                    )
                    task_results.append(result)
            
            report.results[task.value] = task_results
            
            # Determine leaderboard winner for this task
            if task_results:
                # Winner based on success rate, then latency
                valid_results = [r for r in task_results if r.successful_runs > 0]
                if valid_results:
                    # Sort by success rate (desc), then avg latency (asc)
                    winner = max(valid_results, key=lambda r: (
                        r.successful_runs / r.total_runs if r.total_runs > 0 else 0,
                        -r.avg_latency_ms
                    ))
                    report.leaderboard[task.value] = winner.model
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _validate_models(self, tasks: List[BenchmarkTask]) -> List[str]:
        """Validate that models support structured output for required schemas."""
        valid_models = []
        
        # Determine which schemas to test based on tasks
        schemas_to_test = []
        if BenchmarkTask.ITEM_MATCHING in tasks:
            schemas_to_test.append(("ItemMatching", MatchResult, "Compare: 'Tiền mặt' vs 'Cash'"))
        if BenchmarkTask.UNIT_DETECTION in tasks:
            schemas_to_test.append(("UnitDetection", UnitDetectionResult, "Find unit in: Đơn vị: triệu VND"))
        
        for model in self.models:
            logger.info(f"Validating structured output for model: {model}")
            model_valid = True
            
            for schema_name, schema, test_prompt in schemas_to_test:
                success, error = test_model_structured_output(model, schema, test_prompt)
                if not success:
                    logger.warning(f"  ✗ {model} failed {schema_name}: {error}")
                    model_valid = False
                    break
                else:
                    logger.info(f"  ✓ {model} passed {schema_name}")
            
            if model_valid:
                valid_models.append(model)
        
        logger.info(f"Valid models: {len(valid_models)}/{len(self.models)}")
        return valid_models
    
    def _benchmark_task(self, model: str, task: BenchmarkTask) -> ModelTaskBenchmark:
        """Run benchmark for one model on one task."""
        model_info = get_model_info(model)
        
        result = ModelTaskBenchmark(
            model=model,
            provider=model_info["provider"],
            task=task.value
        )
        
        if task == BenchmarkTask.ITEM_MATCHING:
            return self._benchmark_item_matching(model, result)
        elif task == BenchmarkTask.UNIT_DETECTION:
            return self._benchmark_unit_detection(model, result)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _benchmark_item_matching(self, model: str, result: ModelTaskBenchmark) -> ModelTaskBenchmark:
        """Benchmark item matching task."""
        test_cases = self._load_matching_test_cases()
        
        matcher = LLMItemMatcher(model=model)
        latencies = []
        correct_matches = 0
        total_matches = 0
        
        for test_case in test_cases:
            ocr_name = test_case["ocr_name"]
            gt_name = test_case["gt_name"]
            expected_match = test_case["should_match"]
            section = test_case.get("section", "BS")
            
            result.total_runs += 1
            
            try:
                start = time.perf_counter()
                match_result = matcher._compare_items(ocr_name, gt_name, section)
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                latencies.append(elapsed_ms)
                total_matches += 1
                
                # Check if match decision is correct
                if match_result.is_match == expected_match:
                    correct_matches += 1
                
                result.successful_runs += 1
                result.run_results.append({
                    "ocr_name": ocr_name,
                    "gt_name": gt_name,
                    "expected": expected_match,
                    "predicted": match_result.is_match,
                    "confidence": match_result.confidence,
                    "correct": match_result.is_match == expected_match,
                    "latency_ms": elapsed_ms
                })
                
            except Exception as e:
                result.failed_runs += 1
                result.errors.append(f"{ocr_name} vs {gt_name}: {str(e)}")
        
        # Calculate aggregate metrics
        if latencies:
            result.avg_latency_ms = sum(latencies) / len(latencies)
            result.min_latency_ms = min(latencies)
            result.max_latency_ms = max(latencies)
        
        if total_matches > 0:
            result.accuracy_metrics["matching_accuracy"] = correct_matches / total_matches
        
        return result
    
    def _benchmark_unit_detection(self, model: str, result: ModelTaskBenchmark) -> ModelTaskBenchmark:
        """Benchmark unit detection task."""
        test_cases = self._load_unit_detection_test_cases()
        
        detector = LLMUnitDetector(model=model)
        latencies = []
        correct_detections = 0
        
        for test_case in test_cases:
            doc_id = test_case["id"]
            content = test_case["content"]
            expected_unit = test_case["expected_unit"]
            
            result.total_runs += 1
            
            try:
                start = time.perf_counter()
                detected_unit = detector.detect_unit(content)
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                latencies.append(elapsed_ms)
                
                # Normalize and compare units
                is_correct = self._normalize_unit(detected_unit) == self._normalize_unit(expected_unit)
                if is_correct:
                    correct_detections += 1
                
                result.successful_runs += 1
                result.run_results.append({
                    "doc_id": doc_id,
                    "expected_unit": expected_unit,
                    "detected_unit": detected_unit,
                    "correct": is_correct,
                    "latency_ms": elapsed_ms
                })
                
            except Exception as e:
                result.failed_runs += 1
                result.errors.append(f"{doc_id}: {str(e)}")
        
        # Calculate aggregate metrics
        if latencies:
            result.avg_latency_ms = sum(latencies) / len(latencies)
            result.min_latency_ms = min(latencies)
            result.max_latency_ms = max(latencies)
        
        if result.successful_runs > 0:
            result.accuracy_metrics["unit_detection_accuracy"] = correct_detections / result.successful_runs
        
        return result
    
    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit string for comparison."""
        unit = unit.lower().strip()
        # Normalize variations
        unit = unit.replace("đồng", "vnd").replace("đ", "vnd")
        unit = unit.replace("tỷ", "ty").replace("triệu", "trieu").replace("nghìn", "nghin")
        return unit
    
    def _load_test_documents(self) -> List[Dict]:
        """Load test documents for table extraction benchmark."""
        if self._test_documents is not None:
            return self._test_documents
        
        documents = []
        
        # Load from evaluation_results directory
        for report_dir in self.evaluation_results_dir.iterdir():
            if not report_dir.is_dir():
                continue
                
            ocr_file = report_dir / "ocr_output.md"
            if ocr_file.exists():
                content = ocr_file.read_text(encoding="utf-8")
                documents.append({
                    "id": report_dir.name,
                    "content": content,
                    "expected_sections": ["BS", "PL", "CF"]
                })
        
        self._test_documents = documents
        logger.info(f"Loaded {len(documents)} test documents for table extraction")
        return documents
    
    def _load_matching_test_cases(self) -> List[Dict]:
        """Load test cases for item matching benchmark."""
        # Standard test cases for Vietnamese financial item matching
        test_cases = [
            # Should match (synonyms)
            {"ocr_name": "Tiền và tương đương tiền", "gt_name": "Tiền mặt và các khoản tương đương tiền", "should_match": True, "section": "BS"},
            {"ocr_name": "TSCĐ hữu hình", "gt_name": "Tài sản cố định hữu hình", "should_match": True, "section": "BS"},
            {"ocr_name": "Vốn chủ sở hữu", "gt_name": "VCSH", "should_match": True, "section": "BS"},
            {"ocr_name": "Doanh thu thuần", "gt_name": "Doanh thu thuần về bán hàng và cung cấp dịch vụ", "should_match": True, "section": "PL"},
            {"ocr_name": "Lợi nhuận gộp", "gt_name": "Lợi nhuận gộp về bán hàng và cung cấp dịch vụ", "should_match": True, "section": "PL"},
            {"ocr_name": "Lưu chuyển tiền từ HĐKD", "gt_name": "Lưu chuyển tiền thuần từ hoạt động kinh doanh", "should_match": True, "section": "CF"},
            
            # Should match (OCR errors)
            {"ocr_name": "Phái thu ngắn hạn", "gt_name": "Phải thu ngắn hạn", "should_match": True, "section": "BS"},
            {"ocr_name": "Tài sàn ngắn hạn", "gt_name": "Tài sản ngắn hạn", "should_match": True, "section": "BS"},
            
            # Should NOT match (semantic opposites)
            {"ocr_name": "Phải thu", "gt_name": "Phải trả", "should_match": False, "section": "BS"},
            {"ocr_name": "Tài sản ngắn hạn", "gt_name": "Tài sản dài hạn", "should_match": False, "section": "BS"},
            {"ocr_name": "Tiền thu từ bán hàng", "gt_name": "Tiền chi trả cho người cung cấp", "should_match": False, "section": "CF"},
            {"ocr_name": "Doanh thu", "gt_name": "Chi phí", "should_match": False, "section": "PL"},
            
            # Should NOT match (parent vs child)
            {"ocr_name": "Tài sản ngắn hạn", "gt_name": "Tiền mặt", "should_match": False, "section": "BS"},
            {"ocr_name": "Nợ phải trả", "gt_name": "Nợ ngắn hạn", "should_match": False, "section": "BS"},
        ]
        
        return test_cases
    
    def _load_unit_detection_test_cases(self) -> List[Dict]:
        """Load test cases for unit detection benchmark."""
        test_cases = []
        
        # Load from evaluation_results with known units from metadata
        for report_dir in self.evaluation_results_dir.iterdir():
            if not report_dir.is_dir():
                continue
            
            ocr_file = report_dir / "ocr_output.md"
            if not ocr_file.exists():
                continue
            
            content = ocr_file.read_text(encoding="utf-8")[:5000]
            
            # Try to determine expected unit from content or metadata
            expected_unit = "VND"
            if "triệu" in content.lower() or "triệu vnd" in content.lower():
                expected_unit = "triệu VND"
            elif "tỷ" in content.lower() or "tỷ vnd" in content.lower():
                expected_unit = "tỷ VND"
            elif "nghìn" in content.lower():
                expected_unit = "nghìn VND"
            
            test_cases.append({
                "id": report_dir.name,
                "content": content,
                "expected_unit": expected_unit
            })
        
        # Add synthetic test cases with known units
        synthetic_cases = [
            {"id": "synthetic_vnd", "content": "Đơn vị tính: VND\n\nBÁO CÁO TÀI CHÍNH", "expected_unit": "VND"},
            {"id": "synthetic_trieu", "content": "Đơn vị: triệu đồng\n\nBẢNG CÂN ĐỐI KẾ TOÁN", "expected_unit": "triệu VND"},
            {"id": "synthetic_ty", "content": "ĐVT: Tỷ VND\n\nBáo cáo kết quả hoạt động", "expected_unit": "tỷ VND"},
            {"id": "synthetic_nghin", "content": "Đơn vị tính: nghìn VND\n\nCash Flow Statement", "expected_unit": "nghìn VND"},
        ]
        test_cases.extend(synthetic_cases)
        
        logger.info(f"Loaded {len(test_cases)} test cases for unit detection")
        return test_cases
    
    def _save_report(self, report: BenchmarkReport) -> Path:
        """Save benchmark report to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Benchmark report saved to: {filepath}")
        
        # Also save a latest.json for easy access
        latest_path = self.output_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        return filepath


def print_benchmark_summary(report: BenchmarkReport):
    """Print a formatted summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Models tested: {len(report.models_tested)}")
    print(f"Tasks benchmarked: {', '.join(report.tasks_benchmarked)}")
    
    for task, results in report.results.items():
        print(f"\n--- {task.upper()} ---")
        print(f"{'Model':<50} {'Success%':<10} {'Avg(ms)':<10} {'Accuracy':<10}")
        print("-" * 80)
        
        for r in sorted(results, key=lambda x: (-x.successful_runs/max(x.total_runs, 1), x.avg_latency_ms)):
            success_rate = r.successful_runs / r.total_runs * 100 if r.total_runs > 0 else 0
            accuracy = list(r.accuracy_metrics.values())[0] * 100 if r.accuracy_metrics else 0
            print(f"{r.model:<50} {success_rate:>6.1f}%   {r.avg_latency_ms:>8.1f}   {accuracy:>6.1f}%")
    
    print("\n--- LEADERBOARD ---")
    for task, winner in report.leaderboard.items():
        print(f"  {task}: {winner}")
    
    print("=" * 60)
