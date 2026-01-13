"""
OCR Benchmark Module

Evaluates OCR engines on Vietnamese financial reports using the VnPDF dataset.
"""

from .page_level_benchmark import (
    PageLevelBenchmark,
    PageResult,
    CompanyResult,
    PageLevelBenchmarkResult,
)
from .dataset_loader import VnPdfDataset, VnPdfSample
from .metrics import (
    calculate_all_metrics,
    calculate_format_agnostic_cer,
    calculate_content_word_recall,
    calculate_number_precision_recall_f1,
)
from .error_analyzer import ErrorAnalyzer, ErrorAnalysisResult

__all__ = [
    # Benchmark
    "PageLevelBenchmark",
    "PageResult",
    "CompanyResult",
    "PageLevelBenchmarkResult",
    # Dataset
    "VnPdfDataset",
    "VnPdfSample",
    # Metrics
    "calculate_all_metrics",
    "calculate_format_agnostic_cer",
    "calculate_content_word_recall",
    "calculate_number_precision_recall_f1",
    # Analysis
    "ErrorAnalyzer",
    "ErrorAnalysisResult",
]
