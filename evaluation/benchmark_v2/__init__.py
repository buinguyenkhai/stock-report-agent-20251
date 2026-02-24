"""
Benchmark v2 for Vietnamese financial table OCR/extraction.

This module is intentionally decoupled from the legacy HF-based benchmark.
It evaluates:
1) Raw OCR table fidelity (markdown/text level)
2) End-to-end structured output fidelity (UI-visible table rows/values)
"""

from .dataset import BenchmarkDatasetV2, TableSample
from .metrics_raw import RawMetricResult, calculate_raw_metrics
from .metrics_structured import StructuredMetricResult, calculate_structured_metrics
from .predict import generate_predictions
from .run import run_benchmark

__all__ = [
    "BenchmarkDatasetV2",
    "TableSample",
    "RawMetricResult",
    "calculate_raw_metrics",
    "StructuredMetricResult",
    "calculate_structured_metrics",
    "generate_predictions",
    "run_benchmark",
]
