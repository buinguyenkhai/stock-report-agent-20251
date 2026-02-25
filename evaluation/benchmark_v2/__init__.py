"""
Benchmark v2 for Vietnamese financial table OCR/extraction.

This module is intentionally decoupled from the legacy HF-based benchmark.
It evaluates:
1) Raw OCR table fidelity (markdown/text level)
2) End-to-end structured output fidelity (UI-visible table rows/values)
"""

__all__ = [
    # dynamically extended below
]

try:
    from .dataset import BenchmarkDatasetV2, TableSample

    __all__.extend(["BenchmarkDatasetV2", "TableSample"])
except ImportError:
    pass

try:
    from .metrics_raw import RawMetricResult, calculate_raw_metrics

    __all__.extend(["RawMetricResult", "calculate_raw_metrics"])
except ImportError:
    pass

try:
    from .metrics_structured import StructuredMetricResult, calculate_structured_metrics

    __all__.extend(["StructuredMetricResult", "calculate_structured_metrics"])
except ImportError:
    pass

try:
    from .run import run_benchmark

    __all__.append("run_benchmark")
except ImportError:
    pass

try:
    from .predict import generate_predictions

    __all__.append("generate_predictions")
except ImportError:
    pass

try:
    from .csv_codec import csv_to_canonical, canonical_to_csv, compute_pilot_metrics

    __all__.extend(["csv_to_canonical", "canonical_to_csv", "compute_pilot_metrics"])
except ImportError:
    pass
