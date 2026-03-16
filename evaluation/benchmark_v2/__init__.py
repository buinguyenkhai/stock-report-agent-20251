"""
Benchmark v2 for Vietnamese financial table OCR.

This module evaluates raw OCR table fidelity only.
"""

__all__ = []

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
    from .run import run_benchmark

    __all__.append("run_benchmark")
except ImportError:
    pass

try:
    from .predict import generate_predictions

    __all__.append("generate_predictions")
except ImportError:
    pass
