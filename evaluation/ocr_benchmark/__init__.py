"""
OCR Benchmark Module

Evaluates OCR engines on Vietnamese financial reports using the VnPDF dataset.
"""

__all__ = []

from .metrics import (
    calculate_all_metrics,
    calculate_format_agnostic_cer,
    calculate_content_word_recall,
    calculate_number_precision_recall_f1,
)

__all__.extend(
    [
        "calculate_all_metrics",
        "calculate_format_agnostic_cer",
        "calculate_content_word_recall",
        "calculate_number_precision_recall_f1",
    ]
)

try:
    from .dataset_loader import VnPdfDataset, VnPdfSample

    __all__.extend(["VnPdfDataset", "VnPdfSample"])
except ImportError:
    pass

try:
    from .page_level_benchmark import (
        PageLevelBenchmark,
        PageResult,
        CompanyResult,
        PageLevelBenchmarkResult,
    )

    __all__.extend(
        [
            "PageLevelBenchmark",
            "PageResult",
            "CompanyResult",
            "PageLevelBenchmarkResult",
        ]
    )
except ImportError:
    pass

try:
    from .error_analyzer import ErrorAnalyzer, ErrorAnalysisResult

    __all__.extend(["ErrorAnalyzer", "ErrorAnalysisResult"])
except ImportError:
    pass
