"""
Evaluation Module

OCR benchmarking tools for Vietnamese financial reports using the VnPDF dataset.

Dataset:
    VnPdfDataset - HuggingFace dataset loader for kiethuynhanh/vnpdf-financial-reports-dataset

Primary Metrics (Reported):
    - Format-Agnostic CER: CER after stripping formatting
    - Content Word Recall: Bag-of-words recall  
    - Number F1: Precision/Recall/F1 for digit sequences

Benchmark:
    PageLevelBenchmark - Page-by-page OCR evaluation with mean ± std
"""

__all__ = []

try:
    from .ocr_benchmark import (
        VnPdfDataset,
        VnPdfSample,
        calculate_format_agnostic_cer,
        calculate_content_word_recall,
        calculate_number_precision_recall_f1,
        calculate_all_metrics,
        PageLevelBenchmark,
        PageLevelBenchmarkResult,
        CompanyResult,
        PageResult,
        ErrorAnalyzer,
        ErrorAnalysisResult,
    )

    __all__.extend(
        [
            "VnPdfDataset",
            "VnPdfSample",
            "calculate_format_agnostic_cer",
            "calculate_content_word_recall",
            "calculate_number_precision_recall_f1",
            "calculate_all_metrics",
            "PageLevelBenchmark",
            "PageLevelBenchmarkResult",
            "CompanyResult",
            "PageResult",
            "ErrorAnalyzer",
            "ErrorAnalysisResult",
        ]
    )
except ImportError as e:
    import sys

    print(f"Warning: could not import evaluation.ocr_benchmark: {e}", file=sys.stderr)

try:
    from .benchmark_v2 import (
        BenchmarkDatasetV2,
        TableSample,
        csv_to_canonical,
        canonical_to_csv,
        compute_pilot_metrics,
        RawMetricResult,
        calculate_raw_metrics,
        StructuredMetricResult,
        calculate_structured_metrics,
        generate_predictions,
        run_benchmark,
    )

    __all__.extend(
        [
            "BenchmarkDatasetV2",
            "TableSample",
            "csv_to_canonical",
            "canonical_to_csv",
            "compute_pilot_metrics",
            "RawMetricResult",
            "calculate_raw_metrics",
            "StructuredMetricResult",
            "calculate_structured_metrics",
            "generate_predictions",
            "run_benchmark",
        ]
    )
except ImportError as e:
    import sys

    print(f"Warning: could not import evaluation.benchmark_v2: {e}", file=sys.stderr)
