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
    
    __all__ = [
        # Dataset
        "VnPdfDataset",
        "VnPdfSample",
        # Metrics
        "calculate_format_agnostic_cer",
        "calculate_content_word_recall",
        "calculate_number_precision_recall_f1",
        "calculate_all_metrics",
        # Benchmark
        "PageLevelBenchmark",
        "PageLevelBenchmarkResult",
        "CompanyResult",
        "PageResult",
        # Error Analysis
        "ErrorAnalyzer",
        "ErrorAnalysisResult",
    ]
except ImportError as e:
    import sys
    print(f"Error importing OCR benchmark: {e}", file=sys.stderr)
    raise