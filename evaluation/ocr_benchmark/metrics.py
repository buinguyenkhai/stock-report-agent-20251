"""
OCR Evaluation Metrics for Financial Documents

Primary metrics (3 metrics used in benchmark):
1. Format-Agnostic CER: CER after stripping formatting (fair comparison)
2. Content Word Recall: Bag-of-words recall (content completeness)
3. Number F1: Precision/Recall/F1 for digit sequences (financial accuracy)
"""

import re
from dataclasses import dataclass
from typing import Set, Dict, Any, Optional
from jiwer import cer as jiwer_cer


@dataclass
class MetricResult:
    """Result from a metric calculation."""
    value: float  # Primary metric value (0-1 scale typically)
    details: Optional[Dict[str, Any]] = None

def extract_digit_sequences(text: str) -> Set[str]:
    """
    Extract all digit sequences from text.
    """
    return set(re.findall(r'\d+', text))


def calculate_format_agnostic_cer(hypothesis: str, reference: str) -> MetricResult:
    """
    CER after stripping ALL formatting (pipes, dashes, markdown, etc).
    
    Uses same tokenization as Word Recall:
    - Replace all non-alphanumeric with space
    - Collapse whitespace
    - Join tokens back into string for CER calculation
    
    This makes it truly format-agnostic for comparing different table formats.
    """
    def extract_content(text: str) -> str:
        # Replace all non-alphanum (keeping Vietnamese) with space
        normalized = re.sub(r'[^a-zA-Z0-9\u00C0-\u024F\u1E00-\u1EFF]+', ' ', text.lower())
        # Collapse whitespace and return as single string
        return ' '.join(normalized.split())
    
    hyp_clean = extract_content(hypothesis)
    ref_clean = extract_content(reference)
    
    # Calculate CER on cleaned text
    if not ref_clean:
        return MetricResult(value=0.0 if not hyp_clean else 1.0)
    
    try:
        value = jiwer_cer(ref_clean, hyp_clean)
        value = min(value, 1.0)  # Cap at 1.0
        return MetricResult(value=value)
    except Exception:
        return MetricResult(value=1.0)


def calculate_content_word_recall(hypothesis: str, reference: str) -> MetricResult:
    """
    What fraction of reference words appear in hypothesis?
    Simple bag-of-words recall.
    
    Tokenization: replace all non-alphanumeric characters with space, then split.
    This handles any format (markdown tables, HTML, plain text).
    """
    def tokenize(text: str) -> set:
        # Replace all non-alphanum (including Vietnamese) with space, then split
        # Keeps: a-z, A-Z, 0-9, Vietnamese characters (unicode)
        normalized = re.sub(r'[^a-zA-Z0-9\u00C0-\u024F\u1E00-\u1EFF]+', ' ', text.lower())
        return set(normalized.split())
    
    ref_words = tokenize(reference)
    hyp_words = tokenize(hypothesis)
    
    if not ref_words:
        return MetricResult(value=1.0)
    
    matched = ref_words & hyp_words
    recall = len(matched) / len(ref_words)
    
    return MetricResult(
        value=recall,
        details={"matched": len(matched), "total": len(ref_words)}
    )


def calculate_number_precision_recall_f1(hypothesis: str, reference: str) -> MetricResult:
    """
    Precision, Recall, F1 for digit sequences.
    
    - Precision = matched / ocr_total (how many OCR numbers are correct)
    - Recall = matched / gt_total (how many GT numbers were found)
    - F1 = harmonic mean
    """
    gt_nums = extract_digit_sequences(reference)
    ocr_nums = extract_digit_sequences(hypothesis)
    
    if not gt_nums and not ocr_nums:
        return MetricResult(
            value=1.0,
            details={"precision": 1.0, "recall": 1.0, "f1": 1.0}
        )
    
    matched = gt_nums & ocr_nums
    
    precision = len(matched) / len(ocr_nums) if ocr_nums else 0.0
    recall = len(matched) / len(gt_nums) if gt_nums else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return MetricResult(
        value=f1,
        details={
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched": len(matched),
            "gt_count": len(gt_nums),
            "ocr_count": len(ocr_nums),
        }
    )


def calculate_all_metrics(hypothesis: str, reference: str) -> Dict[str, MetricResult]:
    """
    Calculate primary OCR metrics for financial documents.
    
    Returns dict with:
    - format_agnostic_cer: CER after stripping formatting (lower is better)
    - content_word_recall: Word recall (higher is better)
    - number_f1: Digit sequence F1 (higher is better)
    """
    num_f1_result = calculate_number_precision_recall_f1(hypothesis, reference)
    
    # Add details for debugging
    if num_f1_result.details:
        num_f1_result.details["reference_numbers"] = list(extract_digit_sequences(reference))
        num_f1_result.details["hypothesis_numbers"] = list(extract_digit_sequences(hypothesis))
    
    return {
        "format_agnostic_cer": calculate_format_agnostic_cer(hypothesis, reference),
        "content_word_recall": calculate_content_word_recall(hypothesis, reference),
        "number_f1": num_f1_result,
    }

