"""
OCR Evaluation Metrics for Financial Documents

Primary metrics (3 metrics used in benchmark):
1. Format-Agnostic CER: CER after stripping formatting (fair comparison)
2. Content Word Recall: Bag-of-words recall (content completeness)
3. Number F1: Precision/Recall/F1 for digit sequences (financial accuracy)
"""

import re
from collections import Counter
from dataclasses import dataclass

from typing import Counter as CounterType, Dict, Any, Iterable, List, Optional, Set, Tuple

from jiwer import cer as jiwer_cer


@dataclass
class MetricResult:
    """Result from a metric calculation."""
    value: float  # Primary metric value (0-1 scale typically)
    details: Optional[Dict[str, Any]] = None

def extract_digit_sequences(text: str) -> Set[str]:
    """Extract all digit sequences from text (legacy, kept for debugging)."""
    return set(re.findall(r"\d+", text or ""))


_NUMBER_TOKEN_RE = re.compile(
    r"(?ix)"
    r"(?P<sign>-|\()??\s*"
    r"(?P<int>\d{1,3}(?:[.,\s]\d{3})*|\d+)"
    r"(?P<dec>(?:[.,]\d+))?"
    r"\s*(?P<suffix>%|\))?"
)


def _normalize_numeric_token(raw: str) -> Optional[str]:
    """Normalize a numeric token into a canonical string.

    Examples:
      - "(1.234.567)" -> "-1234567"
      - "1,234.56" -> "1234.56"
      - "1.234,56" -> "1234.56"
      - "12,5%" -> "12.5%"
    """
    s = (raw or "").strip()
    if not s:
        return None

    s = s.replace("\u00A0", " ")
    s = s.replace(" ", "")

    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    if s.startswith("-"):
        negative = True
        s = s[1:]

    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1]

    if not re.search(r"\d", s):
        return None

    last_dot = s.rfind(".")
    last_comma = s.rfind(",")

    dec_sep: Optional[str] = None
    if last_dot != -1 and last_comma != -1:
        dec_sep = "." if last_dot > last_comma else ","
    elif last_dot != -1:
        if len(s) - last_dot - 1 in (1, 2):
            dec_sep = "."
        else:
            dec_sep = None
    elif last_comma != -1:
        if len(s) - last_comma - 1 in (1, 2):
            dec_sep = ","
        else:
            dec_sep = None

    if dec_sep is None:
        int_part = re.sub(r"[.,]", "", s)
        if not int_part.isdigit():
            int_part = re.sub(r"\D", "", int_part)
        if not int_part:
            return None
        out = int_part
    else:
        parts = s.split(dec_sep)
        if len(parts) < 2:
            return None
        int_raw = "".join(parts[:-1])
        dec_raw = parts[-1]

        int_part = re.sub(r"[.,]", "", int_raw)
        int_part = re.sub(r"\D", "", int_part)
        dec_part = re.sub(r"\D", "", dec_raw)
        if not int_part:
            int_part = "0"
        if not dec_part:
            out = int_part
        else:
            out = f"{int_part}.{dec_part}"

    if negative and out != "0":
        out = f"-{out}"
    if is_percent:
        out = out + "%"
    return out


def extract_numeric_tokens(text: str) -> List[str]:
    """Extract numeric-like tokens and normalize them."""
    if not text:
        return []

    tokens: List[str] = []
    for m in _NUMBER_TOKEN_RE.finditer(text):
        raw = m.group(0)
        norm = _normalize_numeric_token(raw)
        if norm is None:
            continue
        tokens.append(norm)
    return tokens


def _multiset_f1(hyp: Iterable[str], ref: Iterable[str]) -> Tuple[float, float, float, int, int, int]:
    hyp_c: CounterType[str] = Counter(hyp)
    ref_c: CounterType[str] = Counter(ref)
    matched = sum(min(hyp_c[k], ref_c[k]) for k in hyp_c.keys() & ref_c.keys())
    hyp_total = sum(hyp_c.values())
    ref_total = sum(ref_c.values())

    precision = matched / hyp_total if hyp_total else (1.0 if ref_total == 0 else 0.0)
    recall = matched / ref_total if ref_total else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall, matched, hyp_total, ref_total


def calculate_format_agnostic_cer(hypothesis: str, reference: str) -> MetricResult:
    """
    CER after stripping ALL formatting (pipes, dashes, markdown, etc).
    
    Uses same tokenization as Word Recall:
    - Replace all non-alphanumeric with space
    - Collapse whitespace
    - Join tokens back into string for CER calculation
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


def _word_tokens(text: str) -> List[str]:
    normalized = re.sub(r"[^a-zA-Z0-9\u00C0-\u024F\u1E00-\u1EFF]+", " ", (text or "").lower())
    return [t for t in normalized.split() if t]


def calculate_content_word_recall(hypothesis: str, reference: str) -> MetricResult:
    """Multiset word recall (bag-of-words with multiplicity)."""
    ref_tokens = _word_tokens(reference)
    hyp_tokens = _word_tokens(hypothesis)

    if not ref_tokens:
        return MetricResult(value=1.0)

    _, _, recall, matched, _, ref_total = _multiset_f1(hyp_tokens, ref_tokens)
    return MetricResult(value=recall, details={"matched": matched, "total": ref_total})


def calculate_number_precision_recall_f1(hypothesis: str, reference: str) -> MetricResult:
    """Precision/Recall/F1 for locale-robust numeric tokens (multiset).

    Matching is performed on normalized token strings (multiset), not raw digit
    sequences.
    """
    gt_nums = extract_numeric_tokens(reference)
    ocr_nums = extract_numeric_tokens(hypothesis)

    f1, precision, recall, matched, hyp_total, ref_total = _multiset_f1(ocr_nums, gt_nums)

    return MetricResult(
        value=f1,
        details={
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched": matched,
            "gt_count": ref_total,
            "ocr_count": hyp_total,
        },
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
    
    # Keep legacy digit sequences for debugging comparisons
    if num_f1_result.details is not None:
        num_f1_result.details["reference_digit_sequences"] = list(extract_digit_sequences(reference))
        num_f1_result.details["hypothesis_digit_sequences"] = list(extract_digit_sequences(hypothesis))
    
    return {
        "format_agnostic_cer": calculate_format_agnostic_cer(hypothesis, reference),
        "content_word_recall": calculate_content_word_recall(hypothesis, reference),
        "number_f1": num_f1_result,
    }

