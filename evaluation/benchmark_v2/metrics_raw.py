"""
Raw OCR metrics for benchmark v2.

Focuses on table-markdown/text fidelity and numeric token correctness.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer

from evaluation.ocr_benchmark.metrics import extract_numeric_tokens


@dataclass
class RawMetricResult:
    format_agnostic_cer: float
    format_agnostic_wer: float
    table_cell_f1: float
    number_f1: float
    number_precision: float
    number_recall: float
    reference_cell_count: int
    hypothesis_cell_count: int

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


def _normalize_text_for_edit_metrics(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"[^a-z0-9\u00C0-\u024F\u1E00-\u1EFF]+", " ", s)
    return " ".join(s.split())


def _parse_markdown_pipe_cells(markdown_text: str) -> List[str]:
    cells: List[str] = []
    for line in (markdown_text or "").splitlines():
        s = line.strip()
        if s.count("|") < 2:
            continue
        parts = [p.strip() for p in s.split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        if not parts:
            continue
        if all((set(p.replace(":", "").strip()) <= {"-"} and "-" in p) or p == "" for p in parts):
            continue
        cells.extend(parts)
    return cells


def _multiset_f1(hyp: Iterable[str], ref: Iterable[str]) -> Tuple[float, float, float, int, int, int]:
    hyp_c = Counter(hyp)
    ref_c = Counter(ref)
    matched = sum(min(hyp_c[k], ref_c[k]) for k in hyp_c.keys() & ref_c.keys())
    hyp_total = sum(hyp_c.values())
    ref_total = sum(ref_c.values())

    precision = matched / hyp_total if hyp_total else (1.0 if ref_total == 0 else 0.0)
    recall = matched / ref_total if ref_total else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall, matched, hyp_total, ref_total


def _safe_cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    try:
        return float(min(1.0, jiwer_cer(ref, hyp)))
    except Exception:
        return 1.0


def _safe_wer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    try:
        return float(min(1.0, jiwer_wer(ref, hyp)))
    except Exception:
        return 1.0


def calculate_raw_metrics(hypothesis_markdown: str, reference_markdown: str) -> RawMetricResult:
    ref_clean = _normalize_text_for_edit_metrics(reference_markdown)
    hyp_clean = _normalize_text_for_edit_metrics(hypothesis_markdown)

    ref_cells = [c.strip().lower() for c in _parse_markdown_pipe_cells(reference_markdown) if c.strip()]
    hyp_cells = [c.strip().lower() for c in _parse_markdown_pipe_cells(hypothesis_markdown) if c.strip()]
    table_f1, _, _, _, hyp_cell_count, ref_cell_count = _multiset_f1(hyp_cells, ref_cells)

    ref_nums = extract_numeric_tokens(reference_markdown)
    hyp_nums = extract_numeric_tokens(hypothesis_markdown)
    num_f1, num_p, num_r, _, _, _ = _multiset_f1(hyp_nums, ref_nums)

    return RawMetricResult(
        format_agnostic_cer=_safe_cer(ref_clean, hyp_clean),
        format_agnostic_wer=_safe_wer(ref_clean, hyp_clean),
        table_cell_f1=float(table_f1),
        number_f1=float(num_f1),
        number_precision=float(num_p),
        number_recall=float(num_r),
        reference_cell_count=int(ref_cell_count),
        hypothesis_cell_count=int(hyp_cell_count),
    )

