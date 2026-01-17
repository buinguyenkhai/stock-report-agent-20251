"""
Extracts specific pages from PDFs and benchmarks OCR page-by-page
against the HuggingFace ground truth.
"""

import json
import time
import math
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from PIL import Image
import io
import tempfile
import os
import csv
import subprocess

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from logger import get_logger
from .dataset_loader import VnPdfDataset, VnPdfSample
from .metrics import calculate_all_metrics, calculate_content_word_recall, calculate_number_precision_recall_f1

logger = get_logger(__name__)

# PDF samples directory
PDF_SAMPLES_DIR = Path("data/pdf_samples")

# Company code mapping
COMPANY_CODES = ["AAA", "ACB", "FPT", "MBB", "MWG", "SHB", "TCB", "VIB", "VPB"]


def compute_std(values: List[float]) -> float:
    """Compute standard deviation of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)  # Sample std
    return math.sqrt(variance)


def extract_table_content(text: str) -> str:
    """
    Extract only table content from text.
    
    Ground truth only contains table rows (lines starting with '|').
    This ensures fair comparison by extracting only table content from OCR output.
    
    Args:
        text: Raw OCR output text
        
    Returns:
        Filtered text containing only table rows
    """
    lines = text.split('\n')
    table_lines = []
    for line in lines:
        stripped = line.strip()
        # Table rows start with '|' or contain table separators
        if stripped.startswith('|') or '|---|' in stripped:
            table_lines.append(line)
    return '\n'.join(table_lines)


def extract_table_content_robust(text: str) -> str:
    """Extract table-like content from OCR output in a format-robust way.

    Motivation:
    - GT for this benchmark contains table content.
    - Different OCR engines/pipelines may emit tables as:
      (a) Markdown pipe tables
      (b) Wrapped/indented pipe rows
      (c) Plain text with column alignment (multi-space / tabs)

    This extractor is *not* a generic fallback to raw OCR.
    It returns only lines that satisfy a table-likeness criterion.
    If no table-like lines are found, it returns "".
    """
    if not text:
        return ""

    lines = text.splitlines()

    def is_markdown_pipe_row(line: str) -> bool:
        s = line.strip()
        if s.count("|") < 2:
            return False
        # Avoid keeping pure separator lines
        if set(s.replace("|", "").strip()) <= {"-", ":"} and "-" in s:
            return False
        return True

    # 1) Prefer markdown pipe rows if present
    pipe_rows = [ln for ln in lines if is_markdown_pipe_row(ln)]
    if pipe_rows:
        return "\n".join(pipe_rows)

    # 2) Otherwise, keep only column-aligned lines (multi-space or tab separated)
    #    AND that look table-like (at least one number or many columns)
    aligned: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        has_columns = bool(re.search(r"\S(?:\s{2,}|\t)\S", s))
        has_number = bool(re.search(r"\d", s))
        if has_columns and (has_number or len(re.split(r"\s{2,}|\t", s)) >= 3):
            aligned.append(ln)

    return "\n".join(aligned)


def _normalize_markdownish_rows(text: str) -> List[str]:
    """Best-effort row splitting for pipe-table strings.

    The HF GT in this project is sometimes a *single line* where rows are separated
    by a space followed by a new row starting with '|', e.g. "...| |---|...| |A ...|".
    This normalizer inserts newlines before those row starts.
    """
    if not text:
        return []

    s = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" not in s and "|" in s:
        # Split rows on occurrences of " |" that start a new markdown row.
        s = re.sub(r"\s\|", "\n|", s)
    return s.splitlines()


def parse_pipe_table_to_grid(text: str) -> List[List[str]]:
    """Parse a markdown-ish pipe table into a 2D grid of cell strings.

    This is intentionally permissive and only relies on the '|' delimiter.
    It discards markdown separator rows like: |---|---| or |:---|---:|.
    """
    rows: List[List[str]] = []
    for ln in _normalize_markdownish_rows(text):
        s = ln.strip()
        if s.count("|") < 2:
            continue
        # Split, then remove leading/trailing empties from pipe edges.
        parts = [p.strip() for p in s.split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        if not parts:
            continue

        # Skip markdown separator rows
        if all((set(p.replace(":", "").strip()) <= {"-"} and "-" in p) or p == "" for p in parts):
            continue

        rows.append(parts)

    return rows


def grid_to_canonical_text(grid: List[List[str]]) -> str:
    """Canonicalize a 2D grid into a text representation for metrics.

    We join cells with tabs and rows with newlines. This is robust for
    the existing metrics (format-agnostic CER, word recall, number F1).
    """
    out_lines: List[str] = []
    for row in grid:
        if not row:
            continue
        cleaned = [c.strip() for c in row]
        if not any(cleaned):
            continue
        out_lines.append("\t".join(cleaned))
    return "\n".join(out_lines)


def _tesseract_tsv_words(
    img: Image.Image,
    lang: str = "vie",
) -> List[Dict[str, Any]]:
    """Run Tesseract TSV on an image and return word-level boxes.

    Returns a list of dicts with keys: text, conf, left, top, width, height.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        img.save(tmp_path)
        # Use TSV for word boxes. Keep defaults; downstream logic is tolerant.
        cmd = [
            "tesseract",
            "-l",
            lang,
            tmp_path,
            "stdout",
            "tsv",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode != 0:
            logger.warning(f"Tesseract TSV failed (rc={proc.returncode}): {proc.stderr[:2000]}")
            return []

        rows: List[Dict[str, Any]] = []
        reader = csv.DictReader(io.StringIO(proc.stdout), delimiter="\t")
        for row in reader:
            try:
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                if int(row.get("word_num") or 0) <= 0:
                    continue
                conf = float(row.get("conf") or -1)
                if conf < 0:
                    conf = 0.0
                rows.append(
                    {
                        "text": text,
                        "conf": conf / 100.0,
                        "left": int(float(row.get("left") or 0)),
                        "top": int(float(row.get("top") or 0)),
                        "width": int(float(row.get("width") or 0)),
                        "height": int(float(row.get("height") or 0)),
                    }
                )
            except Exception:
                continue
        return rows
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _tesseract_text_for_crop(
    img: Image.Image,
    lang: str = "vie",
    psm: int = 7,
    whitelist: Optional[str] = None,
) -> str:
    """Run Tesseract on a crop and return plain text (single cell)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        img.save(tmp_path)
        cmd = [
            "tesseract",
            "-l",
            lang,
            "--psm",
            str(psm),
        ]
        if whitelist:
            cmd += ["-c", f"tessedit_char_whitelist={whitelist}"]
        cmd += [tmp_path, "stdout"]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode != 0:
            return ""
        return (proc.stdout or "").strip()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


_NUM_ALLOWED = set("0123456789.,()%+-/ ")


def _numeric_candidate_score(text: str) -> tuple[int, int, int]:
    """Heuristic scoring for numeric OCR candidates.

    Higher is better. Tuple order: (digit_count, -illegal_char_count, -paren_penalty)
    """
    if not text:
        return (0, -9999, -9999)
    s = text.strip()
    digit_count = sum(ch.isdigit() for ch in s)
    illegal = sum((ch not in _NUM_ALLOWED) for ch in s)
    paren_penalty = 0
    if s.count("(") != s.count(")"):
        paren_penalty += 5
    # Penalize obvious OCR junk
    paren_penalty += sum(ch in {"|", "\\", "_"} for ch in s)
    return (digit_count, -illegal, -paren_penalty)


def _text_candidate_score(text: str) -> tuple[int, int]:
    """Score general-text candidates: prefer longer, less noisy."""
    if not text:
        return (0, -9999)
    s = text.strip()
    length = len(s)
    noise = sum(ch in {"|", "\\", "_"} for ch in s)
    return (length, -noise)


def _pick_best_candidate(candidates: List[str], prefer_numeric: bool) -> str:
    return _pick_best_candidate_with_baseline(candidates, prefer_numeric=prefer_numeric)


def _pick_best_candidate_with_baseline(
    candidates: List[str],
    *,
    prefer_numeric: bool,
    baseline: Optional[str] = None,
) -> str:
    """Pick a candidate conservatively.

    We strongly bias toward the provided baseline (typically the Docling cell text)
    to avoid regressions from noisy alternate sources (e.g., Marker).
    """
    if not candidates:
        return ""

    uniq: List[str] = []
    seen = set()
    for c in candidates:
        c2 = (c or "").strip()
        if not c2:
            continue
        if c2 not in seen:
            seen.add(c2)
            uniq.append(c2)

    if not uniq:
        return ""

    base = (baseline or "").strip() if baseline is not None else ""
    if not base:
        base = uniq[0]

    if prefer_numeric:
        base_score = _numeric_candidate_score(base)

        # If baseline looks clean numeric already, don't override.
        if base_score[0] > 0 and base_score[1] == 0 and base_score[2] == 0:
            return base

        best = max(uniq, key=_numeric_candidate_score)
        best_score = _numeric_candidate_score(best)
        return best if best_score > base_score else base

    base_score_t = _text_candidate_score(base)
    best = max(uniq, key=_text_candidate_score)
    best_score_t = _text_candidate_score(best)

    # Only override baseline for text if we get a meaningful, non-noisier gain.
    if best != base:
        if best_score_t[0] >= base_score_t[0] + 8 and best_score_t[1] >= base_score_t[1]:
            return best
        return base
    return base


def _extract_pipe_table_blocks(text: str) -> List[List[List[str]]]:
    """Extract multiple pipe-table blocks from markdown-like text."""
    if not text:
        return []
    blocks: List[str] = []
    cur: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        is_pipey = s.count("|") >= 2
        if is_pipey:
            cur.append(ln)
        else:
            if cur:
                blocks.append("\n".join(cur))
                cur = []
    if cur:
        blocks.append("\n".join(cur))

    grids: List[List[List[str]]] = []
    for b in blocks:
        g = parse_pipe_table_to_grid(b)
        if g:
            grids.append(g)
    return grids


def _layout_table_from_words(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Strategy 1: derive a table-like grid (rows x cols) from Tesseract word boxes.

    This is intended as a *fallback* when Docling extracts no tables, but the page
    still contains a table-like aligned structure (including borderless 2-col lists).

    Returns a grid of cell dicts: {text, bbox:{l,t,r,b,coord_origin}}.
    """
    if not words:
        return []

    # Build rough text lines by y.
    words_sorted = sorted(words, key=lambda w: (w["top"], w["left"]))
    heights = [max(1, int(w["height"])) for w in words_sorted]
    median_h = sorted(heights)[len(heights) // 2]
    y_tol = max(8, int(median_h * 0.7))

    lines: List[List[Dict[str, Any]]] = []
    for w in words_sorted:
        if not lines:
            lines.append([w])
            continue
        last = lines[-1]
        last_y = int(sum(x["top"] for x in last) / len(last))
        if abs(w["top"] - last_y) <= y_tol:
            last.append(w)
        else:
            lines.append([w])

    # Determine column "stops" from left positions.
    lefts = sorted(w["left"] for w in words_sorted)
    if not lefts:
        return []

    # Greedy binning of left positions to reduce noise.
    bins: List[int] = []
    bin_tol = max(12, int(median_h * 0.6))
    for x in lefts:
        if not bins or abs(x - bins[-1]) > bin_tol:
            bins.append(x)

    # Identify significant gaps between bins as potential column breaks.
    gaps = [(bins[i + 1] - bins[i], i) for i in range(len(bins) - 1)]
    gaps_sorted = sorted(gaps, reverse=True)

    # Choose up to 5 breakpoints, but keep max cols modest.
    break_idxs = []
    for gap, idx in gaps_sorted[:8]:
        if gap >= max(60, int(median_h * 3.5)):
            break_idxs.append(idx)
    break_idxs = sorted(set(break_idxs))

    col_starts = [bins[0]]
    for idx in break_idxs:
        col_starts.append(bins[idx + 1])
    col_starts = sorted(col_starts)

    # Cap to avoid pathological over-splitting.
    if len(col_starts) > 6:
        col_starts = col_starts[:6]
    if len(col_starts) < 2:
        # Not enough evidence of multiple columns from gaps.
        # Fallback: attempt a simple 2-column split (common for key/value layouts).
        min_x = min(lefts)
        max_x = max(lefts)
        span = max_x - min_x
        if span < 180:
            return []

        left_thresh = min_x + 0.45 * span
        right_thresh = min_x + 0.60 * span
        left_group = [x for x in lefts if x <= left_thresh]
        right_group = [x for x in lefts if x >= right_thresh]
        if len(left_group) < 10 or len(right_group) < 10:
            return []

        # Use the 10th percentile of the right group as the right column start.
        right_group_sorted = sorted(right_group)
        right_start = right_group_sorted[max(0, int(len(right_group_sorted) * 0.10))]
        col_starts = sorted({int(min_x), int(right_start)})

    # Assign words into cells by nearest column start.
    grid: List[List[Dict[str, Any]]] = []
    for line in lines:
        # For each word, assign to closest column start by x distance.
        buckets: List[List[Dict[str, Any]]] = [[] for _ in col_starts]
        for w in sorted(line, key=lambda ww: ww["left"]):
            distances = [abs(w["left"] - cs) for cs in col_starts]
            col_idx = distances.index(min(distances))
            buckets[col_idx].append(w)

        row_cells: List[Dict[str, Any]] = []
        for col_words in buckets:
            if not col_words:
                row_cells.append(
                    {
                        "text": "",
                        "bbox": {"l": 0, "t": 0, "r": 0, "b": 0, "coord_origin": "TOPLEFT"},
                    }
                )
                continue
            col_words = sorted(col_words, key=lambda ww: ww["left"])
            text = " ".join(w["text"] for w in col_words).strip()
            l = min(w["left"] for w in col_words)
            t = min(w["top"] for w in col_words)
            r = max(w["left"] + w["width"] for w in col_words)
            b = max(w["top"] + w["height"] for w in col_words)
            row_cells.append(
                {
                    "text": text,
                    "bbox": {"l": float(l), "t": float(t), "r": float(r), "b": float(b), "coord_origin": "TOPLEFT"},
                }
            )
        if any(c["text"].strip() for c in row_cells):
            grid.append(row_cells)
    return grid


def extract_docling_tables_grid(doc_dict: Dict[str, Any]) -> List[List[str]]:
    """Extract table content from Docling's export_to_dict structure.

    Docling export schema (observed):
      doc['tables'][i]['data']['grid'] is a 2D list of dicts, each with a 'text' key.
    """
    tables = doc_dict.get("tables")
    if not isinstance(tables, list) or not tables:
        return []

    def _cell_text(cell: Any) -> str:
        # Observed variants:
        # - cell is a dict with 'text'
        # - cell is a list of dicts (spans / merged cells)
        if isinstance(cell, dict):
            return str(cell.get("text") or "")
        if isinstance(cell, list):
            parts = []
            for x in cell:
                if isinstance(x, dict) and (x.get("text") or "").strip():
                    parts.append(str(x.get("text") or "").strip())
            return " ".join(parts)
        return ""

    merged: List[List[str]] = []
    for t in tables:
        if not isinstance(t, dict):
            continue
        data = t.get("data")
        if not isinstance(data, dict):
            continue
        grid = data.get("grid")
        if not isinstance(grid, list) or not grid:
            continue
        for row in grid:
            if not isinstance(row, list):
                continue
            row_texts = [_cell_text(cell) for cell in row]
            if any(c.strip() for c in row_texts):
                merged.append(row_texts)
    return merged

def _coerce_cell_to_bbox(cell_obj: Any) -> Optional[Dict[str, Any]]:
    """Return a representative bbox dict for a Docling grid cell.

    Handles cases where a grid cell is a dict or a list of dicts.
    """
    if isinstance(cell_obj, dict):
        bbox = cell_obj.get("bbox")
        return bbox if isinstance(bbox, dict) else None
    if isinstance(cell_obj, list):
        boxes = []
        for x in cell_obj:
            if isinstance(x, dict) and isinstance(x.get("bbox"), dict):
                boxes.append(x["bbox"])
        if not boxes:
            return None
        try:
            l = min(float(b.get("l") or 0) for b in boxes)
            t = min(float(b.get("t") or 0) for b in boxes)
            r = max(float(b.get("r") or 0) for b in boxes)
            btm = max(float(b.get("b") or 0) for b in boxes)
            return {"l": l, "t": t, "r": r, "b": btm, "coord_origin": "TOPLEFT"}
        except Exception:
            return None
    return None


def count_numbers_in_text(text: str) -> int:
    """
    Count numerical values in text.
    
    Used to determine if a page contains numerical data for NumF1 calculation.
    """
    # Match numbers with optional commas/dots (e.g., 1,234.56 or 1.234,56)
    numbers = re.findall(r'\d[\d,.]*', text)
    # Filter to only count numbers with at least 1 digit
    return len([n for n in numbers if any(c.isdigit() for c in n)])


@dataclass
class PageResult:
    """Result for a single page."""
    company: str
    page_number: int
    # Primary metrics
    format_agnostic_cer: float
    content_word_recall: float
    number_f1: float
    # Number F1 details
    number_precision: float = 0.0
    number_recall: float = 0.0
    # Meta
    ocr_text_length: int = 0
    gt_text_length: int = 0
    processing_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    # OCR output text
    ocr_text: Optional[str] = None
    # Raw OCR output (e.g., Docling markdown before canonicalization)
    ocr_text_raw: Optional[str] = None
    # Ground truth text
    gt_text: Optional[str] = None
    # Raw ground truth text as stored in dataset
    gt_text_raw: Optional[str] = None
    # Extraction mode used to generate ocr_text/gt_text for scoring
    extraction_mode: Optional[str] = None
    # Structured Docling tables for GUI (only for Docling-based engines)
    docling_tables: Optional[Any] = None
    # Flag to indicate if GT has numbers
    gt_has_numbers: bool = True
    # Peak VRAM usage in MB
    peak_vram_mb: Optional[float] = None


@dataclass
class CompanyResult:
    """Results for a single company with mean, std."""
    company: str
    pdf_path: str
    total_pages: int
    successful_pages: int
    
    # Mean metrics
    avg_format_agnostic_cer: float = 0.0
    avg_content_word_recall: float = 0.0
    avg_number_f1: float = 0.0
    
    # Std metrics
    std_format_agnostic_cer: float = 0.0
    std_content_word_recall: float = 0.0
    std_number_f1: float = 0.0
    
    # Aggregated metrics (calculated over all text/numbers in company, not per-page average)
    aggregated_word_recall: float = 0.0  # Total matched words / Total GT words
    aggregated_number_f1: float = 0.0  # F1 over all numbers in company
    aggregated_number_precision: float = 0.0
    aggregated_number_recall: float = 0.0
    pages_with_numbers: int = 0  # Count of pages that have numbers in GT
    
    total_time_seconds: float = 0.0
    
    page_results: List[PageResult] = field(default_factory=list)


@dataclass
class PageLevelBenchmarkResult:
    """Full benchmark results with mean ± std."""
    timestamp: str
    ocr_engine: str
    dpi: int
    total_companies: int
    total_pages: int
    successful_pages: int
    
    # Mean metrics
    overall_avg_format_agnostic_cer: float = 0.0
    overall_avg_content_word_recall: float = 0.0
    overall_avg_number_f1: float = 0.0
    
    # Std metrics
    overall_std_format_agnostic_cer: float = 0.0
    overall_std_content_word_recall: float = 0.0
    overall_std_number_f1: float = 0.0
    
    # Aggregated metrics (calculated over all text/numbers, not per-page average)
    overall_aggregated_word_recall: float = 0.0
    overall_aggregated_number_f1: float = 0.0
    overall_aggregated_number_precision: float = 0.0
    overall_aggregated_number_recall: float = 0.0
    total_pages_with_numbers: int = 0
    
    total_time_seconds: float = 0.0
    
    company_results: List[CompanyResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PageLevelBenchmark:
    """
    Benchmark OCR page-by-page by extracting specific pages from PDFs.
    """
    
    def __init__(
        self, 
        pdf_dir: str = None, 
        ocr_engine: str = "docling",  # "docling" or "marker"
        dpi: int = 300,
        marker_use_llm: bool = False,  # Use LLM for Marker post-processing (requires OpenRouter API key)
        table_only: bool = False,  # Only benchmark pages with financial tables
        save_ocr_outputs: bool = False,  # Save OCR text outputs for debugging
        ocr_outputs_dir: str = None,  # Directory to save OCR outputs
        # Strategy flags
        strategy1_layout_fallback: bool = False,
        strategy2_ensemble_voting: bool = False,
        strategy2_use_marker: bool = True,
    ):
        self.pdf_dir = Path(pdf_dir) if pdf_dir else PDF_SAMPLES_DIR
        self.ocr_engine = ocr_engine
        self.dpi = dpi
        self.marker_use_llm = marker_use_llm
        self.table_only = table_only
        self.save_ocr_outputs = save_ocr_outputs
        self.ocr_outputs_dir = Path(ocr_outputs_dir) if ocr_outputs_dir else Path("results/ocr_outputs")
        self.strategy1_layout_fallback = strategy1_layout_fallback
        self.strategy2_ensemble_voting = strategy2_ensemble_voting
        self.strategy2_use_marker = strategy2_use_marker
        self._dataset = None
        self._gt_by_company = None
        self._marker_service = None
        self._docling_service = None
        self._hybrid_service = None
    
    @property
    def dataset(self) -> VnPdfDataset:
        if self._dataset is None:
            self._dataset = VnPdfDataset()
        return self._dataset
    
    @property
    def gt_by_company(self) -> Dict[str, Dict[int, VnPdfSample]]:
        """Get ground truth samples organized by company and page number."""
        if self._gt_by_company is None:
            self._gt_by_company = {}
            for sample in self.dataset.get_samples():
                # Filter to table pages only if requested
                if self.table_only and not sample.is_table_page:
                    continue
                company = sample.custom_id.split('/')[2]
                if company not in self._gt_by_company:
                    self._gt_by_company[company] = {}
                self._gt_by_company[company][sample.page_number] = sample
        return self._gt_by_company
    
    def get_pdf_path(self, company: str) -> Optional[Path]:
        """Get PDF path for a company."""
        pattern = f"{company}*.pdf"
        matches = list(self.pdf_dir.glob(pattern))
        return matches[0] if matches else None
    
    def extract_page_image(self, pdf_path: Path, page_num: int) -> Optional[Image.Image]:
        """
        Extract a specific page from PDF as high-resolution image.
        
        Args:
            pdf_path: Path to PDF file
            page_num: 1-indexed page number
            
        Returns:
            PIL Image of the page, or None if failed
        """
        try:
            doc = fitz.open(pdf_path)
            
            if page_num < 1 or page_num > len(doc):
                logger.warning(f"Page {page_num} out of range for {pdf_path.name} ({len(doc)} pages)")
                return None
            
            page = doc[page_num - 1]  # fitz uses 0-indexed
            zoom = self.dpi / 72  # 72 is default PDF DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            doc.close()
            return img
            
        except Exception as e:
            logger.error(f"Failed to extract page {page_num} from {pdf_path}: {e}")
            return None
    
    def ocr_image(self, img: Image.Image) -> str:
        """Run OCR on an image using Docling."""
        # Lazy-load Docling service
        if self._docling_service is None:
            from services.ocr.docling import DoclingOCRService
            self._docling_service = DoclingOCRService()
        
        return self._docling_service.process_image(img)
    
    def ocr_pdf_page_with_marker(self, pdf_path: Path, page_num: int) -> str:
        """
        Run OCR on a specific PDF page using Marker.
        """
        # Create a temporary single-page PDF for Marker
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Extract single page to temp PDF
            doc = fitz.open(pdf_path)
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
            single_page_doc.save(tmp_path)
            single_page_doc.close()
            doc.close()
            
            # Lazy-load Marker service
            if self._marker_service is None:
                from services.ocr.marker import MarkerOCRService
                self._marker_service = MarkerOCRService(use_llm=self.marker_use_llm)
            
            # Run Marker OCR
            ocr_text = self._marker_service.process_pdf(tmp_path)
            return ocr_text
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def ocr_pdf_page_with_hybrid(self, pdf_path: Path, page_num: int) -> str:
        """
        Run OCR on a specific PDF page using Hybrid (Tesseract + Surya routing).
        
        Uses confidence-gated routing:
        1. Extract page as image
        2. Run Tesseract to get cells with confidence
        3. Route low-confidence cells to Surya
        4. Merge and return text
        """
        try:
            # Extract page as high-resolution image
            doc = fitz.open(pdf_path)
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return ""
            
            page = doc[page_num - 1]
            zoom = self.dpi / 72  # Convert DPI to zoom factor
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            doc.close()
            
            # Lazy-load Hybrid service
            if self._hybrid_service is None:
                from services.ocr.confidence_gated import ConfidenceGatedOCRService
                self._hybrid_service = ConfidenceGatedOCRService()
            
            # Process image with confidence-gated routing
            ocr_text = self._hybrid_service.process_image(img)
            return ocr_text
            
        except Exception as e:
            logger.error(f"Hybrid OCR failed for page {page_num}: {e}")
            return ""
    
    def ocr_pdf_page_with_hybrid_docling(self, pdf_path: Path, page_num: int) -> str:
        """
        Run OCR using Docling's full pipeline with HybridOcrModel.
        
        This uses:
        1. Docling's layout detection
        2. HybridOcrModel for OCR (Tesseract + Surya routing)
        3. Docling's table structure recognition
        4. Markdown export
        
        Returns formatted markdown output with table structure.
        """
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
            from docling.datamodel.accelerator_options import AcceleratorDevice
            from services.ocr.hybrid_pdf_pipeline import HybridPdfPipeline
            
            # Use HybridPdfPipeline which overrides _make_ocr_model()
            # to inject our confidence-gated HybridOcrModel
            # IMPORTANT: Provide explicit pipeline options.
            # If we rely on Docling defaults, some PDFs end up with OCR disabled
            # (export_to_markdown() becomes empty), which incorrectly looks "fast" and "successful".
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            # HybridPdfPipeline will read lang/force_full_page_ocr from these options.
            pipeline_options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, lang=["vie"])

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=HybridPdfPipeline,
                        pipeline_options=pipeline_options,
                    )
                }
            )
            
            # Convert PDF (single page extraction)
            # Create a single-page PDF for processing
            import tempfile
            import time
            
            doc = fitz.open(pdf_path)
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return ""
            
            # Create temp file path
            tmp_path = Path(tempfile.gettempdir()) / f"hybrid_docling_{page_num}_{time.time_ns()}.pdf"
            
            # Extract single page to temp file
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
            new_doc.save(str(tmp_path))
            new_doc.close()
            doc.close()
            
            try:
                # Run Docling conversion
                result = converter.convert(str(tmp_path))
                
                # Export to markdown
                md_text = result.document.export_to_markdown()
                
                return md_text
                
            finally:
                # Cleanup
                for _ in range(3):
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                        break
                    except PermissionError:
                        time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Hybrid Docling OCR failed for page {page_num}: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _convert_pdf_page_with_docling_pipeline(
        self,
        pdf_path: Path,
        page_num: int,
        pipeline_cls: Any,
        pipeline_options: Any,
    ) -> tuple[str, Dict[str, Any]]:
        """Convert a single PDF page with Docling and return (markdown, export_dict)."""
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=pipeline_cls,
                    pipeline_options=pipeline_options,
                )
            }
        )

        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return "", {}

        import tempfile
        import time as _time

        tmp_path = Path(tempfile.gettempdir()) / f"docling_single_{page_num}_{_time.time_ns()}.pdf"
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
        new_doc.save(str(tmp_path))
        new_doc.close()
        doc.close()

        try:
            result = converter.convert(str(tmp_path))
            md_text = result.document.export_to_markdown()
            export_dict = result.document.export_to_dict()
            return md_text, export_dict
        finally:
            for _ in range(3):
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                    break
                except PermissionError:
                    time.sleep(0.1)
    
    def ocr_pdf_page_with_laso(self, pdf_path: Path, page_num: int) -> str:
        """
        Run OCR using Docling's pipeline with LASOcrModel (Layout-Aware Speculative OCR).
        
        LASO features:
        1. Pre-OCR layout detection to identify table regions
        2. Speculative dual-engine execution for table cells
        3. Vietnamese number format validation for result selection
        4. Confidence routing for non-table regions
        """
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            try:
                import importlib

                LASOPdfPipeline = importlib.import_module("services.ocr.laso_pdf_pipeline").LASOPdfPipeline
            except Exception as e:
                logger.error(f"LASO pipeline is not available: {e}")
                return ""
            
            # Use LASOPdfPipeline for layout-aware speculative OCR
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=LASOPdfPipeline,
                    )
                }
            )
            
            # Convert PDF (single page extraction)
            import tempfile
            import time
            
            doc = fitz.open(pdf_path)
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return ""
            
            # Create temp file path
            tmp_path = Path(tempfile.gettempdir()) / f"laso_{page_num}_{time.time_ns()}.pdf"
            
            # Extract single page to temp file
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
            new_doc.save(str(tmp_path))
            new_doc.close()
            doc.close()
            
            try:
                # Run Docling conversion with LASO
                result = converter.convert(str(tmp_path))
                
                # Export to markdown
                md_text = result.document.export_to_markdown()
                
                return md_text
                
            finally:
                # Cleanup with retry for Windows
                for _ in range(3):
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                        break
                    except PermissionError:
                        time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"LASO OCR failed for page {page_num}: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _save_ocr_output(self, company: str, page_num: int, ocr_text: str, gt_text: str) -> None:
        """
        Save OCR output and ground truth text for debugging/analysis.
        
        Creates files:
        - {ocr_outputs_dir}/{engine}/{company}/page_{page_num}_ocr.txt
        - {ocr_outputs_dir}/{engine}/{company}/page_{page_num}_gt.txt
        """
        output_dir = self.ocr_outputs_dir / self.ocr_engine / company
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save OCR output
        ocr_path = output_dir / f"page_{page_num:03d}_ocr.txt"
        with open(ocr_path, 'w', encoding='utf-8') as f:
            f.write(ocr_text)
        
        # Save ground truth
        gt_path = output_dir / f"page_{page_num:03d}_gt.txt"
        with open(gt_path, 'w', encoding='utf-8') as f:
            f.write(gt_text)
    
    def benchmark_page(self, company: str, page_num: int, pdf_path: Path, gt_sample: VnPdfSample) -> PageResult:
        """Benchmark a single page."""
        start_time = time.time()
        peak_vram_mb = None
        
        try:
            # Reset VRAM peak stats for accurate per-page measurement
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                pass
            
            # Canonicalize GT to a cell-stream representation for scoring.
            # This avoids coupling to a specific markdown rendering while still
            # evaluating table content.
            gt_text_raw = gt_sample.text
            gt_grid = parse_pipe_table_to_grid(gt_text_raw)
            gt_eval_text = grid_to_canonical_text(gt_grid) if gt_grid else gt_text_raw

            # Run OCR based on engine
            if self.ocr_engine == "marker":
                logger.info(f"  Processing page {page_num} with Marker...")
                ocr_text_raw = self.ocr_pdf_page_with_marker(pdf_path, page_num)
                ocr_doc_dict = None
            elif self.ocr_engine == "hybrid":
                # Hybrid: Docling + Surya for low-confidence cells
                logger.info(f"  Processing page {page_num} with Hybrid (Tesseract+Surya)...")
                ocr_text_raw = self.ocr_pdf_page_with_hybrid(pdf_path, page_num)
                ocr_doc_dict = None
            elif self.ocr_engine == "hybrid_docling":
                # Hybrid Docling: Full Docling pipeline with HybridOcrModel
                logger.info(f"  Processing page {page_num} with Hybrid Docling...")
                from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
                from docling.datamodel.accelerator_options import AcceleratorDevice
                from services.ocr.hybrid_pdf_pipeline import HybridPdfPipeline

                pipeline_options = PdfPipelineOptions()
                pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA
                pipeline_options.do_ocr = True
                pipeline_options.do_table_structure = True
                pipeline_options.table_structure_options.do_cell_matching = True
                pipeline_options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, lang=["vie"])

                ocr_text_raw, ocr_doc_dict = self._convert_pdf_page_with_docling_pipeline(
                    pdf_path=pdf_path,
                    page_num=page_num,
                    pipeline_cls=HybridPdfPipeline,
                    pipeline_options=pipeline_options,
                )
            elif self.ocr_engine == "laso":
                # LASO: Layout-Aware Speculative OCR
                logger.info(f"  Processing page {page_num} with LASO...")
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.datamodel.accelerator_options import AcceleratorDevice
                try:
                    import importlib

                    LASOPdfPipeline = importlib.import_module("services.ocr.laso_pdf_pipeline").LASOPdfPipeline
                except Exception as e:
                    return PageResult(
                        company=company,
                        page_number=page_num,
                        format_agnostic_cer=1.0,
                        content_word_recall=0.0,
                        number_f1=0.0,
                        success=False,
                        error=f"LASO pipeline is not available: {e}",
                        gt_text=gt_eval_text,
                        gt_text_raw=gt_text_raw,
                    )

                pipeline_options = PdfPipelineOptions()
                pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA
                pipeline_options.do_ocr = True
                pipeline_options.do_table_structure = True
                pipeline_options.table_structure_options.do_cell_matching = True

                ocr_text_raw, ocr_doc_dict = self._convert_pdf_page_with_docling_pipeline(
                    pdf_path=pdf_path,
                    page_num=page_num,
                    pipeline_cls=LASOPdfPipeline,
                    pipeline_options=pipeline_options,
                )
            else:
                # Docling: Extract image and OCR
                img = self.extract_page_image(pdf_path, page_num)
                if img is None:
                    return PageResult(
                        company=company,
                        page_number=page_num,
                        format_agnostic_cer=1.0,
                        content_word_recall=0.0,
                        number_f1=0.0,
                        success=False,
                        error=f"Failed to extract page {page_num}"
                    )
                
                logger.info(f"  Extracted page {page_num}: {img.size[0]}x{img.size[1]} @ {self.dpi}dpi")
                ocr_text_raw = self.ocr_image(img)
                ocr_doc_dict = None

            # If OCR produced nothing, treat as a failure (do not score as 0 and mark success).
            if not ocr_text_raw or not ocr_text_raw.strip():
                return PageResult(
                    company=company,
                    page_number=page_num,
                    format_agnostic_cer=1.0,
                    content_word_recall=0.0,
                    number_f1=0.0,
                    success=False,
                    error="Empty OCR output",
                    gt_text=gt_eval_text,
                    gt_text_raw=gt_text_raw,
                )

            # Canonicalize OCR output into a table cell-stream for scoring.
            extraction_mode = "raw"
            docling_tables_payload: Any = None
            page_img: Optional[Image.Image] = None

            if self.ocr_engine in {"hybrid_docling", "laso"}:
                # For Docling-based pipelines, require actual table extraction.
                doc_dict = ocr_doc_dict if isinstance(ocr_doc_dict, dict) else {}
                ocr_grid_str = extract_docling_tables_grid(doc_dict)
                docling_tables_payload = doc_dict.get("tables")

                # Strategy 1: fallback layout-table extraction if Docling yields no tables.
                if not ocr_grid_str and self.strategy1_layout_fallback:
                    page_img = self.extract_page_image(pdf_path, page_num)
                    if page_img is not None:
                        words = _tesseract_tsv_words(page_img, lang="vie")
                        grid_cells = _layout_table_from_words(words)
                        if grid_cells:
                            ocr_grid_str = [[c.get("text", "") for c in row] for row in grid_cells]
                            extraction_mode = "layout_table_fallback"
                            # Synthetic payload for GUI/debug.
                            docling_tables_payload = [
                                {
                                    "label": "layout_table_fallback",
                                    "data": {"grid": grid_cells},
                                }
                            ]

                if not ocr_grid_str:
                    return PageResult(
                        company=company,
                        page_number=page_num,
                        format_agnostic_cer=1.0,
                        content_word_recall=0.0,
                        number_f1=0.0,
                        success=False,
                        error="No tables extracted by Docling",
                        ocr_text_raw=ocr_text_raw,
                        gt_text=gt_eval_text,
                        gt_text_raw=gt_text_raw,
                        extraction_mode="docling_no_tables",
                        docling_tables=docling_tables_payload,
                    )

                # Default extraction mode when Docling tables exist.
                if extraction_mode == "raw":
                    extraction_mode = "docling_grid"

                # Strategy 2: ensemble voting with Marker (when available).
                marker_grid: Optional[List[List[str]]] = None
                if self.strategy2_ensemble_voting and self.strategy2_use_marker:
                    # Only pay the Marker cost when Docling output looks sparse.
                    total_cells = sum(len(r) for r in ocr_grid_str)
                    nonempty_cells = sum(1 for r in ocr_grid_str for c in r if (c or "").strip())
                    sparse = (total_cells == 0) or (nonempty_cells / max(1, total_cells) < 0.55)
                    if sparse:
                        try:
                            marker_md = self.ocr_pdf_page_with_marker(pdf_path, page_num)
                            marker_grids = _extract_pipe_table_blocks(marker_md)
                            if marker_grids:
                                def grid_quality(g: List[List[str]]) -> tuple[int, int, int, int]:
                                    r = len(g)
                                    cmax = max((len(rr) for rr in g), default=0)
                                    tot = sum(len(rr) for rr in g)
                                    nonempty = sum(1 for rr in g for cc in rr if (cc or "").strip())
                                    # Prefer denser/larger tables.
                                    return (nonempty, tot, r, cmax)

                                marker_grid = max(marker_grids, key=grid_quality)
                        except Exception:
                            marker_grid = None

                # Strategy 2: fill-only voting (avoid regressions).
                if self.strategy2_ensemble_voting and marker_grid:
                    new_grid: List[List[str]] = []
                    for ri, row in enumerate(ocr_grid_str):
                        new_row: List[str] = []
                        for ci, cell_text in enumerate(row):
                            baseline = cell_text
                            candidates = [baseline]

                            # Pull a marker candidate only to fill missing cells (avoid regressions).
                            if (
                                not (baseline or "").strip()
                                and ri < len(marker_grid)
                                and ci < len(marker_grid[ri])
                            ):
                                candidates.append(marker_grid[ri][ci])

                            chosen = _pick_best_candidate_with_baseline(
                                candidates,
                                prefer_numeric=bool(re.search(r"\d", baseline or "")),
                                baseline=baseline,
                            )
                            new_row.append(chosen)
                        new_grid.append(new_row)
                    ocr_grid_str = new_grid

                ocr_eval_text = grid_to_canonical_text(ocr_grid_str)
            else:
                # Non-Docling engines: best-effort parsing.
                ocr_grid_md = parse_pipe_table_to_grid(ocr_text_raw)
                if ocr_grid_md:
                    ocr_eval_text = grid_to_canonical_text(ocr_grid_md)
                    extraction_mode = "pipe_table"
                else:
                    ocr_table_text = extract_table_content_robust(ocr_text_raw)
                    if ocr_table_text.strip():
                        ocr_eval_text = ocr_table_text
                        extraction_mode = "aligned_lines"
                    else:
                        # Fail-closed: for this benchmark, we evaluate table extraction.
                        # If we can't recover any table-like structure/text, treat as failure.
                        return PageResult(
                            company=company,
                            page_number=page_num,
                            format_agnostic_cer=1.0,
                            content_word_recall=0.0,
                            number_f1=0.0,
                            success=False,
                            error="No table-like content extracted",
                            ocr_text_raw=ocr_text_raw,
                            gt_text=gt_eval_text,
                            gt_text_raw=gt_text_raw,
                            extraction_mode="no_table_like_content",
                        )
            
            # Check if ground truth has numbers (for conditional NumF1 averaging)
            gt_number_count = count_numbers_in_text(gt_sample.text)
            has_numbers = gt_number_count > 0
            
            # Calculate metrics using canonicalized table content
            metrics = calculate_all_metrics(ocr_eval_text, gt_eval_text)
            
            # Capture peak VRAM usage
            try:
                import torch
                if torch.cuda.is_available():
                    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            except ImportError:
                pass
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Extract number F1 details
            num_details = metrics["number_f1"].details or {}
            
            result = PageResult(
                company=company,
                page_number=page_num,
                # Primary metrics
                format_agnostic_cer=metrics["format_agnostic_cer"].value,
                content_word_recall=metrics["content_word_recall"].value,
                number_f1=metrics["number_f1"].value,
                # Number F1 details
                number_precision=num_details.get("precision", 0.0),
                number_recall=num_details.get("recall", 0.0),
                # Meta
                ocr_text_length=len(ocr_eval_text),
                gt_text_length=len(gt_eval_text),
                processing_time_ms=elapsed_ms,
                success=True,
                # Always store OCR and GT text for aggregation
                ocr_text=ocr_eval_text,
                ocr_text_raw=ocr_text_raw,
                gt_text=gt_eval_text,
                gt_text_raw=gt_text_raw,
                extraction_mode=extraction_mode,
                docling_tables=docling_tables_payload,
                # Flag for conditional NumF1 averaging
                gt_has_numbers=has_numbers,
                # GPU memory usage
                peak_vram_mb=peak_vram_mb,
            )
            
            # Save OCR output to file if enabled
            if self.save_ocr_outputs:
                self._save_ocr_output(company, page_num, ocr_text_raw, gt_text_raw)
            
            logger.info(f"    FA-CER: {result.format_agnostic_cer:.4f}, WordRecall: {result.content_word_recall:.2%}, NumF1: {result.number_f1:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return PageResult(
                company=company,
                page_number=page_num,
                format_agnostic_cer=1.0,
                content_word_recall=0.0,
                number_f1=0.0,
                success=False,
                error=str(e)
            )
    
    def benchmark_company(self, company: str, max_pages: int = None) -> Optional[CompanyResult]:
        """Benchmark pages for a company.
        
        Args:
            company: Company code
            max_pages: Maximum pages to process (None = all pages)
        """
        pdf_path = self.get_pdf_path(company)
        if not pdf_path:
            logger.warning(f"No PDF found for {company}")
            return None
        
        gt_pages = self.gt_by_company.get(company, {})
        if not gt_pages:
            logger.warning(f"No ground truth for {company}")
            return None
        
        # Apply page limit BEFORE processing
        sorted_pages = sorted(gt_pages.items())
        if max_pages is not None:
            sorted_pages = sorted_pages[:max_pages]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking {company}: {len(sorted_pages)} pages" + (f" (limited from {len(gt_pages)})" if max_pages else ""))
        logger.info(f"PDF: {pdf_path.name}")
        logger.info(f"{'='*60}")
        
        result = CompanyResult(
            company=company,
            pdf_path=str(pdf_path),
            total_pages=len(sorted_pages),
            successful_pages=0,
        )
        
        start_time = time.time()
        
        for page_num, gt_sample in sorted_pages:
            logger.info(f"Processing page {page_num}...")
            page_result = self.benchmark_page(company, page_num, pdf_path, gt_sample)
            result.page_results.append(page_result)
            
            if page_result.success:
                result.successful_pages += 1
            
            # Clear CUDA cache after each page to prevent OOM from memory fragmentation
            if self.ocr_engine == "marker":
                try:
                    import torch
                    import gc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                except ImportError:
                    pass
        
        result.total_time_seconds = time.time() - start_time
        
        # Calculate mean ± std for all metrics
        successful = [p for p in result.page_results if p.success]
        # For NumF1, only include pages where ground truth has numbers
        pages_with_numbers = [p for p in successful if p.gt_has_numbers]
        
        if successful:
            # Mean (FA-CER and WordRecall use all successful pages)
            result.avg_format_agnostic_cer = sum(p.format_agnostic_cer for p in successful) / len(successful)
            result.avg_content_word_recall = sum(p.content_word_recall for p in successful) / len(successful)
            
            # Std
            result.std_format_agnostic_cer = compute_std([p.format_agnostic_cer for p in successful])
            result.std_content_word_recall = compute_std([p.content_word_recall for p in successful])
        
        # NumF1: Only average over pages with numbers in ground truth
        if pages_with_numbers:
            result.avg_number_f1 = sum(p.number_f1 for p in pages_with_numbers) / len(pages_with_numbers)
            result.std_number_f1 = compute_std([p.number_f1 for p in pages_with_numbers])
        
        result.pages_with_numbers = len(pages_with_numbers)
        
        # Calculate AGGREGATED metrics (over all text/numbers, not per-page averages)
        # This is more robust for pages with few numbers
        if successful:
            # Concatenate all OCR and GT text
            all_ocr_text = '\n'.join(p.ocr_text or '' for p in successful)
            all_gt_text = '\n'.join(p.gt_text or '' for p in successful)
            
            # Aggregated Word Recall
            agg_word_recall = calculate_content_word_recall(all_ocr_text, all_gt_text)
            result.aggregated_word_recall = agg_word_recall.value
            
            # Aggregated Number F1
            agg_num_f1 = calculate_number_precision_recall_f1(all_ocr_text, all_gt_text)
            result.aggregated_number_f1 = agg_num_f1.value
            result.aggregated_number_precision = agg_num_f1.details.get("precision", 0.0) if agg_num_f1.details else 0.0
            result.aggregated_number_recall = agg_num_f1.details.get("recall", 0.0) if agg_num_f1.details else 0.0
        
        logger.info(f"\n{company} Summary:")
        logger.info(f"  Pages: {result.successful_pages}/{result.total_pages}")
        logger.info("  Per-Page Avg (mean ± std):")
        logger.info(f"    FA-CER: {result.avg_format_agnostic_cer:.2%} ± {result.std_format_agnostic_cer:.2%}")
        logger.info(f"    Word Recall: {result.avg_content_word_recall:.2%} ± {result.std_content_word_recall:.2%}")
        logger.info(f"    Number F1: {result.avg_number_f1:.2%} ± {result.std_number_f1:.2%} (n={result.pages_with_numbers} pages)")
        logger.info("  Aggregated:")
        logger.info(f"    Word Recall: {result.aggregated_word_recall:.2%}")
        logger.info(f"    Number F1: {result.aggregated_number_f1:.2%} (P={result.aggregated_number_precision:.2%}, R={result.aggregated_number_recall:.2%})")
        logger.info(f"  Time: {result.total_time_seconds:.1f}s")
        
        return result
    
    def run(self, companies: List[str] = None, max_pages_per_company: int = None) -> PageLevelBenchmarkResult:
        """
        Run benchmark on specified companies.
        
        Args:
            companies: List of company codes, or None for all
            max_pages_per_company: Limit pages per company for quick testing
        """
        logger.info("Starting Page-Level OCR Benchmark")
        logger.info(f"DPI: {self.dpi}, OCR Engine: {self.ocr_engine}")
        
        if companies is None:
            companies = COMPANY_CODES
        
        result = PageLevelBenchmarkResult(
            timestamp=datetime.now().isoformat(),
            ocr_engine=self.ocr_engine,
            dpi=self.dpi,
            total_companies=len(companies),
            total_pages=0,
            successful_pages=0,
        )
        
        start_time = time.time()
        
        for company in companies:
            # Pass max_pages to benchmark_company to limit BEFORE processing
            company_result = self.benchmark_company(company, max_pages=max_pages_per_company)
            if company_result:
                result.company_results.append(company_result)
                result.total_pages += company_result.total_pages
                result.successful_pages += company_result.successful_pages
        
        result.total_time_seconds = time.time() - start_time
        
        # Calculate overall mean ± std
        all_successful = []
        for cr in result.company_results:
            all_successful.extend([p for p in cr.page_results if p.success])
        
        # For NumF1, only include pages where ground truth has numbers
        all_with_numbers = [p for p in all_successful if p.gt_has_numbers]
        
        if all_successful:
            # Mean (FA-CER and WordRecall use all successful pages)
            result.overall_avg_format_agnostic_cer = sum(p.format_agnostic_cer for p in all_successful) / len(all_successful)
            result.overall_avg_content_word_recall = sum(p.content_word_recall for p in all_successful) / len(all_successful)
            
            # Std
            result.overall_std_format_agnostic_cer = compute_std([p.format_agnostic_cer for p in all_successful])
            result.overall_std_content_word_recall = compute_std([p.content_word_recall for p in all_successful])
        
        # NumF1: Only average over pages with numbers in ground truth
        if all_with_numbers:
            result.overall_avg_number_f1 = sum(p.number_f1 for p in all_with_numbers) / len(all_with_numbers)
            result.overall_std_number_f1 = compute_std([p.number_f1 for p in all_with_numbers])
        
        result.total_pages_with_numbers = len(all_with_numbers)
        
        # Calculate OVERALL AGGREGATED metrics
        if all_successful:
            # Concatenate all OCR and GT text
            all_ocr_text = '\n'.join(p.ocr_text or '' for p in all_successful)
            all_gt_text = '\n'.join(p.gt_text or '' for p in all_successful)
            
            # Aggregated Word Recall
            agg_word_recall = calculate_content_word_recall(all_ocr_text, all_gt_text)
            result.overall_aggregated_word_recall = agg_word_recall.value
            
            # Aggregated Number F1
            agg_num_f1 = calculate_number_precision_recall_f1(all_ocr_text, all_gt_text)
            result.overall_aggregated_number_f1 = agg_num_f1.value
            result.overall_aggregated_number_precision = agg_num_f1.details.get("precision", 0.0) if agg_num_f1.details else 0.0
            result.overall_aggregated_number_recall = agg_num_f1.details.get("recall", 0.0) if agg_num_f1.details else 0.0
        
        logger.info(f"\n{'='*60}")
        logger.info("PAGE-LEVEL BENCHMARK COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Companies: {result.total_companies}")
        logger.info(f"Pages: {result.successful_pages}/{result.total_pages}")
        logger.info("\nPer-Page Avg (mean ± std):")
        logger.info(f"  FA-CER: {result.overall_avg_format_agnostic_cer:.4f} ± {result.overall_std_format_agnostic_cer:.4f}")
        logger.info(f"  Word Recall: {result.overall_avg_content_word_recall:.2%} ± {result.overall_std_content_word_recall:.2%}")
        logger.info(f"  Number F1: {result.overall_avg_number_f1:.2%} ± {result.overall_std_number_f1:.2%} (n={result.total_pages_with_numbers} pages)")
        logger.info("\nAggregated:")
        logger.info(f"  Word Recall: {result.overall_aggregated_word_recall:.2%}")
        logger.info(f"  Number F1: {result.overall_aggregated_number_f1:.2%} (P={result.overall_aggregated_number_precision:.2%}, R={result.overall_aggregated_number_recall:.2%})")
        logger.info(f"\nTotal Time: {result.total_time_seconds:.1f}s")
        
        return result
    
    def save_results(self, result: PageLevelBenchmarkResult, output_path: str) -> None:
        """Save results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Run benchmark from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run page-level OCR benchmark")
    parser.add_argument("--companies", nargs="*", help="Company codes to benchmark")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages per company")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for page extraction (default: 300, try 400-600 for scanned PDFs)")
    parser.add_argument("--engine", type=str, default="docling", choices=["docling", "marker", "hybrid", "hybrid_docling", "laso"], help="OCR engine")
    parser.add_argument("--marker-llm", action="store_true", help="Use LLM with Marker (requires OPENROUTER_API_KEY)")
    parser.add_argument("--table-only", action="store_true", help="Only benchmark pages with financial tables")
    parser.add_argument("--output", type=str, default="results/page_level_benchmark.json")
    parser.add_argument("--save-outputs", action="store_true", help="Save OCR outputs for debugging/analysis")
    parser.add_argument("--outputs-dir", type=str, default="results/ocr_outputs", help="Directory to save OCR outputs")

    # Strategy flags (1/2)
    parser.add_argument(
        "--s1-layout-fallback",
        action="store_true",
        help="Strategy 1: if Docling extracts no tables, infer a layout-table grid from Tesseract TSV words",
    )
    parser.add_argument(
        "--s2-ensemble",
        action="store_true",
        help="Strategy 2: ensemble voting using Marker as a second candidate table extraction",
    )
    parser.add_argument(
        "--s2-no-marker",
        action="store_true",
        help="Disable Marker usage inside Strategy 2 (keeps flag compatibility)",
    )
    
    args = parser.parse_args()
    
    benchmark = PageLevelBenchmark(
        ocr_engine=args.engine,
        dpi=args.dpi,
        marker_use_llm=args.marker_llm,
        table_only=args.table_only,
        save_ocr_outputs=args.save_outputs,
        ocr_outputs_dir=args.outputs_dir,
        strategy1_layout_fallback=args.s1_layout_fallback,
        strategy2_ensemble_voting=args.s2_ensemble,
        strategy2_use_marker=(args.s2_ensemble and (not args.s2_no_marker)),
    )
    result = benchmark.run(companies=args.companies, max_pages_per_company=args.max_pages)
    benchmark.save_results(result, args.output)


if __name__ == "__main__":
    main()
