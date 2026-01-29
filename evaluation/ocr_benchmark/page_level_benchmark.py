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
import unicodedata
from datetime import datetime
from PIL import Image, ImageDraw
import io
import tempfile
import os

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
    """Compute sample standard deviation of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def extract_table_content_robust(text: str) -> str:
    """Extract table-like content from OCR output in a format-robust way."""
    if not text:
        return ""

    lines = text.splitlines()

    def is_markdown_pipe_row(line: str) -> bool:
        s = line.strip()
        if s.count("|") < 2:
            return False
        if set(s.replace("|", "").strip()) <= {"-", ":"} and "-" in s:
            return False
        return True

    pipe_rows = [ln for ln in lines if is_markdown_pipe_row(ln)]
    if pipe_rows:
        return "\n".join(pipe_rows)

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


def extract_table_content_fallback(text: str) -> str:
    """Fallback extraction when no table-like rows are detected."""
    if not text:
        return ""

    s = text.replace("\r\n", "\n").replace("\r", "\n")
    # Drop fenced code blocks.
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    # Drop markdown table separators.
    s = re.sub(r"^\s*\|?\s*[:-]+\s*(?:\|\s*[:-]+\s*)+\|?\s*$", "", s, flags=re.M)
    # Drop headings.
    s = re.sub(r"^\s{0,3}#{1,6}\s+.*$", "", s, flags=re.M)

    kept: list[str] = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t:
            continue
        has_digit = bool(re.search(r"\d", t))
        has_currency = bool(re.search(r"(?i)\b(vnd|vnđ|usd|eur)\b|%|đ", t))
        has_cols = bool(re.search(r"\S(?:\s{2,}|\t)\S", t))
        if has_digit or has_currency or has_cols:
            kept.append(ln)

    return "\n".join(kept)


def extract_sectioned_rows(text: str) -> str:
    """Extract section-numbered rows (e.g. "2.1 ...") as a pseudo table.
    """
    if not text:
        return ""

    s = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = s.splitlines()

    # Match common section header forms:
    # - "## 2.1 Title"
    # - "2.1 Title"
    # - "2; Title" (OCR noise for "2.")
    header_re = re.compile(r"^\s*(?:#{1,6}\s*)?(\d+(?:\.\d+)*)\s*[\.;:,;)]?\s+(\S.*)$")

    out: list[str] = []
    cur_num: Optional[str] = None
    cur_text_parts: list[str] = []

    def _flush() -> None:
        nonlocal cur_num, cur_text_parts
        if cur_num is None:
            return
        rhs = " ".join(p for p in (x.strip() for x in cur_text_parts) if p)
        if rhs:
            out.append(f"{cur_num}\t{rhs}")
        cur_num = None
        cur_text_parts = []

    for ln in lines:
        t = ln.strip()
        if not t:
            continue

        m = header_re.match(t)
        if m:
            _flush()
            num = (m.group(1) or "").strip()
            # If it's a top-level section number like "2", align with GT "2.".
            if num.isdigit():
                num = f"{num}."
            cur_num = num
            cur_text_parts = [(m.group(2) or "").strip()]
            continue

        # Continuation line: attach to the current section (paragraphs / bullets).
        if cur_num is not None:
            # Strip common markdown bullet prefixes.
            t2 = re.sub(r"^\s*(?:[-*•]+)\s+", "", t)
            cur_text_parts.append(t2)

    _flush()
    return "\n".join(out)


def _normalize_markdownish_rows(text: str) -> List[str]:
    """Best-effort row splitting for pipe-table strings."""
    if not text:
        return []

    s = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" not in s and "|" in s:
        s = re.sub(r"\s\|", "\n|", s)
    return s.splitlines()


def parse_pipe_table_to_grid(text: str) -> List[List[str]]:
    """Parse a markdown-ish pipe table into a 2D grid of cell strings."""
    rows: List[List[str]] = []
    for ln in _normalize_markdownish_rows(text):
        s = ln.strip()
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
        rows.append(parts)
    return rows


def grid_to_canonical_text(grid: List[List[str]]) -> str:
    """Canonicalize a 2D grid into a stable text representation for metrics."""
    out_lines: List[str] = []
    for row in grid:
        if not row:
            continue
        cleaned = [c.strip() for c in row]
        if not any(cleaned):
            continue
        out_lines.append("\t".join(cleaned))
    return "\n".join(out_lines)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _normalize_for_match(s: str) -> str:
    s = (s or "").lower()
    s = _strip_accents(s)
    s = s.replace("đ", "d")
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_FINANCIAL_STATEMENT_KEYWORDS = (
    "bang can doi ke toan",
    "bao cao ket qua hoat dong kinh doanh",
    "bao cao luu chuyen tien te",
    "thuyet minh bao cao tai chinh",
    "bao cao tai chinh",
    "can doi ke toan",
    "ket qua hoat dong kinh doanh",
    "luu chuyen tien te",
    "thuyet minh",
    "balance sheet",
    "income statement",
    "statement of financial position",
    "cash flow",
    "notes to the financial statements",
)


def looks_like_financial_statement(text: str) -> bool:
    """Heuristic: does this page look like a financial statement / notes table page?"""
    s = _normalize_for_match(text)
    if not s:
        return False
    if any(k in s for k in _FINANCIAL_STATEMENT_KEYWORDS):
        return True
    digits = sum(1 for c in s if c.isdigit())
    if len(s) > 0 and (digits / len(s)) > 0.10:
        return True
    return False


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


def _markdown_table_to_rows(md_text: str) -> List[List[str]]:
    """Best-effort parse of the first markdown table block.
    """
    lines = (md_text or "").splitlines()
    table_lines: List[str] = []
    in_table = False

    for ln in lines:
        s = ln.strip("\n")
        if not s:
            if in_table:
                break
            continue
        if "|" in s and s.lstrip().startswith("|"):
            in_table = True
            table_lines.append(s)
        elif in_table:
            break

    rows: List[List[str]] = []
    for tl in table_lines:
        # Skip separator rows like: | --- | --- |
        if re.fullmatch(r"\|?\s*[-: ]+(\|\s*[-: ]+)+\|?\s*", tl):
            continue
        parts = [p.strip() for p in tl.strip().strip("|").split("|")]
        if parts:
            rows.append(parts)

    return rows


def _coerce_bbox_to_tuple(bbox: Any) -> Optional[tuple[float, float, float, float]]:
    if not isinstance(bbox, dict):
        return None
    try:
        return (
            float(bbox.get("l") or 0.0),
            float(bbox.get("t") or 0.0),
            float(bbox.get("r") or 0.0),
            float(bbox.get("b") or 0.0),
        )
    except Exception:
        return None


def _intersect_area_lt(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    al, at, ar, ab = a
    bl, bt, br, bb = b
    x0 = max(al, bl)
    y0 = max(at, bt)
    x1 = min(ar, br)
    y1 = min(ab, bb)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


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
            left = min(float(b.get("l") or 0) for b in boxes)
            t = min(float(b.get("t") or 0) for b in boxes)
            r = max(float(b.get("r") or 0) for b in boxes)
            btm = max(float(b.get("b") or 0) for b in boxes)
            return {"l": left, "t": t, "r": r, "b": btm, "coord_origin": "TOPLEFT"}
        except Exception:
            return None
    return None


def rewrite_docling_table_grid_text_from_ocr_cells(
    export_dict: Dict[str, Any],
    *,
    ocr_cells_debug: Optional[Dict[str, Any]],
    min_overlap_ratio: float = 0.08,
) -> bool:
    """Rewrite export_dict['tables'][*]['data']['grid'][r][c]['text'] using OCR cells.

    Uses overlap between each grid cell bbox and OCR `parsed_textline_cells` bboxes.
    Returns True if any cell text was changed.
    """
    if not isinstance(export_dict, dict):
        return False

    tables = export_dict.get("tables")
    if not isinstance(tables, list) or not tables:
        return False

    cells = []
    if isinstance(ocr_cells_debug, dict):
        raw = ocr_cells_debug.get("parsed_textline_cells")
        if isinstance(raw, list):
            cells = raw

    def _sanitize_ocr_text(s: str) -> str:
        s2 = (s or "").replace("|", " ")
        s2 = re.sub(r"\s+", " ", s2).strip()
        return s2

    ocr_items: list[tuple[tuple[float, float, float, float], str]] = []
    for c in cells:
        if not isinstance(c, dict):
            continue
        bbox = _coerce_bbox_to_tuple(c.get("bbox"))
        if bbox is None:
            continue
        text = _sanitize_ocr_text(str(c.get("text") or ""))
        if not text:
            continue
        ocr_items.append((bbox, text))

    if not ocr_items:
        return False

    changed = False

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
            for cell_obj in row:
                cell_bbox_dict = _coerce_cell_to_bbox(cell_obj)
                cell_bbox = _coerce_bbox_to_tuple(cell_bbox_dict)
                if cell_bbox is None:
                    continue

                cl, ct, cr, cb = cell_bbox
                cell_area = max(0.0, (cr - cl)) * max(0.0, (cb - ct))
                if cell_area <= 0.0:
                    continue

                hits: list[tuple[float, str]] = []
                for obox, otext in ocr_items:
                    ia = _intersect_area_lt(cell_bbox, obox)
                    if ia <= 0.0:
                        continue
                    if ia / cell_area < float(min_overlap_ratio):
                        continue
                    hits.append((float(obox[0]), otext))

                if not hits:
                    continue

                hits.sort(key=lambda x: x[0])
                new_text = _sanitize_ocr_text(" ".join(h[1] for h in hits))
                if not new_text:
                    continue

                if sum(ch.isalnum() for ch in new_text) == 0:
                    continue

                def _rewrite_one(cell_dict: dict) -> None:
                    nonlocal changed
                    old = str(cell_dict.get("text") or "")
                    if old != new_text:
                        cell_dict["text"] = new_text
                        changed = True

                if isinstance(cell_obj, dict):
                    _rewrite_one(cell_obj)
                elif isinstance(cell_obj, list):
                    for x in cell_obj:
                        if isinstance(x, dict):
                            _rewrite_one(x)

    return changed


def rewrite_docling_table_grid_text_from_surya_updates(
    export_dict: Dict[str, Any],
    *,
    update_diffs: Any,
    min_overlap_ratio: float = 0.08,
) -> bool:
    """Safer rewrite: only touch grid cells overlapped by accepted Surya updates.

    `update_diffs` is expected to be a list of dicts (from HybridOcrModel.get_update_diffs)
    with keys: bbox{l,t,r,b}, candidate, accepted.
    """

    if not isinstance(export_dict, dict):
        return False

    tables = export_dict.get("tables")
    if not isinstance(tables, list) or not tables:
        return False

    if not isinstance(update_diffs, list) or not update_diffs:
        return False

    def _sanitize(s: str) -> str:
        s2 = (s or "").replace("|", " ")
        s2 = re.sub(r"\s+", " ", s2).strip()
        return s2

    updates: list[tuple[tuple[float, float, float, float], str, str]] = []
    for d in update_diffs:
        if not isinstance(d, dict):
            continue
        if not bool(d.get("accepted")):
            continue
        bbox = _coerce_bbox_to_tuple(d.get("bbox"))
        if bbox is None:
            continue
        base_txt = _sanitize(str(d.get("baseline") or ""))
        cand_txt = _sanitize(str(d.get("candidate") or ""))

        # Ignore no-op updates (including whitespace-only diffs).
        if base_txt == cand_txt:
            continue

        if not cand_txt:
            continue
        # Numeric-only safety: only use updates that contain at least one digit.
        if not re.search(r"\d", cand_txt):
            continue
        updates.append((bbox, base_txt, cand_txt))

    if not updates:
        return False

    changed = False

    def _cell_text(cell: Any) -> str:
        if isinstance(cell, dict):
            return str(cell.get("text") or "")
        if isinstance(cell, list):
            # Merge spans / merged cells best-effort.
            parts: list[str] = []
            for x in cell:
                if isinstance(x, dict) and (x.get("text") or "").strip():
                    parts.append(str(x.get("text") or "").strip())
            return " ".join(parts)
        return ""

    def _area_lt(b: tuple[float, float, float, float]) -> float:
        l, t0, r, b0 = b
        return max(0.0, (r - l)) * max(0.0, (b0 - t0))

    def _is_numeric_like_cell_text(s: str) -> bool:
        # Conservative gate: only allow rewriting cells that are very likely numeric.
        # Prevent a numeric update bbox from clobbering a text label cell.
        try:
            from services.ocr.hybrid_ocr_model import numeric_likeness as _nl

            is_num, is_hdr, _score = _nl(s)
            return bool(is_num and (not is_hdr))
        except Exception:
            # Fallback: digits required and few alpha chars.
            s2 = (s or "").strip()
            if not re.search(r"\d", s2):
                return False
            alpha = sum(ch.isalpha() for ch in s2)
            return alpha <= 1

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
            for cell_obj in row:
                cell_bbox_dict = _coerce_cell_to_bbox(cell_obj)
                cell_bbox = _coerce_bbox_to_tuple(cell_bbox_dict)
                if cell_bbox is None:
                    continue

                old_text = _sanitize(_cell_text(cell_obj))
                if old_text and (not _is_numeric_like_cell_text(old_text)):
                    # Only rewrite numeric-like target cells.
                    continue

                cl, ct, cr, cb = cell_bbox
                cell_area = max(0.0, (cr - cl)) * max(0.0, (cb - ct))
                if cell_area <= 0.0:
                    continue

                matches: list[tuple[float, float, str, str]] = []
                for ub, ubase, ucand in updates:
                    ia = _intersect_area_lt(cell_bbox, ub)
                    if ia <= 0.0:
                        continue
                    cell_overlap = ia / cell_area
                    if cell_overlap < float(min_overlap_ratio):
                        continue

                    ua = _area_lt(ub)
                    update_overlap = (ia / ua) if ua > 0 else 0.0
                    # Require the update box to mostly lie within the chosen cell.
                    if update_overlap < 0.35:
                        continue

                    score = cell_overlap * 0.6 + update_overlap * 0.4
                    matches.append((float(score), float(ub[0]), str(ubase), str(ucand)))

                if not matches:
                    continue

                # Apply updates in descending score order.
                # Use string-level substitution to avoid clobbering multi-value cells.
                matches.sort(key=lambda x: (x[0], x[1]), reverse=True)

                def _apply_updates_to_text(old: str) -> str:
                    cur = _sanitize(old)
                    for _score, _x, ubase, ucand in matches:
                        if not ubase or not ucand:
                            continue
                        if ubase == ucand:
                            continue
                        # Prefer targeted replacement when the baseline appears as a
                        # substring inside a merged cell.
                        if ubase in cur:
                            cur = cur.replace(ubase, ucand, 1)
                            continue
                        # Fallback: only overwrite if this cell is essentially the
                        # baseline (avoid dropping other numbers).
                        if cur.strip() == ubase.strip():
                            cur = ucand
                    return cur

                def _rewrite_one(cell_dict: dict) -> None:
                    nonlocal changed
                    old = str(cell_dict.get("text") or "")
                    new = _apply_updates_to_text(old)
                    if new and old != new:
                        cell_dict["text"] = new
                        changed = True

                if isinstance(cell_obj, dict):
                    _rewrite_one(cell_obj)
                elif isinstance(cell_obj, list):
                    for x in cell_obj:
                        if isinstance(x, dict):
                            _rewrite_one(x)

    return changed


def _draw_overlays_for_hybrid_docling(
    *,
    pdf_path: Path,
    page_num: int,
    dpi: int,
    export_dict: Dict[str, Any],
    overlays_dir: Path,
) -> None:
    """Optionally export a PNG overlay for the page.

    Draws:
    - table bbox (yellow) if present in Docling export
    - routed-to-Surya cell bbox (red) if present in debug snapshot
    """

    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return

        page = doc[page_num - 1]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
        doc.close()

        draw = ImageDraw.Draw(img, "RGBA")

        # Table boxes from Docling export.
        tables = export_dict.get("tables") if isinstance(export_dict, dict) else None
        if isinstance(tables, list):
            for t in tables:
                if not isinstance(t, dict):
                    continue
                bbox = _coerce_bbox_to_tuple(t.get("bbox"))
                if bbox is None:
                    continue
                l, t0, r, b0 = bbox
                draw.rectangle((l * zoom, t0 * zoom, r * zoom, b0 * zoom), outline=(255, 215, 0, 220), width=3)

        # Surya-routed cells.
        snap = export_dict.get("ocr_cells_debug") if isinstance(export_dict, dict) else None
        if isinstance(snap, dict):
            routed = snap.get("cells_routed_to_surya")
            if isinstance(routed, list):
                for c in routed:
                    if not isinstance(c, dict):
                        continue
                    bbox = _coerce_bbox_to_tuple(c.get("bbox"))
                    if bbox is None:
                        continue
                    l, t0, r, b0 = bbox
                    draw.rectangle((l * zoom, t0 * zoom, r * zoom, b0 * zoom), outline=(255, 0, 0, 220), width=2)

        company = "unknown"
        if isinstance(export_dict, dict):
            company = str(export_dict.get("company") or "unknown")

        out_dir = overlays_dir / company
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"page_{page_num:03d}.png"
        img.save(out_path)

    except Exception:
        # Never fail a benchmark run due to overlay rendering.
        return


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

    # Page number used to extract from the source PDF.
    # Normally equals page_number, but can differ when a PDF has an extra cover page.
    pdf_page_number: Optional[int] = None
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

    # Hybrid Docling routing diagnostics (None for other engines)
    hybrid_ocr_stats: Optional[Dict[str, Any]] = None

    # Debug snapshot of OCR cells after hybrid updates (only for hybrid_docling)
    ocr_cells_debug: Optional[Dict[str, Any]] = None



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

    # HybridDocling tuning knobs (None for other engines)
    hybrid_confidence_threshold: Optional[float] = None
    hybrid_number_confidence_threshold: Optional[float] = None
    
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

    # Composite score for threshold sweeps
    quality_score: float = 0.0
    
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
        pdf_dir: Optional[str] = None,
        ocr_engine: str = "docling_pdf",  # docling_pdf | hybrid_docling | marker
        dpi: int = 300,
        marker_use_llm: bool = False,  # Use LLM for Marker post-processing (requires OpenRouter API key)
        table_only: bool = False,  # Only benchmark pages with financial tables
        financial_only: bool = False,  # Only benchmark financial statement/notes pages (Balance Sheet, IS, CF, Notes)
        # How to interpret --table-only (based on GT, not model output).
        # - heuristic: uses VnPdfSample.is_table_page
        # - gt_pipe_any: require GT parses as a pipe table (>=5 rows, >=3 cols)
        # - gt_pipe_strict: require GT pipe table with >=min_gt_rows and >=min_gt_cols
        table_only_mode: str = "heuristic",
        min_gt_rows: int = 5,
        min_gt_cols: int = 3,
        min_gt_numbers: int = 20,
        hybrid_confidence_threshold: float = 0.7,
        hybrid_number_confidence_threshold: float = 0.85,
        save_ocr_outputs: bool = False,  # Save OCR text outputs for debugging
        ocr_outputs_dir: Optional[str] = None,  # Directory to save OCR outputs
        export_overlays: bool = False,  # Export PNG overlays (hybrid only)
        overlays_dir: Optional[str] = None,
        export_hybrid_diffs: bool = False,  # Persist Surya update diffs (hybrid only)
        hybrid_diffs_dir: Optional[str] = None,
        minimal_json: bool = False,
        # Per-company PDF page offsets applied as:
        #   pdf_page = dataset_page + offset
        # Example: if the dataset page_number is 1-based after removing a cover page,
        # and your local PDF still includes that cover page, you may need offset=+1.
        page_offsets: Optional[Dict[str, int]] = None,
        # Restrict evaluation to specific dataset page numbers (applies to every company).
        pages: Optional[List[int]] = None,
    ):
        self.pdf_dir = Path(pdf_dir) if pdf_dir else PDF_SAMPLES_DIR
        self.ocr_engine = ocr_engine
        self.dpi = dpi
        self.marker_use_llm = marker_use_llm
        self.table_only = table_only
        self.table_only_mode = str(table_only_mode or "heuristic")
        self.min_gt_rows = int(min_gt_rows)
        self.min_gt_cols = int(min_gt_cols)
        self.min_gt_numbers = int(min_gt_numbers)
        self.financial_only = financial_only
        self.hybrid_confidence_threshold = hybrid_confidence_threshold
        self.hybrid_number_confidence_threshold = hybrid_number_confidence_threshold
        self.save_ocr_outputs = save_ocr_outputs
        self.ocr_outputs_dir = Path(ocr_outputs_dir) if ocr_outputs_dir else Path("results/ocr_outputs")
        self.export_overlays = bool(export_overlays)
        self.overlays_dir = Path(overlays_dir) if overlays_dir else Path("results/overlays")
        self.export_hybrid_diffs = bool(export_hybrid_diffs)
        self.hybrid_diffs_dir = Path(hybrid_diffs_dir) if hybrid_diffs_dir else Path("results/hybrid_diffs")
        self.minimal_json = bool(minimal_json)
        self.page_offsets = dict(page_offsets or {})
        self.pages = list(pages) if pages else None
        self._dataset = None
        self._gt_by_company = None
        self._marker_service = None
        self._docling_service = None
        self._hybrid_service = None
        self._prefer_cuda = None

    def _gt_passes_table_only(self, sample: VnPdfSample) -> bool:
        if not self.table_only:
            return True

        mode = (self.table_only_mode or "heuristic").strip().lower()
        if mode == "heuristic":
            return bool(sample.is_table_page)

        gt_raw = sample.text or ""
        grid = parse_pipe_table_to_grid(gt_raw)
        if not grid:
            return False

        rows = len(grid)
        cols = max((len(r) for r in grid), default=0)

        if mode == "gt_pipe_any":
            if rows < 2 or cols < 2:
                return False
        elif mode == "gt_pipe_strict":
            if rows < int(self.min_gt_rows) or cols < int(self.min_gt_cols):
                return False
        else:
            # Unknown mode => be safe and behave like heuristic.
            return bool(sample.is_table_page)

        if int(self.min_gt_numbers) > 0:
            if count_numbers_in_text(gt_raw) < int(self.min_gt_numbers):
                return False

        return True
    
    def _get_preferred_docling_device(self):
        """Choose a stable accelerator for Docling pipelines."""
        if self._prefer_cuda is not None:
            return self._prefer_cuda

        try:
            import torch

            self._prefer_cuda = bool(torch.cuda.is_available())
        except Exception:
            self._prefer_cuda = False

        return self._prefer_cuda

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
                if not self._gt_passes_table_only(sample):
                    continue

                # Filter to financial statement tables/notes only if requested.
                # This uses GT text to define the evaluation slice (not strategy logic).
                if self.financial_only and not looks_like_financial_statement(sample.text):
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
        """Extract a specific page from PDF as high-resolution image."""
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
        """
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
            from docling.datamodel.accelerator_options import AcceleratorDevice
            from services.ocr.hybrid_pdf_pipeline import HybridPdfPipeline
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options.device = (
                AcceleratorDevice.CUDA if self._get_preferred_docling_device() else AcceleratorDevice.CPU
            )
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            # Docling table structure options vary across versions; keep defaults.
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
        """Convert a single PDF page with Docling and return (markdown, export_dict).

        Note: when using our HybridPdfPipeline, the pipeline instance is cached inside
        DocumentConverter, so we can also pull routing stats from it and attach to the
        returned export_dict.
        """

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

        import os
        import tempfile
        import time as _time
        import uuid

        # Temp file must be unique across concurrent benchmark processes.
        # time_ns() alone can collide on Windows when multiple processes start together.
        tmp_path = (
            Path(tempfile.gettempdir())
            / f"docling_single_{page_num}_{os.getpid()}_{_time.time_ns()}_{uuid.uuid4().hex}.pdf"
        )
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)

        # Best-effort cleanup in case a prior crash left a file behind.
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

        new_doc.save(str(tmp_path))
        new_doc.close()
        doc.close()

        try:
            result = converter.convert(str(tmp_path))
            export_dict = result.document.export_to_dict()
            if not isinstance(export_dict, dict):
                export_dict = {}
            export_dict["company"] = str(pdf_path.stem).split("_")[0]
            export_dict["page_number"] = int(page_num)


            # For hybrid validation: also export a table grid text built from export_to_dict.
            # This makes it easy to check whether Surya-updated cell text flows into the
            # exported table structures used for scoring.
            try:
                grid = extract_docling_tables_grid(export_dict)
                export_dict["_grid_text_debug"] = grid_to_canonical_text(grid) if grid else ""
            except Exception:
                export_dict["_grid_text_debug"] = ""

            # Export to markdown for evaluation/debugging.
            md_text = result.document.export_to_markdown()

            try:
                pipeline = None
                pipeline_get = getattr(converter, "_get_pipeline", None)
                if callable(pipeline_get):
                    pipeline = pipeline_get(InputFormat.PDF)

                model = getattr(pipeline, "_hybrid_ocr_model", None) if pipeline is not None else None
                if model is None:
                    return md_text, export_dict

                stats_get = getattr(model, "get_stats", None)
                if callable(stats_get):
                    try:
                        export_dict["hybrid_ocr_stats"] = stats_get()
                    except Exception:
                        pass

                small_snap_get = getattr(model, "get_debug_snapshot", None)
                small_snap = None
                if callable(small_snap_get):
                    try:
                        small_snap = small_snap_get()
                    except Exception:
                        small_snap = None

                if isinstance(small_snap, dict):
                    export_dict["ocr_cells_debug"] = small_snap

                diffs = None
                diffs_get = getattr(model, "get_update_diffs", None)
                if callable(diffs_get):
                    try:
                        diffs = diffs_get()
                    except Exception:
                        diffs = None

                # Optional: persist detailed Surya update diffs to sidecar JSON.
                # This keeps the main results JSON small (especially with --minimal-json)
                # while enabling targeted debugging of hybrid regressions.
                try:
                    if bool(getattr(self, "export_hybrid_diffs", False)):
                        if isinstance(diffs, list) and diffs:
                            out_root = Path(getattr(self, "hybrid_diffs_dir", Path("results/hybrid_diffs")))
                            out_dir = out_root / str(export_dict.get("company") or "unknown")
                            out_dir.mkdir(parents=True, exist_ok=True)
                            out_path = out_dir / f"page_{int(page_num):03d}.json"
                            payload = {
                                "company": export_dict.get("company"),
                                "page_number": int(page_num),
                                "hybrid_ocr_stats": export_dict.get("hybrid_ocr_stats"),
                                "update_diffs": diffs,
                            }
                            with open(out_path, "w", encoding="utf-8") as f:
                                json.dump(payload, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

                # Use a targeted rewrite that only touches cells overlapped by accepted Surya updates.
                try:
                    do_rewrite = False
                    stats_obj = export_dict.get("hybrid_ocr_stats")
                    if isinstance(stats_obj, dict):
                        do_rewrite = int(stats_obj.get("surya_cells_updated", 0) or 0) > 0

                    if do_rewrite and isinstance(diffs, list):
                        rewrite_docling_table_grid_text_from_surya_updates(
                            export_dict,
                            update_diffs=diffs,
                        )
                except Exception:
                    pass
            except Exception:
                pass

            # Optional debug overlay export (flag-only).
            try:
                if self.export_overlays:
                    if isinstance(export_dict, dict):
                        _draw_overlays_for_hybrid_docling(
                            pdf_path=pdf_path,
                            page_num=page_num,
                            dpi=self.dpi,
                            export_dict=export_dict,
                            overlays_dir=self.overlays_dir,
                        )
            except Exception:
                pass

            return md_text, export_dict

        finally:
            for _ in range(3):
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                    break
                except PermissionError:
                    time.sleep(0.1)
    
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

            page_offset = int(self.page_offsets.get(company, 0))
            pdf_page_num = int(page_num + page_offset)

            # Run OCR based on engine
            if self.ocr_engine == "marker":
                logger.info(f"  Processing page {page_num} with Marker...")
                ocr_text_raw = self.ocr_pdf_page_with_marker(pdf_path, pdf_page_num)
                ocr_doc_dict = None
            elif self.ocr_engine == "docling_pdf":
                # Docling PDF pipeline baseline
                logger.info(f"  Processing page {page_num} with Docling PDF pipeline...")
                from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
                from docling.datamodel.accelerator_options import AcceleratorDevice
                from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

                pipeline_options = PdfPipelineOptions()
                pipeline_options.accelerator_options.device = (
                    AcceleratorDevice.CUDA if self._get_preferred_docling_device() else AcceleratorDevice.CPU
                )
                pipeline_options.do_ocr = True
                pipeline_options.do_table_structure = True
                # Docling table structure options vary across versions; keep defaults.
                pipeline_options.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, lang=["vie"])

                ocr_text_raw, ocr_doc_dict = self._convert_pdf_page_with_docling_pipeline(
                    pdf_path=pdf_path,
                    page_num=pdf_page_num,
                    pipeline_cls=StandardPdfPipeline,
                    pipeline_options=pipeline_options,
                )
            elif self.ocr_engine == "hybrid_docling":
                # Hybrid Docling: Full Docling pipeline with HybridOcrModel
                logger.info(f"  Processing page {page_num} with Hybrid Docling...")
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.datamodel.accelerator_options import AcceleratorDevice
                from services.ocr.hybrid_pdf_pipeline import HybridPdfPipeline
                from services.ocr.hybrid_ocr_model import HybridOcrOptions

                pipeline_options = PdfPipelineOptions()
                pipeline_options.accelerator_options.device = (
                    AcceleratorDevice.CUDA if self._get_preferred_docling_device() else AcceleratorDevice.CPU
                )
                pipeline_options.do_ocr = True
                pipeline_options.do_table_structure = True
                # Docling table structure options vary across versions; keep defaults.
                pipeline_options.ocr_options = HybridOcrOptions(
                    force_full_page_ocr=True,
                    lang=["vie"],
                    confidence_threshold=float(self.hybrid_confidence_threshold),
                    number_confidence_threshold=float(self.hybrid_number_confidence_threshold),
                    log_routing_stats=True,
                )

                ocr_text_raw, ocr_doc_dict = self._convert_pdf_page_with_docling_pipeline(
                    pdf_path=pdf_path,
                    page_num=pdf_page_num,
                    pipeline_cls=HybridPdfPipeline,
                    pipeline_options=pipeline_options,
                )
            else:
                return PageResult(
                    company=company,
                    page_number=page_num,
                    pdf_page_number=pdf_page_num,
                    format_agnostic_cer=1.0,
                    content_word_recall=0.0,
                    number_f1=0.0,
                    success=False,
                    error=f"Unsupported OCR engine: {self.ocr_engine}",
                    gt_text=gt_eval_text,
                    gt_text_raw=gt_text_raw,
                )

            # If OCR produced nothing, treat as a failure (do not score as 0 and mark success).
            if not ocr_text_raw or not ocr_text_raw.strip():
                return PageResult(
                    company=company,
                    page_number=page_num,
                    pdf_page_number=pdf_page_num,
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

            if self.ocr_engine in {"docling_pdf", "hybrid_docling"}:
                doc_dict = ocr_doc_dict if isinstance(ocr_doc_dict, dict) else {}
                ocr_grid_str = extract_docling_tables_grid(doc_dict)
                docling_tables_payload = doc_dict.get("tables")

                if ocr_grid_str:
                    extraction_mode = "docling_grid"
                    ocr_eval_text = grid_to_canonical_text(ocr_grid_str)
                else:
                    # Fallback: parse markdown-ish pipe tables from Docling markdown.
                    ocr_grid_md = parse_pipe_table_to_grid(ocr_text_raw)
                    if ocr_grid_md:
                        ocr_eval_text = grid_to_canonical_text(ocr_grid_md)
                        extraction_mode = "docling_pipe_table"
                    else:
                        # Next, try extracting numbered/section rows (common in GT pipe tables).
                        sect = extract_sectioned_rows(ocr_text_raw)
                        if sect.strip():
                            ocr_eval_text = sect
                            extraction_mode = "docling_sectioned_rows"
                        else:
                            ocr_table_text = extract_table_content_robust(ocr_text_raw)
                            if ocr_table_text.strip():
                                ocr_eval_text = ocr_table_text
                                extraction_mode = "docling_aligned_lines"
                            else:
                                return PageResult(
                                    company=company,
                                    page_number=page_num,
                                    pdf_page_number=pdf_page_num,
                                    format_agnostic_cer=1.0,
                                    content_word_recall=0.0,
                                    number_f1=0.0,
                                    success=False,
                                    error="No table-like content extracted",
                                    ocr_text_raw=ocr_text_raw,
                                    gt_text=gt_eval_text,
                                    gt_text_raw=gt_text_raw,
                                    extraction_mode="docling_no_tables",
                                    docling_tables=docling_tables_payload,
                                )
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
                        if self.ocr_engine == "marker":
                            fallback = extract_table_content_fallback(ocr_text_raw)
                            if fallback.strip():
                                ocr_eval_text = fallback
                                extraction_mode = "raw_fallback"
                            else:
                                return PageResult(
                                    company=company,
                                    page_number=page_num,
                                    pdf_page_number=pdf_page_num,
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
                        else:
                            # Fail-closed for other engines.
                            return PageResult(
                                company=company,
                                page_number=page_num,
                                pdf_page_number=pdf_page_num,
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

            hybrid_stats = None
            if isinstance(ocr_doc_dict, dict):
                hybrid_stats = ocr_doc_dict.get("hybrid_ocr_stats")

            
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
                pdf_page_number=pdf_page_num,
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
                # Store scoring text for correct aggregated metrics.
                # Large payloads are pruned at JSON export time when --minimal-json is set.
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

                # Hybrid routing diagnostics (only present for hybrid_docling)
                hybrid_ocr_stats=hybrid_stats,

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
                error=str(e),
            )
    
    def benchmark_company(self, company: str, max_pages: Optional[int] = None) -> Optional[CompanyResult]:
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
        
        # Build evaluation slice (dataset page numbers)
        sorted_pages = sorted(gt_pages.items())

        if self.pages is not None:
            allowed = set(int(p) for p in self.pages)
            sorted_pages = [(p, s) for (p, s) in sorted_pages if int(p) in allowed]

        # Apply page limit BEFORE processing
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
            
        # Clear CUDA cache after each page to prevent OOM from memory fragmentation.
        # Hybrid/Docling pipelines are sensitive to host/GPU memory pressure, so we
        # only do this for Marker which loads large models.
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
    
    def run(self, companies: Optional[List[str]] = None, max_pages_per_company: Optional[int] = None) -> PageLevelBenchmarkResult:
        """
        Run benchmark on specified companies.
        
        Args:
            companies: List of company codes, or None for all
            max_pages_per_company: Limit pages per company for quick testing
        """
        logger.info("Starting Page-Level OCR Benchmark")
        logger.info(f"DPI: {self.dpi}, OCR Engine: {self.ocr_engine}")
        if self.ocr_engine == "hybrid_docling":
            logger.info(
                f"Hybrid thresholds: conf={self.hybrid_confidence_threshold}, num={self.hybrid_number_confidence_threshold}"
            )
        
        if companies is None:
            companies = COMPANY_CODES
        
        result = PageLevelBenchmarkResult(
            timestamp=datetime.now().isoformat(),
            ocr_engine=self.ocr_engine,
            dpi=self.dpi,
            hybrid_confidence_threshold=(
                float(self.hybrid_confidence_threshold) if self.ocr_engine == "hybrid_docling" else None
            ),
            hybrid_number_confidence_threshold=(
                float(self.hybrid_number_confidence_threshold) if self.ocr_engine == "hybrid_docling" else None
            ),
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

        # Composite score for tuning: 0.5 * WordRecall + 0.5 * NumF1
        # NumF1 is averaged only on pages with numbers in GT (by design).
        result.quality_score = 0.5 * result.overall_avg_content_word_recall + 0.5 * result.overall_avg_number_f1
        
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
    
    def save_results(self, result: PageLevelBenchmarkResult, output_path: str | Path) -> None:
        """Save results to JSON."""
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        payload = result.to_dict()

        if bool(getattr(self, "minimal_json", False)):
            # Remove large per-page payload fields (keep only metrics/timing/errors/diagnostics).
            for company in payload.get("company_results", []) or []:
                for page in company.get("page_results", []) or []:
                    for k in (
                        "ocr_text",
                        "ocr_text_raw",
                        "gt_text",
                        "gt_text_raw",
                        "docling_tables",
                        "ocr_cells_debug",
                    ):
                        page.pop(k, None)
        
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path_obj}")


def main():
    """Run benchmark from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run page-level OCR benchmark")
    parser.add_argument("--pdf-dir", type=str, default=None, help="Directory containing PDF files")
    parser.add_argument("--companies", nargs="*", help="Company codes to benchmark")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages per company")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for page extraction (default: 300, try 400-600 for scanned PDFs)")
    parser.add_argument(
        "--engine",
        type=str,
        default="docling_pdf",
        choices=["docling_pdf", "hybrid_docling", "marker"],
        help=(
            "OCR engine for benchmark: docling_pdf vs hybrid_docling vs marker."
        ),
    )
    parser.add_argument("--marker-llm", action="store_true", help="Use LLM with Marker (requires OPENROUTER_API_KEY)")
    parser.add_argument("--table-only", action="store_true", help="Only benchmark pages with financial tables")
    parser.add_argument(
        "--table-only-mode",
        type=str,
        default="heuristic",
        choices=["heuristic", "gt_pipe_any", "gt_pipe_strict"],
        help=(
            "Interpretation of --table-only based on GT. "
            "heuristic uses GT heuristics; gt_pipe_* require GT pipe table parsing."
        ),
    )
    parser.add_argument(
        "--min-gt-rows",
        type=int,
        default=3,
        help="With --table-only-mode gt_pipe_strict: minimum GT table rows.",
    )
    parser.add_argument(
        "--min-gt-cols",
        type=int,
        default=3,
        help="With --table-only-mode gt_pipe_strict: minimum GT table columns.",
    )
    parser.add_argument(
        "--min-gt-numbers",
        type=int,
        default=0,
        help="With --table-only-mode gt_pipe_*: minimum number tokens in GT.",
    )
    parser.add_argument(
        "--financial-only",
        action="store_true",
        help="Only benchmark financial statement/notes pages (Balance Sheet, Income Statement, Cash Flow, Notes) based on GT keywords",
    )
    parser.add_argument("--output", type=str, default=" page_level_benchmark.json")
    parser.add_argument(
        "--minimal-json",
        action="store_true",
        help="Write compact results JSON without per-page OCR/GT text payloads (off by default).",
    )
    parser.add_argument("--save-outputs", action="store_true", help="Save OCR outputs for debugging/analysis")
    parser.add_argument("--outputs-dir", type=str, default="results/ocr_outputs", help="Directory to save OCR outputs")

    parser.add_argument(
        "--pages",
        nargs="*",
        type=int,
        default=None,
        help="Restrict evaluation to these dataset page numbers (e.g., --pages 2 3 4)",
    )

    parser.add_argument(
        "--page-offsets",
        nargs="*",
        type=str,
        default=None,
        help="Per-company PDF page offset(s) as CODE:INT (pdf_page = dataset_page + INT). Example: --page-offsets TCB:1",
    )

    parser.add_argument(
        "--export-overlays",
        action="store_true",
        help="HybridDocling: export per-page PNG overlays (off by default).",
    )
    parser.add_argument(
        "--overlays-dir",
        type=str,
        default="results/overlays",
        help="Directory to save overlay PNGs when --export-overlays is set.",
    )

    parser.add_argument(
        "--export-hybrid-diffs",
        action="store_true",
        help="HybridDocling: persist per-page Surya update diffs to JSON sidecar files (off by default).",
    )
    parser.add_argument(
        "--hybrid-diffs-dir",
        type=str,
        default="results/hybrid_diffs",
        help="Directory to save diff JSONs when --export-hybrid-diffs is set.",
    )


    parser.add_argument(
        "--hybrid-threshold",
        type=float,
        default=0.7,
        help="HybridDocling: confidence threshold for routing a cell to Surya (higher => more Surya).",
    )
    parser.add_argument(
        "--hybrid-number-threshold",
        type=float,
        default=0.85,
        help="HybridDocling: confidence threshold for number-containing cells (higher => more Surya).",
    )

    
    args = parser.parse_args()
    
    page_offsets: Dict[str, int] = {}
    if args.page_offsets:
        for item in args.page_offsets:
            if not isinstance(item, str) or ":" not in item:
                raise SystemExit(f"Invalid --page-offsets entry: {item!r} (expected CODE:INT)")
            code, raw = item.split(":", 1)
            code = code.strip().upper()
            try:
                offset = int(raw.strip())
            except ValueError as e:
                raise SystemExit(f"Invalid offset for {code}: {raw!r} (expected int)") from e
            page_offsets[code] = offset

    benchmark = PageLevelBenchmark(
        pdf_dir=args.pdf_dir,
        ocr_engine=args.engine,
        dpi=args.dpi,
        marker_use_llm=args.marker_llm,
        table_only=args.table_only,
        table_only_mode=args.table_only_mode,
        min_gt_rows=int(args.min_gt_rows),
        min_gt_cols=int(args.min_gt_cols),
        min_gt_numbers=int(args.min_gt_numbers),
        financial_only=args.financial_only,
        hybrid_confidence_threshold=args.hybrid_threshold,
        hybrid_number_confidence_threshold=args.hybrid_number_threshold,
        save_ocr_outputs=args.save_outputs,
        ocr_outputs_dir=args.outputs_dir,
        export_overlays=args.export_overlays,
        overlays_dir=args.overlays_dir,
        export_hybrid_diffs=args.export_hybrid_diffs,
        hybrid_diffs_dir=args.hybrid_diffs_dir,
        minimal_json=args.minimal_json,
        page_offsets=page_offsets,
        pages=args.pages,
    )
    result = benchmark.run(companies=args.companies, max_pages_per_company=args.max_pages)
    benchmark.save_results(result, args.output)


if __name__ == "__main__":
    main()
