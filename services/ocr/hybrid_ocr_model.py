"""
HybridOcrModel - Confidence-Gated Dual-Engine OCR for Docling

This module provides a drop-in replacement for Docling's TesseractOcrCliModel
that intelligently routes low-confidence OCR cells to Surya for re-OCR.

Key Innovation:
- Uses Tesseract's per-word confidence scores to identify unreliable OCR
- Routes low-confidence cells (especially numbers) to Surya for higher accuracy
- Preserves bounding boxes for correct table structure detection
"""

import logging
import re
import gc
import threading
import importlib
import unicodedata
from typing import TYPE_CHECKING

from docling_core.types.doc.base import BoundingBox
from pathlib import Path
from typing import Any, ClassVar, Iterable, List, Literal, Optional, Sequence, Type, cast
from docling_core.types.doc.page import TextCell
from PIL import Image
from pydantic import ConfigDict

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import TesseractCliOcrOptions


def _import_docling_symbol(module_names: Sequence[str], symbol: str) -> Any:
    tried: list[str] = []
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            tried.append(mod_name)
            continue
        if hasattr(mod, symbol):
            return getattr(mod, symbol)
    try:
        import pkgutil
        import docling

        for m in pkgutil.walk_packages(docling.__path__, docling.__name__ + "."):
            name = m.name
            if "tesseract" not in name:
                continue
            if name in tried:
                continue
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue
            if hasattr(mod, symbol):
                return getattr(mod, symbol)
    except Exception:
        pass

    raise ModuleNotFoundError(
        f"Cannot import Docling symbol {symbol!r}. Tried modules: {', '.join(module_names)}"
    )


TesseractOcrCliModel: Type[Any] = cast(
    Type[Any],
    _import_docling_symbol(
        (
            "docling.models.tesseract_ocr_cli_model",
            "docling.models.ocr.tesseract_ocr_cli_model",
            "docling.models.ocr_cli.tesseract_ocr_cli_model",
            "docling.models.stages.ocr.tesseract_ocr_cli_model",
        ),
        "TesseractOcrCliModel",
    ),
)


def _parse_orientation_compat(df_osd: Any) -> int:
    parse_fn = _import_docling_symbol(
        (
            "docling.models.tesseract_ocr_cli_model",
            "docling.models.ocr.tesseract_ocr_cli_model",
            "docling.models.ocr_cli.tesseract_ocr_cli_model",
            "docling.models.stages.ocr.tesseract_ocr_cli_model",
        ),
        "_parse_orientation",
    )
    return int(parse_fn(df_osd))
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

_SURYA_LOCK = threading.Lock()
_SURYA_SHARED: dict[str, tuple[Any, Any]] = {}

if TYPE_CHECKING:
    from typing import Tuple as _TupleBoolBoolFloat


_HEADER_NUMERIC_RE = re.compile(r"(?ix)^(?:q[1-4][./-]?\d{4}|note\s*\d+(?:\.\d+)*|ghi\s*chu\s*\d+(?:\.\d+)*)$")


def _normalize_for_numeric_likeness(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    trans = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})
    return s.translate(trans)


_NUMERIC_TOKEN_RE = re.compile(r"(?ix)(?:\(\s*)?-?\s*\d[\d\s.,/%đvndusdEUR]*\d\s*\)?")


def _strip_accents_basic(s: str) -> str:
    s = s or ""
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def numeric_likeness(text: str) -> tuple[bool, bool, float]:
    """Return (is_numeric_like, is_header_numeric, score).

    Designed for Vietnamese financial tables where separators and note markers are common.
    """
    raw = (text or "").strip()
    if not raw:
        return False, False, 0.0

    s = _normalize_for_numeric_likeness(raw)
    s2 = re.sub(r"\s+", " ", s).strip().lower()

    is_header_numeric = bool(_HEADER_NUMERIC_RE.match(s2))

    digit_count = sum(1 for c in s if c.isdigit())
    alpha_count = sum(1 for c in s if c.isalpha())
    length = max(len(s), 1)

    digit_ratio = digit_count / length

    has_currency = any(x in s2 for x in ("vnd", "vnđ", "usd", "eur", "đ"))
    has_percent = "%" in s2
    has_paren_neg = raw.startswith("(") and raw.endswith(")")
    has_sep = "," in s2 or "." in s2

    score = 0.0
    score += min(digit_ratio * 1.5, 1.0)
    if has_sep and digit_count >= 2:
        score += 0.25
    if has_currency:
        score += 0.25
    if has_percent:
        score += 0.15
    if has_paren_neg:
        score += 0.15
    if alpha_count == 0 and digit_count > 0:
        score += 0.10

    score = max(0.0, min(score, 1.0))
    is_numeric_like = score >= 0.45

    return is_numeric_like, is_header_numeric, score


def _alnum_count(s: str) -> int:
    return sum(1 for ch in (s or "") if ch.isalnum())


def _charclass_signature(s: str) -> str:
    s = (s or "")
    digs = sum(ch.isdigit() for ch in s)
    alphas = sum(ch.isalpha() for ch in s)
    other = max(0, len(s) - digs - alphas)

    parts = []
    if digs:
        parts.append("D")
    if alphas:
        parts.append("A")
    if other:
        parts.append("O")
    return "".join(parts) or "E"


def _lcs_len(a: str, b: str) -> int:
    # DP LCS length for small strings (cells are small).
    a = a or ""
    b = b or ""
    if not a or not b:
        return 0

    if len(a) > 256 or len(b) > 256:
        return 0

    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0]
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _is_plausible_surya_replacement(
    *,
    baseline: str,
    candidate: str,
    max_len_ratio: float,
    max_abs_len: int,
    require_same_charclass: bool = True,
    min_normalized_lcs_ratio: float = 0.15,
) -> bool:
    b = (baseline or "").strip()
    c = (candidate or "").strip()

    if not c:
        return False

    if len(c) > max_abs_len:
        return False

    if b:
        if len(c) > int(max_len_ratio * max(1, len(b))):
            return False

    if _alnum_count(c) == 0:
        return False

    if "|" in c:
        return False

    for ch in c:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            # Unnamed alpha codepoints are extremely unlikely here; reject.
            return False
        if "LATIN" not in name:
            return False

    if b:
        nb = re.sub(r"\s+", " ", b).strip().lower()
        nc = re.sub(r"\s+", " ", c).strip().lower()
        if nb and nc:
            lcs = _lcs_len(nb, nc)
            denom = max(1, min(len(nb), len(nc)))
            if (lcs / denom) < float(min_normalized_lcs_ratio):
                return False

            base_num_like, base_header_num, _ = numeric_likeness(nb)
            if require_same_charclass and (not base_num_like) and (_charclass_signature(nb) != _charclass_signature(nc)):
                return False

    base_num_like, base_header_num, _ = numeric_likeness(b)
    if base_num_like and not base_header_num:
        cand_num_like, cand_header_num, _ = numeric_likeness(c)
        if not cand_num_like or cand_header_num:
            return False

    return True


class HybridOcrOptions(TesseractCliOcrOptions):
    """Extended options for Hybrid OCR with confidence-gated routing."""

    kind: ClassVar[Literal["tesseract"]] = "tesseract"

    # Confidence thresholds (0.0 - 1.0)
    confidence_threshold: float = 0.7
    number_confidence_threshold: float = 0.85

    # If True, route numeric-like cells to Surya regardless of confidence.
    # Default False so threshold sweeps are meaningful and faster.
    force_surya_for_numbers: bool = False

    # If True, route ALL cells inside inferred table regions.
    # Default False: only route low-confidence (mostly numeric) cells for speed.
    force_surya_in_table_regions: bool = False

    # Surya batch size for re-OCR
    surya_batch_size: int = 32

    # Hardening knobs
    max_replacement_len_ratio: float = 3.0
    max_replacement_abs_len: int = 128

    # Match-back safety knobs
    # NOTE: `require_same_charclass=False` by default because the baseline can be OCR garbage (e.g., '0y0tD024') while Surya produces valid digits.
    require_same_charclass: bool = False
    min_normalized_lcs_ratio: float = 0.15

    # Routing policy
    route_table_only: bool = True

    # Safer-by-default policy: only route numeric-like cells unless confidence is extremely low.
    route_numeric_only: bool = True
    non_numeric_confidence_threshold: float = 0.35

    # Additional safety cap: even if thresholds are high, do not route cells with relatively high Tesseract confidence.
    # This prevents Surya from overwriting already-correct numbers.
    numeric_route_confidence_cap: float = 0.95

    # When False, we only apply Surya replacements to numeric-like cells.
    # This protects Vietnamese text recall and avoids wasting compute.
    update_non_numeric: bool = True

    # If True, only apply non-numeric updates for cells inferred to be inside a table region.
    # This keeps hybrid behavior focused on the benchmark target (financial tables) and
    # reduces the risk of overwriting running text.
    update_non_numeric_table_only: bool = True

    # Routing/acceptance knobs for Vietnamese text inside tables.
    # Only consider re-OCR for non-numeric table cells below this confidence.
    table_text_confidence_threshold: float = 0.50
    # Acceptance gate: require strong overlap in accent-stripped form.
    table_text_min_accent_stripped_lcs_ratio: float = 0.65

    # Numeric acceptance hardening: only accept numeric replacements when token-level
    # evidence suggests the baseline is unreliable.
    accept_numeric_only_if_low_token_conf: bool = True

    # Separate acceptance threshold for numeric token confidence.
    # This should usually be <= number_confidence_threshold, otherwise we may overwrite
    # correct numbers that Tesseract scored moderately-high.
    numeric_accept_token_confidence_threshold: float = 0.75

    # Logging
    log_routing_stats: bool = True
    
    model_config = ConfigDict(
        extra="forbid",
    )


def _intersect_area(a: BoundingBox, b: BoundingBox) -> float:
    l = max(float(a.l), float(b.l))
    t = max(float(a.t), float(b.t))
    r = min(float(a.r), float(b.r))
    btm = min(float(a.b), float(b.b))
    if r <= l or btm <= t:
        return 0.0
    return (r - l) * (btm - t)


def _union_box(boxes: List[BoundingBox]) -> Optional[BoundingBox]:
    if not boxes:
        return None
    left = min(float(b.l) for b in boxes)
    top = min(float(b.t) for b in boxes)
    right = max(float(b.r) for b in boxes)
    bottom = max(float(b.b) for b in boxes)
    return BoundingBox(
        l=left,
        t=top,
        r=right,
        b=bottom,
        coord_origin=boxes[0].coord_origin,
    )


def _infer_table_boxes_from_tsv(df_result) -> List[BoundingBox]:
    """Infer table-like horizontal bands from Tesseract TSV words.

    Heuristic:
    - group word boxes into rows by y-center proximity
    - mark rows that look table-like (multiple columns + numeric density)
    - merge consecutive table-like rows into larger bboxes
    """
    try:
        if df_result is None or getattr(df_result, "empty", False):
            return []

        words: List[dict[str, Any]] = []
        for _ix, row in df_result.iterrows():
            txt = str(row.get("text") or "").strip()
            if not txt:
                continue
            if int(float(row.get("word_num") or 0)) <= 0:
                continue

            left = float(row.get("left") or 0.0)
            top = float(row.get("top") or 0.0)
            width = float(row.get("width") or 0.0)
            height = float(row.get("height") or 0.0)
            if width <= 0 or height <= 0:
                continue

            from docling_core.types.doc.base import CoordOrigin

            # Use TOPLEFT consistently in this OCR stage
            bbox = BoundingBox(
                l=left,
                t=top,
                r=left + width,
                b=top + height,
                coord_origin=CoordOrigin.TOPLEFT,
            )

            is_num_like, is_header_num, _ = numeric_likeness(txt)
            words.append(
                {
                    "bbox": bbox,
                    "x": (left + left + width) * 0.5,
                    "y": (top + top + height) * 0.5,
                    "h": height,
                    "text": txt,
                    "is_num_like": bool(is_num_like and not is_header_num),
                }
            )

        if not words:
            return []

        hs = sorted(w["h"] for w in words)
        median_h = hs[len(hs) // 2]
        y_tol = max(8.0, float(median_h) * 0.75)

        words_sorted = sorted(words, key=lambda w: w["y"])
        rows: List[List[dict[str, Any]]] = []
        cur: List[dict[str, Any]] = []
        cur_y: Optional[float] = None
        for w in words_sorted:
            if not cur:
                cur = [w]
                cur_y = w["y"]
                continue
            assert cur_y is not None
            if abs(w["y"] - cur_y) <= y_tol:
                cur.append(w)
                cur_y = (cur_y * (len(cur) - 1) + w["y"]) / len(cur)
            else:
                rows.append(cur)
                cur = [w]
                cur_y = w["y"]
        if cur:
            rows.append(cur)

        def row_is_table_like(r: List[dict[str, Any]]) -> bool:
            if len(r) < 6:
                return False
            r_sorted = sorted(r, key=lambda w: w["x"])
            xs = [w["x"] for w in r_sorted]
            if not xs:
                return False

            # Count coarse column "groups" by large x gaps
            x_gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            gap_thr = max(40.0, float(median_h) * 3.0)
            col_groups = 1 + sum(1 for g in x_gaps if g >= gap_thr)

            num_like = sum(1 for w in r if w["is_num_like"])
            num_ratio = num_like / max(1, len(r))

            # tables tend to have >=3 columns and decent numeric density
            return (col_groups >= 3 and num_ratio >= 0.15) or (col_groups >= 4)

        row_boxes: List[Optional[BoundingBox]] = []
        row_flags: List[bool] = []
        for r in rows:
            boxes = [w["bbox"] for w in r]
            row_boxes.append(_union_box(boxes))
            row_flags.append(row_is_table_like(r))

        # Merge consecutive table-like rows into bands
        bands: List[BoundingBox] = []
        band_rows: List[BoundingBox] = []
        for flag, rb in zip(row_flags, row_boxes):
            if rb is None:
                continue
            if flag:
                band_rows.append(rb)
            else:
                if band_rows:
                    u = _union_box(band_rows)
                    if u is not None:
                        bands.append(u)
                    band_rows = []
        if band_rows:
            u = _union_box(band_rows)
            if u is not None:
                bands.append(u)

        # Require at least 2 table-like rows total to consider it a table region
        if sum(1 for f in row_flags if f) < 2:
            return []

        return bands
    except Exception:
        return []


def _tesseract_word_min_conf_in_bbox(df_result, cell_bbox: BoundingBox) -> Optional[float]:
    """Min confidence among numeric-like TSV words that overlap a bbox."""
    try:
        if df_result is None or getattr(df_result, "empty", False):
            return None

        min_conf: Optional[float] = None
        for _ix, row in df_result.iterrows():
            txt = str(row.get("text") or "").strip()
            if not txt:
                continue

            # Word bbox in the *region image* coordinate system (same as we use for cell bboxes below).
            left = float(row.get("left") or 0.0)
            top = float(row.get("top") or 0.0)
            width = float(row.get("width") or 0.0)
            height = float(row.get("height") or 0.0)
            if width <= 0 or height <= 0:
                continue

            word_bbox = BoundingBox(
                l=left,
                t=top,
                r=left + width,
                b=top + height,
                coord_origin=cell_bbox.coord_origin,
            )
            if _intersect_area(word_bbox, cell_bbox) <= 0:
                continue

            conf_raw = float(row.get("conf") or 0.0)
            conf = max(0.0, min(conf_raw / 100.0, 1.0))

            is_num_like, is_header, _ = numeric_likeness(txt)
            if not is_num_like or is_header:
                continue

            if min_conf is None or conf < min_conf:
                min_conf = conf

        return min_conf
    except Exception:
        return None


def _tesseract_word_min_conf_for_text(df_result) -> Optional[float]:
    """Compute min word confidence for numeric-like tokens in a cell.

    Returns:
      - min confidence in [0,1] across numeric-like words, if any
      - None if no numeric-like word tokens found
    """
    try:
        if df_result is None or getattr(df_result, "empty", False):
            return None

        min_conf: Optional[float] = None
        for _ix, row in df_result.iterrows():
            txt = str(row.get("text") or "").strip()
            if not txt:
                continue
            conf_raw = float(row.get("conf") or 0.0)
            conf = max(0.0, min(conf_raw / 100.0, 1.0))

            # Exclude obvious header numerics like Q1.2025 and Note 3.1
            is_num_like, is_header, _ = numeric_likeness(txt)
            if not is_num_like or is_header:
                continue

            if min_conf is None or conf < min_conf:
                min_conf = conf

        return min_conf
    except Exception:
        return None


def _build_line_min_numeric_conf(df_result) -> dict[tuple[int, int, int], float]:
    """Return min numeric-like token confidence per TSV (block, par, line).

    Tesseract TSV contains multi-level rows. For routing, using the *minimum* numeric-like
    word confidence inside a whole cell bbox is often too pessimistic (large boxes overlap
    many words). Instead, we compute a per-line key and use that as a proxy for
    token-level evidence for that cell.
    """
    out: dict[tuple[int, int, int], float] = {}
    try:
        if df_result is None or getattr(df_result, "empty", False):
            return out

        for _ix, row in df_result.iterrows():
            try:
                word_num = int(float(row.get("word_num") or 0))
            except Exception:
                word_num = 0
            if word_num <= 0:
                continue

            txt = str(row.get("text") or "").strip()
            if not txt:
                continue

            is_num_like, is_header, _ = numeric_likeness(txt)
            if not is_num_like or is_header:
                continue

            try:
                block_num = int(float(row.get("block_num") or 0))
                par_num = int(float(row.get("par_num") or 0))
                line_num = int(float(row.get("line_num") or 0))
            except Exception:
                continue

            if block_num <= 0 or par_num <= 0 or line_num <= 0:
                continue

            conf_raw = float(row.get("conf") or 0.0)
            conf = max(0.0, min(conf_raw / 100.0, 1.0))
            key = (block_num, par_num, line_num)
            prev = out.get(key)
            if prev is None or conf < prev:
                out[key] = conf

        return out
    except Exception:
        return out


class HybridOcrModel(TesseractOcrCliModel):  # type: ignore[misc]
    def _make_debug_snapshot(self, page: Page, all_ocr_cells: List[TextCell]) -> dict[str, Any]:
        def _cell_to_obj(c: TextCell) -> dict[str, Any]:
            bb = c.rect.to_bounding_box()
            return {
                "index": int(c.index),
                "text": str(c.text or ""),
                "orig": str(getattr(c, "orig", "") or ""),
                "confidence": float(getattr(c, "confidence", 0.0) or 0.0),
                "from_ocr": bool(getattr(c, "from_ocr", False)),
            "text_cell_unit": str(getattr(c, "text_cell_unit", "")),
                "bbox": {
                    "l": float(bb.l),
                    "t": float(bb.t),
                    "r": float(bb.r),
                    "b": float(bb.b),
                    "coord_origin": "TOPLEFT",
                },
            }

        parsed = page.parsed_page
        snap: dict[str, Any] = {
            "page_number": int(getattr(page, "page_num", 0) or 0),
            "all_ocr_cells": [_cell_to_obj(c) for c in (all_ocr_cells or [])],
            "parsed_textline_cells": [_cell_to_obj(c) for c in (getattr(parsed, "textline_cells", None) or [])] if parsed is not None else [],
            "parsed_word_cells": [_cell_to_obj(c) for c in (getattr(parsed, "word_cells", None) or [])] if parsed is not None else [],
            "counts": {
                "all_ocr_cells": int(len(all_ocr_cells or [])),
                "parsed_textline_cells": int(len(getattr(parsed, "textline_cells", None) or [])) if parsed is not None else 0,
                "parsed_word_cells": int(len(getattr(parsed, "word_cells", None) or [])) if parsed is not None else 0,
            },
        }
        return snap

    def get_debug_snapshot_full(self) -> Optional[dict[str, Any]]:
        """Full snapshot for internal use (do not persist to results JSON)."""
        snap = self._last_snapshot
        return snap if isinstance(snap, dict) else None

    def get_debug_snapshot(self) -> Optional[dict[str, Any]]:
        """Small snapshot safe to persist to results JSON."""
        snap = self._last_snapshot
        if not isinstance(snap, dict):
            return None

        counts = snap.get("counts") if isinstance(snap.get("counts"), dict) else {}
        return {
            "page_number": snap.get("page_number"),
            "counts": counts,
        }

    """
    Confidence-Gated Hybrid OCR Model for Docling.
    
    Extends TesseractOcrCliModel with intelligent routing:
    - Runs Tesseract first (fast, gets bounding boxes + confidence)
    - Filters low-confidence cells
    - Re-OCRs those cells with Surya (more accurate)
    - Returns enhanced cells with original bboxes preserved
    """
    
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: HybridOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        # Initialize parent class (TesseractOcrCliModel)
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        
        # Store hybrid-specific options
        self.hybrid_options = options
        
        # Lazy-loaded Surya model
        self._surya_model = None
        self._surya_foundation = None
        
        # Routing statistics
        self._stats: dict[str, int] = {
            "total_cells": 0,
            "surya_cells": 0,
            "tesseract_cells": 0,
            "table_cells": 0,
            "non_table_cells": 0,
            "eligible_cells": 0,
            "routed_low_conf": 0,
            "routed_low_num_conf": 0,
            "skipped_no_table_clusters": 0,
            "skipped_header_numeric": 0,
            "skipped_missing_region_bbox": 0,
            "inferred_table_boxes": 0,
            "surya_cells_updated": 0,
            "surya_update_skipped_sanity": 0,
            "surya_update_skipped_count_mismatch": 0,
            "surya_update_skipped_non_numeric": 0,
            "surya_failures": 0,
        }

        self._last_update_diffs: Optional[list[dict[str, Any]]] = None
    
    @property
    def surya_model(self):
        """Lazy load Surya recognition model.
        """
        if self._surya_model is not None:
            return self._surya_model

        try:
            import torch
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            from surya.settings import settings as surya_settings
        except Exception as e:
            _log.error(f"Failed to import Surya: {e}")
            raise

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_key = device

        with _SURYA_LOCK:
            cached = _SURYA_SHARED.get(cache_key)
            if cached is None:
                _log.info("Loading Surya recognition model for hybrid OCR...")
                foundation = FoundationPredictor(
                    checkpoint=surya_settings.RECOGNITION_MODEL_CHECKPOINT,
                    device=device,
                )
                model = RecognitionPredictor(foundation)
                _SURYA_SHARED[cache_key] = (foundation, model)
                _log.info(f"Surya model loaded on {device}")
            else:
                foundation, model = cached

        self._surya_foundation = foundation
        self._surya_model = model
        return self._surya_model
    
    def _should_route_to_surya(self, cell: TextCell, *, min_numeric_token_conf: Optional[float] = None) -> bool:
        """
        Determine if a cell should be re-OCR'd by Surya.
        
        Returns True if:
        - Cell contains numbers AND confidence < number_threshold
        - OR force_surya_for_numbers is True AND cell has numbers
        - OR confidence < general_threshold
        """
        confidence = float(cell.confidence or 0.0)
        is_numeric_like, is_header_numeric, _score = numeric_likeness(cell.text)

        # Prefer true token-level evidence when available.
        # If any numeric-like token has low confidence, route aggressively.
        if min_numeric_token_conf is not None and not is_header_numeric:
            if min_numeric_token_conf < float(self.hybrid_options.number_confidence_threshold):
                return True


        # Avoid aggressive routing for section headers like "Q1.2025" / "Note 3.1"
        if is_header_numeric:
            return confidence < float(self.hybrid_options.confidence_threshold)

        # Guardrail: don't route high-confidence numeric cells.
        if is_numeric_like:
            cap = float(getattr(self.hybrid_options, "numeric_route_confidence_cap", 0.65) or 0.65)
            if confidence >= cap:
                return False

        # Safer-by-default routing: only numeric-like cells are routed unless the
        # non-numeric confidence is extremely low (acts as a last-resort rescue).
        if bool(getattr(self.hybrid_options, "route_numeric_only", True)) and (not is_numeric_like):
            return confidence < float(getattr(self.hybrid_options, "non_numeric_confidence_threshold", 0.35))

        if is_numeric_like and self.hybrid_options.force_surya_for_numbers:
            return True

        threshold = (
            self.hybrid_options.number_confidence_threshold
            if is_numeric_like
            else self.hybrid_options.confidence_threshold
        )

        return confidence < float(threshold)


    def get_update_diffs(self) -> Optional[list[dict[str, Any]]]:
        """Return last page's Surya update decisions (for debugging)."""
        diffs = self._last_update_diffs
        return diffs if isinstance(diffs, list) else None


    
    def _surya_reocr_cells(
        self, 
        page_image: Image.Image, 
        cells: List[TextCell],
        scale: float,
    ) -> None:
        """
        Re-OCR cells using Surya and update their text in-place.
        
        Args:
            page_image: High-resolution page image
            cells: List of TextCell objects to re-OCR
            scale: Scale factor applied to the image
        """
        if not cells:
            return

        self._last_update_diffs = []

        def _canon_noop(s: str) -> str:
            # Treat whitespace-only changes as no-ops.
            s2 = (s or "").strip()
            s2 = re.sub(r"\s+", " ", s2)
            return s2

        def _sanitize_surya_text(s: str) -> str:
            """Normalize common Surya artifacts (HTML-ish tags, hard line breaks)."""
            s2 = (s or "")
            if not s2:
                return ""
            # Normalize HTML-ish line breaks to spaces.
            s2 = re.sub(r"(?i)<\s*br\s*/?\s*>", " ", s2)
            # Drop other tags (we don't want markup in table cells).
            s2 = re.sub(r"<[^>]+>", " ", s2)
            # Normalize whitespace.
            s2 = s2.replace("\u00a0", " ")
            s2 = re.sub(r"\s+", " ", s2).strip()
            return s2

        def _normalize_numeric_replacement(*, baseline: str, candidate: str) -> str:
            """Canonicalize numeric strings to reduce separator noise."""
            b = (baseline or "").strip()
            c = (candidate or "").strip()
            if not c:
                return ""

            c2 = re.sub(r"\s+", " ", c).strip()

            # If the candidate is purely numeric-ish, strip all whitespace.
            if re.fullmatch(r"[0-9\s.,/%()\-+đvnusdeur]*", c2.lower()):
                c2 = re.sub(r"\s+", "", c2)

            # Unify mixed separators based on baseline style.
            if "." in b and ("," in c2) and ("." in c2):
                c2 = c2.replace(",", ".")
            elif "," in b and ("," in c2) and ("." in c2):
                c2 = c2.replace(".", ",")

            c2 = re.sub(r"\.{2,}", ".", c2)
            c2 = re.sub(r",{2,}", ",", c2)
            return c2

        def _digits_only(s: str) -> str:
            return "".join(ch for ch in (s or "") if ch.isdigit())

        def _numeric_digit_ratio_ok(baseline: str, candidate: str) -> bool:
            """Reject catastrophic truncations like 8.283.166.222 -> 789."""
            b = (baseline or "").strip()
            c = (candidate or "").strip()
            bd = _digits_only(b)
            cd = _digits_only(c)
            if len(bd) < 4 or len(cd) < 1:
                return True
            # Candidate must retain most digits; allow small drops for OCR noise.
            if len(cd) < int(0.80 * len(bd)):
                return False
            # Also prevent large spurious expansions.
            if len(cd) > int(1.25 * len(bd)):
                return False
            return True

        def _bbox_obj(cell: TextCell) -> dict[str, float]:
            bb = cell.rect.to_bounding_box()
            return {
                "l": float(bb.l),
                "t": float(bb.t),
                "r": float(bb.r),
                "b": float(bb.b),
            }

        def _is_strict_numeric_candidate(s: str) -> bool:
            s2 = (s or "").strip().lower()
            if not s2:
                return False
            if not any(ch.isdigit() for ch in s2):
                return False
            # Permit separators, percent, currency markers, parentheses.
            return bool(re.fullmatch(r"[0-9\s.,/%()\-+đvnusdeur]*", s2))
        
        try:
            import torch

            # Prepare polygons for Surya in the *region image* coordinate system.
            # Note: `page_image` here is a cropped, high-res OCR region image (possibly rotated).
            # Tesseract's TSV coordinates (left/top/width/height) are already in this coordinate
            # system. We attach those coords to each TextCell as `_region_bbox` at creation time.
            w, h = page_image.size

            cell_polys: list[tuple[TextCell, list[list[int]]]] = []
            for cell in cells:
                rb = getattr(cell, "_region_bbox", None)
                if rb is None:
                    if isinstance(self._last_update_diffs, list):
                        self._last_update_diffs.append(
                            {
                                "bbox": _bbox_obj(cell),
                                "baseline": str(cell.text or ""),
                                "candidate": "",
                                "candidate_raw": "",
                                "accepted": False,
                                "reason": "skip_missing_region_bbox",
                            }
                        )
                    continue

                try:
                    l = int(round(float(rb.l)))
                    t = int(round(float(rb.t)))
                    r = int(round(float(rb.r)))
                    b = int(round(float(rb.b)))
                except Exception:
                    if isinstance(self._last_update_diffs, list):
                        self._last_update_diffs.append(
                            {
                                "bbox": _bbox_obj(cell),
                                "baseline": str(cell.text or ""),
                                "candidate": "",
                                "candidate_raw": "",
                                "accepted": False,
                                "reason": "skip_invalid_region_bbox",
                            }
                        )
                    continue

                # Clamp to image bounds (Surya will fail if polygons are out-of-bounds).
                l = max(0, min(l, max(0, w - 1)))
                r = max(0, min(r, max(0, w - 1)))
                t = max(0, min(t, max(0, h - 1)))
                b = max(0, min(b, max(0, h - 1)))

                if r <= l or b <= t or (r - l) < 2 or (b - t) < 2:
                    if isinstance(self._last_update_diffs, list):
                        self._last_update_diffs.append(
                            {
                                "bbox": _bbox_obj(cell),
                                "baseline": str(cell.text or ""),
                                "candidate": "",
                                "candidate_raw": "",
                                "accepted": False,
                                "reason": "skip_too_small_bbox",
                            }
                        )
                    continue

                polygon = [[l, t], [r, t], [r, b], [l, b]]
                cell_polys.append((cell, polygon))

            if not cell_polys:
                return

            polygons = [p for _c, p in cell_polys]
            
            # Batch OCR with Surya
            results = self.surya_model(
                images=[page_image],
                polygons=[polygons],
                recognition_batch_size=self.hybrid_options.surya_batch_size,
            )
            
            # Update cell text with Surya results
            if results and results[0].text_lines:
                text_lines = results[0].text_lines
                if len(text_lines) != len(cell_polys):
                    self._stats["surya_update_skipped_count_mismatch"] += 1
                    _log.warning(
                        "Surya output count mismatch: got %d text lines for %d cells; skipping update",
                        len(text_lines),
                        len(cell_polys),
                    )
                    return
  
                for idx, text_line in enumerate(text_lines):
                    cell = cell_polys[idx][0]
                    original_text = str(cell.text or "")
                    new_text_raw = str(getattr(text_line, "text", "") or "")
                    new_text = _sanitize_surya_text(new_text_raw)

                    in_table_region = bool(getattr(cell, "_in_table_region", False))
                    routed_min_num_conf = getattr(cell, "_min_numeric_token_conf", None)

                    base_num_like, base_header_num, _ = numeric_likeness(original_text)
                    if base_num_like and (not base_header_num):
                        new_text = _normalize_numeric_replacement(baseline=original_text, candidate=new_text)

                    # Skip no-op updates (Surya sometimes returns the exact same string, or only differs in leading/trailing whitespace).
                    if _canon_noop(original_text) == _canon_noop(new_text):
                        if isinstance(self._last_update_diffs, list):
                            self._last_update_diffs.append(
                                {
                                    "bbox": _bbox_obj(cell),
                                    "baseline": original_text,
                                    "candidate": new_text,
                                    "candidate_raw": new_text_raw,
                                    "accepted": False,
                                    "reason": "no_change",
                                }
                            )
                        continue

                    if (
                        (not base_header_num)
                        and (not base_num_like)
                        and (not bool(getattr(self.hybrid_options, "update_non_numeric", False)))
                    ):
                        self._stats["surya_update_skipped_non_numeric"] = int(self._stats.get("surya_update_skipped_non_numeric", 0)) + 1
                        if isinstance(self._last_update_diffs, list):
                            self._last_update_diffs.append(
                                {
                                    "bbox": _bbox_obj(cell),
                                    "baseline": original_text,
                                    "candidate": new_text,
                                    "candidate_raw": new_text_raw,
                                    "accepted": False,
                                    "reason": "skip_non_numeric",
                                }
                            )
                        continue

                    # If enabled, only apply Vietnamese text updates inside inferred table regions.
                    if (
                        (not base_header_num)
                        and (not base_num_like)
                        and bool(getattr(self.hybrid_options, "update_non_numeric", False))
                        and bool(getattr(self.hybrid_options, "update_non_numeric_table_only", True))
                        and (not in_table_region)
                    ):
                        self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                        if isinstance(self._last_update_diffs, list):
                            self._last_update_diffs.append(
                                {
                                    "bbox": _bbox_obj(cell),
                                    "baseline": original_text,
                                    "candidate": new_text,
                                    "candidate_raw": new_text_raw,
                                    "accepted": False,
                                    "reason": "skip_text_not_in_table",
                                }
                            )
                        continue

                    # Numeric acceptance hardening: if the baseline numeric tokens looked confident,
                    # do not overwrite the cell even if it was routed due to table heuristics / high thresholds.
                    if base_num_like and (not base_header_num) and bool(
                        getattr(self.hybrid_options, "accept_numeric_only_if_low_token_conf", True)
                    ):
                        try:
                            thr = float(
                                getattr(
                                    self.hybrid_options,
                                    "numeric_accept_token_confidence_threshold",
                                    min(0.75, float(getattr(self.hybrid_options, "number_confidence_threshold", 0.85))),
                                )
                            )
                            if isinstance(routed_min_num_conf, (int, float)) and float(routed_min_num_conf) >= thr:
                                self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                                if isinstance(self._last_update_diffs, list):
                                    self._last_update_diffs.append(
                                        {
                                            "bbox": _bbox_obj(cell),
                                            "baseline": original_text,
                                            "candidate": new_text,
                                            "candidate_raw": new_text_raw,
                                            "accepted": False,
                                            "reason": "skip_numeric_high_token_conf",
                                        }
                                    )
                                continue
                        except Exception:
                            pass

                    # Additional numeric hardening: if the baseline looks numeric-like, only accept candidates that also look strictly numeric.
                    if base_num_like and (not base_header_num):
                        if not _is_strict_numeric_candidate(new_text):
                            self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                            if isinstance(self._last_update_diffs, list):
                                self._last_update_diffs.append(
                                    {
                                        "bbox": _bbox_obj(cell),
                                        "baseline": original_text,
                                        "candidate": new_text,
                                        "candidate_raw": new_text_raw,
                                        "accepted": False,
                                        "reason": "skip_candidate_not_numeric",
                                    }
                                )
                            continue

                        if not _numeric_digit_ratio_ok(original_text, new_text):
                            self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                            if isinstance(self._last_update_diffs, list):
                                self._last_update_diffs.append(
                                    {
                                        "bbox": _bbox_obj(cell),
                                        "baseline": original_text,
                                        "candidate": new_text,
                                        "candidate_raw": new_text_raw,
                                        "accepted": False,
                                        "reason": "skip_numeric_truncation",
                                    }
                                )
                            continue

                    # Non-numeric hardening when enabled: only accept small, "diacritic-like" corrections.
                    if (not base_header_num) and (not base_num_like) and bool(getattr(self.hybrid_options, "update_non_numeric", False)):
                        b = (original_text or "").strip()
                        c = (new_text or "").strip()

                        # Only attempt to "fix" text when the baseline was low-confidence.
                        try:
                            conf_thr = float(
                                getattr(
                                    self.hybrid_options,
                                    "table_text_confidence_threshold",
                                    getattr(self.hybrid_options, "confidence_threshold", 0.7),
                                )
                            )
                        except Exception:
                            conf_thr = float(getattr(self.hybrid_options, "confidence_threshold", 0.7) or 0.7)

                        base_conf = float(getattr(cell, "confidence", 0.0) or 0.0)
                        if base_conf >= conf_thr:
                            self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                            if isinstance(self._last_update_diffs, list):
                                self._last_update_diffs.append(
                                    {
                                        "bbox": _bbox_obj(cell),
                                        "baseline": original_text,
                                        "candidate": new_text,
                                        "candidate_raw": new_text_raw,
                                        "accepted": False,
                                        "reason": "skip_text_high_conf",
                                    }
                                )
                            continue

                        # Prevent large spurious expansions.
                        if b and len(c) > int(1.6 * len(b)):
                            self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                            if isinstance(self._last_update_diffs, list):
                                self._last_update_diffs.append(
                                    {
                                        "bbox": _bbox_obj(cell),
                                        "baseline": original_text,
                                        "candidate": new_text,
                                        "candidate_raw": new_text_raw,
                                        "accepted": False,
                                        "reason": "skip_text_expansion",
                                    }
                                )
                            continue
                        if b and len(c) < int(0.50 * len(b)):
                            self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                            if isinstance(self._last_update_diffs, list):
                                self._last_update_diffs.append(
                                    {
                                        "bbox": _bbox_obj(cell),
                                        "baseline": original_text,
                                        "candidate": new_text,
                                        "candidate_raw": new_text_raw,
                                        "accepted": False,
                                        "reason": "skip_text_truncation",
                                    }
                                )
                            continue

                        # Require strong overlap in accent-stripped form (prevents unrelated replacements).
                        nb = re.sub(r"\s+", " ", _strip_accents_basic(b)).strip().lower()
                        nc = re.sub(r"\s+", " ", _strip_accents_basic(c)).strip().lower()
                        if nb and nc:
                            lcs = _lcs_len(nb, nc)
                            denom = max(1, min(len(nb), len(nc)))
                            try:
                                min_ratio = float(getattr(self.hybrid_options, "table_text_min_accent_stripped_lcs_ratio", 0.65))
                            except Exception:
                                min_ratio = 0.65

                            if (lcs / denom) < min_ratio:
                                self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                                if isinstance(self._last_update_diffs, list):
                                    self._last_update_diffs.append(
                                        {
                                            "bbox": _bbox_obj(cell),
                                            "baseline": original_text,
                                            "candidate": new_text,
                                            "candidate_raw": new_text_raw,
                                            "accepted": False,
                                            "reason": "skip_text_low_overlap",
                                        }
                                    )
                                continue

                    if not _is_plausible_surya_replacement(
                        baseline=original_text,
                        candidate=new_text,
                        max_len_ratio=float(getattr(self.hybrid_options, "max_replacement_len_ratio", 3.0)),
                        max_abs_len=int(getattr(self.hybrid_options, "max_replacement_abs_len", 128)),
                        require_same_charclass=bool(getattr(self.hybrid_options, "require_same_charclass", True)),
                        # For non-numeric updates we want near-identity (mostly diacritics/typos).
                            min_normalized_lcs_ratio=(
                                0.35
                                if (not base_num_like) and bool(getattr(self.hybrid_options, "update_non_numeric", False))
                                else float(getattr(self.hybrid_options, "min_normalized_lcs_ratio", 0.15))
                            ),
                    ):
                        self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                        if isinstance(self._last_update_diffs, list):
                            self._last_update_diffs.append(
                                {
                                    "bbox": _bbox_obj(cell),
                                    "baseline": original_text,
                                    "candidate": new_text,
                                    "candidate_raw": new_text_raw,
                                    "accepted": False,
                                    "reason": "skip_plausibility",
                                }
                            )
                        continue

                    cell.text = new_text
                    cell.orig = new_text  # Also update original
                    self._stats["surya_cells_updated"] += 1

                    if isinstance(self._last_update_diffs, list):
                        self._last_update_diffs.append(
                            {
                                "bbox": _bbox_obj(cell),
                                "baseline": original_text,
                                "candidate": new_text,
                                "candidate_raw": new_text_raw,
                                "accepted": True,
                                "reason": "updated",
                            }
                        )

                    if self.hybrid_options.log_routing_stats:
                        _log.debug(f"Surya re-OCR: '{original_text}' -> '{new_text}'")
            
        except Exception as e:
            self._stats["surya_failures"] += 1
            _log.warning(f"Surya re-OCR failed: {e}")
        finally:
            # Clean up GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            try:
                import torch
                if torch.cuda.is_available():
                    gc.collect()
            except Exception:
                pass
    
    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """
        Process pages with confidence-gated hybrid OCR.
        
        This overrides TesseractOcrCliModel.__call__ to add Surya routing
        after Tesseract OCR but before post-processing.
        """
        if not self.enabled:
            yield from page_batch
            return
        
        for page_i, page in enumerate(page_batch):
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue
            
            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                
                all_ocr_cells = []
                
                for ocr_rect_i, ocr_rect in enumerate(ocr_rects):
                    # Skip zero area boxes
                    if ocr_rect.area() == 0:
                        continue
                    
                    # Get high-resolution image for this OCR region
                    high_res_image = page._backend.get_page_image(
                        scale=self.scale, cropbox=ocr_rect
                    )
                    
                    # Run Tesseract on this region (use parent's method)
                    try:
                        import tempfile
                        import os
                        import pandas as pd
                        
                        with tempfile.NamedTemporaryFile(
                            suffix=".png", mode="w+b", delete=False
                        ) as image_file:
                            fname = image_file.name
                            high_res_image.save(image_file)
                        
                        try:
                            # OSD for orientation detection
                            df_osd = None
                            doc_orientation = 0
                            try:
                                df_osd = self._perform_osd(fname)
                                doc_orientation = _parse_orientation_compat(df_osd)
                            except Exception:
                                pass
                            
                            # Rotate if needed
                            if doc_orientation != 0:
                                high_res_image = high_res_image.rotate(
                                    -doc_orientation, expand=True
                                )
                                high_res_image.save(fname)
                            
                            # Run Tesseract
                            df_result = self._run_tesseract(fname, df_osd)
                            
                        finally:
                            if os.path.exists(fname):
                                os.remove(fname)
                        
                        # Convert Tesseract results to TextCell objects
                        from docling_core.types.doc.base import BoundingBox, CoordOrigin
                        from docling.utils.ocr_utils import tesseract_box_to_bounding_rectangle

                        region_cells: List[TextCell] = []
                        for ix, row in df_result.iterrows():
                            text = row["text"]
                            conf = row["conf"]

                            # TSV structural ids (used for per-line confidence aggregation).
                            try:
                                block_num = int(float(row.get("block_num") or 0))
                                par_num = int(float(row.get("par_num") or 0))
                                line_num = int(float(row.get("line_num") or 0))
                            except Exception:
                                block_num, par_num, line_num = 0, 0, 0

                            left, top = float(row["left"]), float(row["top"])
                            right = left + float(row["width"])
                            bottom = top + row["height"]

                            bbox = BoundingBox(
                                l=left,
                                t=top,
                                r=right,
                                b=bottom,
                                coord_origin=CoordOrigin.TOPLEFT,
                            )
                            rect = tesseract_box_to_bounding_rectangle(
                                bbox,
                                original_offset=ocr_rect,
                                scale=self.scale,
                                orientation=doc_orientation,
                                im_size=high_res_image.size,
                            )

                            cell_index = int(ix) if isinstance(ix, (int, float, str)) else 0

                            cell = TextCell(
                                index=cell_index,
                                text=str(text),
                                orig=str(text),
                                from_ocr=True,
                                confidence=float(conf) / 100.0,
                                rect=rect,
                            )
                            # Store Tesseract TSV bbox in the *region image* coordinate system.
                            # This is what Surya needs when `page_image` is a cropped OCR region.
                            setattr(cell, "_region_bbox", bbox)

                            if block_num > 0 and par_num > 0 and line_num > 0:
                                setattr(cell, "_tsv_line_key", (block_num, par_num, line_num))
                            region_cells.append(cell)

                        # HYBRID ROUTING: Filter and re-OCR
                        table_boxes = []
                        if region_cells:
                            line_min_num_conf = _build_line_min_numeric_conf(df_result)

                            # Routing policy: by default, route only inside inferred table regions.
                            # Docling layout/table predictions are not available yet at OCR stage,
                            # so we infer table regions directly from the TSV words.
                            table_boxes = _infer_table_boxes_from_tsv(df_result)
                            self._stats["inferred_table_boxes"] += len(table_boxes)

                            if not table_boxes:
                                if bool(getattr(self.hybrid_options, "route_table_only", True)):
                                    self._stats["skipped_no_table_clusters"] += len(region_cells)
                                else:
                                    # Table-preferred mode: allow routing even without clusters.
                                    pass


                            route_table_only = bool(getattr(self.hybrid_options, "route_table_only", True))
                            force_surya_in_tables = bool(getattr(self.hybrid_options, "force_surya_in_table_regions", False))

                            cells_to_reocr: List[TextCell] = []
                            for c in region_cells:
                                # IMPORTANT: routing/table overlap must use the same coordinate system.
                                # - `table_boxes` are inferred from TSV word boxes (region-image pixel coords).
                                # - We store each cell's original TSV bbox as `c._region_bbox` (same system).
                                rb = getattr(c, "_region_bbox", None)
                                if rb is None:
                                    # Shouldn't happen for freshly created cells, but be safe.
                                    self._stats["skipped_missing_region_bbox"] = int(self._stats.get("skipped_missing_region_bbox", 0)) + 1
                                    continue

                                in_table = any(_intersect_area(rb, tb) > 0 for tb in table_boxes)
                                setattr(c, "_in_table_region", bool(in_table))

                                if in_table:
                                    self._stats["table_cells"] += 1
                                else:
                                    self._stats["non_table_cells"] += 1
                                    if route_table_only:
                                        continue

                                is_num_like, is_header_num, _ = numeric_likeness(c.text)
                                if is_header_num:
                                    self._stats["skipped_header_numeric"] += 1

                                key = getattr(c, "_tsv_line_key", None)
                                min_num_conf = line_min_num_conf.get(key) if isinstance(key, tuple) else None
                                setattr(c, "_min_numeric_token_conf", min_num_conf)
                                should_route = False
                                if in_table and force_surya_in_tables and (not is_header_num):
                                    should_route = True
                                else:
                                    # Allow Vietnamese text improvements inside tables when enabled.
                                    # This bypasses `route_numeric_only` for table text cells.
                                    if (
                                        in_table
                                        and (not is_header_num)
                                        and (not is_num_like)
                                        and bool(getattr(self.hybrid_options, "update_non_numeric", False))
                                    ):
                                        try:
                                            text_thr = float(
                                                getattr(
                                                    self.hybrid_options,
                                                    "table_text_confidence_threshold",
                                                    getattr(self.hybrid_options, "confidence_threshold", 0.7),
                                                )
                                            )
                                        except Exception:
                                            text_thr = float(getattr(self.hybrid_options, "confidence_threshold", 0.7) or 0.7)

                                        should_route = float(getattr(c, "confidence", 0.0) or 0.0) < text_thr
                                    else:
                                        should_route = self._should_route_to_surya(c, min_numeric_token_conf=min_num_conf)
                            if should_route:
                                cells_to_reocr.append(c)
                                self._stats["eligible_cells"] += 1
                                if min_num_conf is not None and (not is_header_num) and (
                                    min_num_conf < float(self.hybrid_options.number_confidence_threshold)
                                ):
                                    self._stats["routed_low_num_conf"] += 1
                                else:
                                    self._stats["routed_low_conf"] += 1
                            
                            # Update statistics
                            self._stats['total_cells'] += len(region_cells)
                            self._stats['surya_cells'] += len(cells_to_reocr)
                            self._stats['tesseract_cells'] += (
                                len(region_cells) - len(cells_to_reocr)
                            )
                            
                            # Re-OCR with Surya if we have cells to process
                            if cells_to_reocr:
                                if self.hybrid_options.log_routing_stats:
                                    pct = len(cells_to_reocr) / len(region_cells) * 100
                                    _log.info(
                                        f"Routing {len(cells_to_reocr)}/{len(region_cells)} "
                                        f"cells ({pct:.1f}%) to Surya"
                                    )

                                # Re-OCR (modifies cells in-place)
                                self._surya_reocr_cells(
                                    high_res_image, 
                                    cells_to_reocr,
                                    scale=self.scale,
                                )
                        
                        all_ocr_cells.extend(region_cells)
                        
                    except Exception as e:
                        _log.error(f"OCR failed for region {ocr_rect_i}: {e}")
                        continue
                
                # Post-process the cells (parent class method)
                self.post_process_cells(all_ocr_cells, page)

                # Debug snapshot: what OCR produced (after Surya updates + post_process_cells)
                try:
                    self._last_snapshot = self._make_debug_snapshot(page, all_ocr_cells)
                except Exception:
                    self._last_snapshot = None

            
            yield page
    
    def get_stats(self) -> dict:
        """Get routing statistics."""
        stats = {
            "total_cells": int(self._stats.get("total_cells", 0)),
            "surya_cells": int(self._stats.get("surya_cells", 0)),
            "tesseract_cells": int(self._stats.get("tesseract_cells", 0)),
            "table_cells": int(self._stats.get("table_cells", 0)),
            "non_table_cells": int(self._stats.get("non_table_cells", 0)),
            "eligible_cells": int(self._stats.get("eligible_cells", 0)),
            "routed_low_conf": int(self._stats.get("routed_low_conf", 0)),
            "routed_low_num_conf": int(self._stats.get("routed_low_num_conf", 0)),
            "skipped_no_table_clusters": int(self._stats.get("skipped_no_table_clusters", 0)),
            "skipped_header_numeric": int(self._stats.get("skipped_header_numeric", 0)),
            "skipped_missing_region_bbox": int(self._stats.get("skipped_missing_region_bbox", 0)),
            "inferred_table_boxes": int(self._stats.get("inferred_table_boxes", 0)),
            "surya_cells_updated": int(self._stats.get("surya_cells_updated", 0)),
            "surya_update_skipped_sanity": int(self._stats.get("surya_update_skipped_sanity", 0)),
            "surya_update_skipped_count_mismatch": int(self._stats.get("surya_update_skipped_count_mismatch", 0)),
            "surya_update_skipped_non_numeric": int(self._stats.get("surya_update_skipped_non_numeric", 0)),
            "surya_failures": int(self._stats.get("surya_failures", 0)),
        }

        total = stats["total_cells"]
        surya_percentage = (stats["surya_cells"] / total * 100.0) if total > 0 else 0.0
        return {
            **stats,
            "surya_percentage": surya_percentage,
        }
    
    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self._stats = {
            "total_cells": 0,
            "surya_cells": 0,
            "tesseract_cells": 0,
            "table_cells": 0,
            "non_table_cells": 0,
            "eligible_cells": 0,
            "routed_low_conf": 0,
            "routed_low_num_conf": 0,
            "skipped_no_table_clusters": 0,
            "skipped_header_numeric": 0,
            "skipped_missing_region_bbox": 0,
            "inferred_table_boxes": 0,
            "surya_cells_updated": 0,
            "surya_update_skipped_sanity": 0,
            "surya_update_skipped_count_mismatch": 0,
            "surya_update_skipped_non_numeric": 0,
            "surya_failures": 0,
        }

    
    @classmethod
    def get_options_type(cls) -> Type[TesseractCliOcrOptions]:
        return HybridOcrOptions
