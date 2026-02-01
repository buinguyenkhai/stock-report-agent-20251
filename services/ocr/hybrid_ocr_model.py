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


_HEADER_NUMERIC_RE = re.compile(
    r"(?ix)^(?:"
    r"q[1-4][./-]?\d{4}"
    r"|note\s*\d+(?:\.\d+)*"
    r"|ghi\s*chu\s*\d+(?:\.\d+)*"
    r"|[ivxlcdm]{1,10}"
    r"|[a-z]{1,3}\s*\d{1,3}"
    r")$"
)


_STRICT_NUMERIC_CANDIDATE_RE = re.compile(r"(?i)^[0-9\s.,/%()\-+đvnusdeur]*$")
_SHORT_ALNUM_CODE_RE = re.compile(r"(?i)^[a-z]{1,3}\d{1,3}$")


_STRICT_NUMERIC_NO_LETTERS_RE = re.compile(r"(?i)^[0-9\s.,/%()\-+đ₫]*$")
_STRICT_NUMERIC_GARBLED_RE = re.compile(r"(?i)^[0-9\s.,/%()\-+đ₫]*(?:vnd|usd|eur)?$")


def _compact_ws(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip())


def _digits_only(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isdigit())


def _canon_noop(s: str) -> str:
    """Treat whitespace-only differences as no-ops."""
    s2 = (s or "").strip()
    return re.sub(r"\s+", " ", s2)


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


def _is_one_digit_substitution(baseline: str, candidate: str) -> bool:
    """True if numeric digit strings differ by exactly one digit (same length)."""
    bd = _digits_only(baseline)
    cd = _digits_only(candidate)
    if not bd or not cd:
        return False
    if len(bd) != len(cd):
        return False
    diffs = sum(1 for a, b in zip(bd, cd) if a != b)
    return diffs == 1


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


def _numeric_signature(s: str) -> tuple[bool, bool, str]:
    s0 = (s or "").strip()
    neg = False
    if s0.startswith("(") and s0.endswith(")"):
        neg = True
    if s0.startswith("-"):
        neg = True
    pct = ("%" in s0)
    digs = _digits_only(s0)
    return (bool(neg), bool(pct), str(digs))


def _digit_count_plausible(candidate: str, *, median: Optional[int]) -> bool:
    if not isinstance(median, int) or median <= 0:
        return True
    d = len(_digits_only(candidate))
    if d <= 0:
        return True
    allowed = max(1, int(round(0.25 * float(median))))
    return abs(d - int(median)) <= allowed


def _translate_ocr_lookalikes(s: str) -> str:
    trans = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})
    return (s or "").translate(trans)


def _is_strict_numeric_candidate(s: str) -> bool:
    s2 = (s or "").strip().lower()
    if not s2:
        return False
    if not any(ch.isdigit() for ch in s2):
        return False
    return bool(_STRICT_NUMERIC_CANDIDATE_RE.fullmatch(s2))


def _is_strict_numeric_candidate_no_letters(s: str) -> bool:
    """Stricter numeric gate for garbled pages: no alphabetic suffixes."""
    s2 = (s or "").strip().lower()
    if not s2:
        return False
    if not any(ch.isdigit() for ch in s2):
        return False
    return bool(_STRICT_NUMERIC_NO_LETTERS_RE.fullmatch(s2))


def _is_strict_numeric_candidate_garbled(s: str) -> bool:
    """Numeric gate for garbled pages.

    Reject arbitrary alphabetic suffixes (e.g. "876.AET"), but allow a small,
    explicit set of currency markers that commonly appear in financial tables.
    """
    s2 = (s or "").strip().lower()
    if not s2:
        return False
    if not any(ch.isdigit() for ch in s2):
        return False
    return bool(_STRICT_NUMERIC_GARBLED_RE.fullmatch(s2))


def _sanitize_surya_text(s: str) -> str:
    """Normalize common Surya artifacts (HTML-ish tags, hard line breaks)."""
    s2 = (s or "")
    if not s2:
        return ""
    # Normalize HTML-ish line breaks to spaces.
    s2 = re.sub(r"(?i)<\s*br\s*/?\s*>", " ", s2)
    # Drop other tags.
    s2 = re.sub(r"<[^>]+>", " ", s2)
    # Normalize whitespace.
    s2 = s2.replace("\u00a0", " ")
    s2 = re.sub(r"\s+", " ", s2).strip()

    # Normalize non-ASCII digits to ASCII (e.g., Arabic-Indic digits).
    out_chars: list[str] = []
    for ch in s2:
        if ch.isdigit() and (ch < "0" or ch > "9"):
            try:
                out_chars.append(str(unicodedata.digit(ch)))
                continue
            except Exception:
                pass
        out_chars.append(ch)
    s2 = "".join(out_chars)

    # Numeric cleanup: Surya sometimes emits quote-like separators (" / ' / ` / ’).
    # Only strip them when the string is otherwise numeric-ish (no alphabetic letters).
    if any(ch.isdigit() for ch in s2) and (not any(ch.isalpha() for ch in s2)):
        cleaned = re.sub(r"[\"'`’]", "", s2)
        if cleaned and _STRICT_NUMERIC_CANDIDATE_RE.fullmatch(cleaned.lower()):
            s2 = cleaned
    return s2


def _looks_like_short_alnum_code(s: str) -> bool:
    compact = _compact_ws(s)
    if not compact:
        return False
    if not _SHORT_ALNUM_CODE_RE.fullmatch(compact):
        return False
    # Disambiguate from numeric cells with separators/currency; those should be treated as numbers.
    if re.search(r"[\.,/%()\-+]", compact):
        return False
    return True


def _looks_like_numeric_with_alpha_suffix_junk(s: str) -> bool:
    """Detect numeric-looking tokens with an unexpected alpha suffix.

    Examples we want to reject (common OCR junk on garbled pages):
    - "876.AET"
    - "123ABC" (unless the suffix is an allowed currency token)
    """
    compact = _compact_ws(s)
    if not compact:
        return False
    if not compact[0].isdigit():
        return False
    if not any(ch.isalpha() for ch in compact):
        return False

    first_alpha = None
    for i, ch in enumerate(compact):
        if ch.isalpha():
            first_alpha = int(i)
            break
    if first_alpha is None:
        return False

    prefix = compact[:first_alpha]
    suffix = compact[first_alpha:]
    if sum(ch.isdigit() for ch in prefix) < 3:
        return False

    # Prefix must be numeric-ish (digits + common separators).
    if re.search(r"[a-z]", prefix, flags=re.IGNORECASE):
        return False
    if re.search(r"[^0-9.,/%()\-+]", prefix):
        return False

    # Suffix must be purely letters.
    if not suffix.isalpha():
        return False

    # Allow explicit currency tokens.
    if suffix.lower() in ("vnd", "usd", "eur"):
        return False
    return True


def _normalize_for_numeric_likeness(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""

    s_low = s.lower()
    has_digit = any(ch.isdigit() for ch in s)
    has_currency_token = bool(re.search(r"(?i)\b(vnd|vnđ|usd|eur)\b", s))
    # Treat 'đ' as a currency symbol only when there is numeric context.
    has_currency_symbol = ("₫" in s) or (("đ" in s_low) and has_digit)
    has_currency = has_currency_token or has_currency_symbol

    # Avoid mapping arbitrary Vietnamese words into digits.
    # Only translate lookalikes when the token is already numeric (has digits/currency), or when
    # it is a long token composed entirely of OCR lookalikes (+ optional numeric punctuation).
    if not has_digit and not has_currency:
        compact = re.sub(r"\s+", "", s)

        # Don't reinterpret roman numerals as arabic digits.
        if re.fullmatch(r"(?i)[ivxlcdm]+", compact or ""):
            return s

        if len(compact) < 4:
            return s

        # Pure lookalike-only token (common OCR: "lOOO" => "1000").
        if re.fullmatch(r"[OoIlSB]+", compact):
            trans = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})
            return s.translate(trans)

        # Lookalikes with punctuation (e.g. "O.OO" / "l,OOO").
        if re.fullmatch(r"[OoIlSB\.,/%()\+\-]+", compact) and re.search(r"[OoIlSB]", compact):
            trans = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})
            return s.translate(trans)

        return s

    trans = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})
    return s.translate(trans)


_SUSPICIOUS_NUMERIC_LOOKALIKE_RE = re.compile(r"[OoIlSB]")
_SUSPICIOUS_NUMERIC_SEP_GLITCH_RE = re.compile(r"[.,]\s*[.,]")
_SUSPICIOUS_NUMERIC_THOUSANDS_RE = re.compile(r"(?x)^(?:\(\s*)?-?\s*\d{1,3}(?:[\.,]\d{3}){2,}\s*(?:\))?$")


def _is_suspicious_numeric_ocr(text: str) -> bool:
    """Heuristic: a numeric-like token that is likely wrong despite high confidence."""

    s = (text or "").strip()
    if not s:
        return False

    is_num_like, is_header_num, _ = numeric_likeness(s)
    if (not is_num_like) or is_header_num:
        return False

    has_digit = any(ch.isdigit() for ch in s)

    # Missing thousands separators is a common OCR error (e.g., "24327" vs "2.327").
    # Route long digit-only tokens so Surya can restore formatting.
    compact_ws = re.sub(r"\s+", "", s)
    if compact_ws.isdigit() and len(compact_ws) >= 5:
        return True

    # Thousand-grouped large numbers are high-impact; route for a second opinion even when
    # Tesseract confidence is high (acceptance still remains conservative).
    if _SUSPICIOUS_NUMERIC_THOUSANDS_RE.fullmatch(compact_ws):
        return True

    # Unbalanced parentheses in numeric cells are common OCR artifacts.
    if has_digit and ((s.startswith("(") and (not s.endswith(")"))) or (s.endswith(")") and (not s.startswith("(")))):
        return True

    # Separator glitches.
    if _SUSPICIOUS_NUMERIC_SEP_GLITCH_RE.search(s):
        return True

    # Unexpected junk characters in an otherwise numeric-like token.
    remainder = s
    remainder = re.sub(r"(?i)\b(vnd|vnđ|usd|eur)\b", "", remainder)
    remainder = remainder.replace("đ", "").replace("Đ", "").replace("₫", "")
    remainder = re.sub(r"[0-9\s\.,/%()\+\-]", "", remainder)
    if has_digit and remainder and (not remainder.isalpha()):
        return True

    compact = re.sub(r"\s+", "", s)
    if not _SUSPICIOUS_NUMERIC_LOOKALIKE_RE.search(compact):
        return False

    # Exclude short alnum codes like "A1" / "B12" (often row/column markers).
    if re.fullmatch(r"(?i)[a-z]{1,3}\d{1,3}", compact) and len(compact) <= 4:
        return False

    has_digit = any(ch.isdigit() for ch in compact)
    has_cue = any(ch in compact for ch in (".", ",", "%", "/", "(", ")", "+", "-"))
    if has_digit or has_cue:
        return True

    # Pure lookalike tokens: require length to avoid routing single-letter labels.
    if len(compact) >= 4 and re.fullmatch(r"[OoIlSB]+", compact):
        if re.fullmatch(r"(?i)[ivxlcdm]+", compact):
            return False
        return True

    return False


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

    raw2 = re.sub(r"\s+", " ", raw).strip().lower()

    # Treat short alnum codes (e.g., S80, A12) as non-numeric-like by default.
    # These are common in Vietnamese financial tables (row/column codes, note markers).
    # We still return is_header_numeric=True so routing/acceptance stays conservative.
    if _looks_like_short_alnum_code(raw2):
        return False, True, 0.0

    is_header_numeric = bool(_HEADER_NUMERIC_RE.match(raw2))

    s = _normalize_for_numeric_likeness(raw)
    s2 = re.sub(r"\s+", " ", s).strip().lower()

    digit_count = sum(1 for c in s if c.isdigit())
    alpha_count = sum(1 for c in s if c.isalpha())
    length = max(len(s), 1)

    digit_ratio = digit_count / length

    has_currency = any(x in s2 for x in ("vnd", "vnđ", "usd", "eur")) or (("đ" in s2) and (digit_count > 0))
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


def _annotate_table_column_numericness(cells: list[TextCell], table_boxes: list[BoundingBox]) -> None:
    """Annotate cells with a coarse per-column numeric ratio.

    This is used as a semantic guardrail: do not allow alnum codes like "S80" to be
    overwritten as numbers unless the surrounding column is predominantly numeric.
    """

    if not cells or not table_boxes:
        return

    def _x_center(rb: BoundingBox) -> float:
        return (float(rb.l) + float(rb.r)) * 0.5

    def _height(rb: BoundingBox) -> float:
        return max(1.0, float(rb.b) - float(rb.t))

    for tb in table_boxes:
        tb_cells: list[TextCell] = []
        for c in cells:
            rb = getattr(c, "_region_bbox", None)
            if rb is None:
                continue
            if _intersect_area(rb, tb) <= 0:
                continue
            tb_cells.append(c)

        if len(tb_cells) < 12:
            continue

        hs = sorted(_height(getattr(c, "_region_bbox")) for c in tb_cells if getattr(c, "_region_bbox", None) is not None)
        if not hs:
            continue
        median_h = hs[len(hs) // 2]
        gap_thr = max(40.0, float(median_h) * 3.0)

        xs = sorted((_x_center(getattr(c, "_region_bbox")), c) for c in tb_cells if getattr(c, "_region_bbox", None) is not None)
        if not xs:
            continue

        clusters: list[list[TextCell]] = []
        cur: list[TextCell] = [xs[0][1]]
        prev_x = float(xs[0][0])
        for x, c in xs[1:]:
            x = float(x)
            if (x - prev_x) >= gap_thr:
                clusters.append(cur)
                cur = [c]
            else:
                cur.append(c)
            prev_x = x
        if cur:
            clusters.append(cur)

        for col_cells in clusters:
            if not col_cells:
                continue
            strict_num = 0
            digit_heavy = 0
            denom = 0
            digit_counts: list[int] = []
            for c in col_cells:
                txt = str(getattr(c, "text", "") or "")
                _is_num_like, is_hdr, _ = numeric_likeness(txt)
                if is_hdr:
                    continue
                denom += 1

                # Strict numeric count (low false positives).
                if _is_strict_numeric_candidate(txt):
                    strict_num += 1

                # Digit-heavy count (more robust on garbled pages where letters leak into numeric cells).
                compact = _compact_ws(txt)
                if compact:
                    digs = sum(ch.isdigit() for ch in compact)
                    alphas = sum(ch.isalpha() for ch in compact)
                    dig_ratio = digs / max(1, len(compact))
                    if (digs >= 3) and (dig_ratio >= 0.45):
                        digit_heavy += 1
                        if digs > 0:
                            digit_counts.append(int(digs))

            ratio = (strict_num / denom) if denom > 0 else 0.0
            digit_heavy_ratio = (digit_heavy / denom) if denom > 0 else 0.0
            digit_median: Optional[int] = None
            if digit_counts:
                digit_counts.sort()
                digit_median = int(digit_counts[len(digit_counts) // 2])
            for c in col_cells:
                prev = getattr(c, "_table_col_numeric_ratio", None)
                prev_dh = getattr(c, "_table_col_digit_heavy_ratio", None)
                if (
                    (prev is None)
                    or (ratio > float(prev))
                    or (prev_dh is None)
                    or (digit_heavy_ratio > float(prev_dh))
                ):
                    setattr(c, "_table_col_numeric_ratio", float(ratio))
                    setattr(c, "_table_col_digit_heavy_ratio", float(digit_heavy_ratio))
                    if digit_median is not None:
                        setattr(c, "_table_col_digit_median", int(digit_median))


def _tsv_looks_garbled(df_result) -> bool:
    try:
        if df_result is None or getattr(df_result, "empty", False):
            return False

        word_count = 0
        conf_sum = 0.0
        conf_n = 0
        low_conf = 0
        weird_mixed = 0
        weird_symbol = 0
        weird_non_latin_alpha = 0
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
            word_count += 1

            # Confidence (Tesseract word conf 0..100, -1 for invalid)
            try:
                conf = float(row.get("conf"))
            except Exception:
                conf = -1.0
            if conf >= 0:
                conf_sum += conf
                conf_n += 1
                if conf < 30:
                    low_conf += 1

            compact = _compact_ws(txt)
            has_weird_symbol = False
            has_non_latin_alpha = False
            for ch in compact:
                if ch == "₫":
                    continue
                cat = unicodedata.category(ch)
                if cat and cat[0] == "S":
                    has_weird_symbol = True
                if ch.isalpha():
                    try:
                        name = unicodedata.name(ch)
                    except ValueError:
                        name = ""
                    if name and ("LATIN" not in name):
                        has_non_latin_alpha = True
            if has_weird_symbol:
                weird_symbol += 1
            if has_non_latin_alpha:
                weird_non_latin_alpha += 1

            # Mixed digit+alpha tokens without common separators are often garbage.
            if len(compact) >= 6:
                digs = sum(ch.isdigit() for ch in compact)
                alphas = sum(ch.isalpha() for ch in compact)
                if digs >= 2 and alphas >= 2 and (not re.search(r"[\.,/%()\-+]", compact)):
                    weird_mixed += 1

        if word_count < 30:
            return False
        wm = weird_mixed / max(1, word_count)
        ws = weird_symbol / max(1, word_count)
        wnl = weird_non_latin_alpha / max(1, word_count)

        avg_conf = (conf_sum / conf_n) if conf_n > 0 else 100.0
        low_conf_frac = low_conf / max(1, conf_n)

        return (
            (wm >= 0.20)
            or (ws >= 0.05)
            or (wnl >= 0.03)
            or ((word_count >= 60) and (avg_conf < 35.0) and (low_conf_frac >= 0.55))
        )
    except Exception:
        return False


def _cell_looks_garbled(text: str, conf01: float) -> bool:
    """Cell-level garble heuristic for acceptance gating.

    Intentionally conservative: only triggers on very low-confidence cells with
    a high ratio of symbols/non-alnum noise.
    """
    try:
        t = str(text or "").strip()
        if not t:
            return False
        c = float(conf01 or 0.0)
        compact = _compact_ws(t)
        if len(compact) < 4:
            return False

        alnum = _alnum_count(compact)
        alnum_ratio = alnum / max(1, len(compact))

        has_symbol = False
        has_non_latin_alpha = False
        for ch in compact:
            if ch == "₫":
                continue
            cat = unicodedata.category(ch)
            if cat and cat[0] == "S":
                has_symbol = True
            if ch.isalpha():
                try:
                    name = unicodedata.name(ch)
                except ValueError:
                    name = ""
                if name and ("LATIN" not in name):
                    has_non_latin_alpha = True

        if c <= 0.15:
            return True
        if c <= 0.25 and (has_non_latin_alpha or has_symbol):
            return True
        if c <= 0.25 and alnum_ratio < 0.55:
            return True

        # High-risk numeric junk: digit-heavy strings with alphabetic suffixes
        # (common on garbled pages, e.g. "876.AET"). Mark as garbled even if the
        # cell confidence isn't extremely low so numeric-context gating can apply.
        if _digits_only(compact) and (sum(ch.isdigit() for ch in compact) >= 3) and any(ch.isalpha() for ch in compact):
            dig_ratio = sum(ch.isdigit() for ch in compact) / max(1, len(compact))
            if dig_ratio >= 0.45:
                return True
        return False
    except Exception:
        return False


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
    confidence_threshold: float = 0.9
    number_confidence_threshold: float = 0.95

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

    # If True, route numeric-like cells that look suspicious even when confidence is high.
    route_suspicious_numeric: bool = True
    suspicious_numeric_confidence_cap: float = 1.0

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

    # Tail rescue: allow much looser overlap when a cell looks garbled.
    garbled_text_min_accent_stripped_lcs_ratio: float = 0.20

    # Table numeric context: inferred per-column numericness threshold.
    numeric_context_min_col_ratio: float = 0.60

    # Numeric acceptance hardening: only accept numeric replacements when token-level
    # evidence suggests the baseline is unreliable.
    accept_numeric_only_if_low_token_conf: bool = True

    # Separate acceptance threshold for numeric token confidence.
    # This should usually be <= number_confidence_threshold, otherwise we may overwrite
    # correct numbers that Tesseract scored moderately-high.
    numeric_accept_token_confidence_threshold: float = 0.85

    # Semantic guardrail: only allow alnum->numeric rewrites in columns that are predominantly numeric.
    numeric_column_min_ratio_for_alnum_to_numeric: float = 0.70

    # Numeric recovery: allow non-numeric/garbled baselines to be replaced with numeric
    # candidates only when the column is strongly numeric.
    numeric_column_min_ratio_for_numeric_recovery: float = 0.85
    numeric_recovery_max_baseline_confidence: float = 0.50

    # Extra safety for high-impact numeric overwrites: re-OCR a subset of numeric cells
    # twice with a small polygon padding and require agreement.
    enable_numeric_self_consistency: bool = True
    numeric_self_consistency_pad_px: int = 2

    # Tail rescue: detect garbled TSV regions and force Surya for in-table cells.
    enable_garble_rescue: bool = True

    # Tail rescue (garbled TSV): Surya can be sensitive to contrast. Optionally run a
    # second Surya pass on a lightly enhanced image and choose the better candidate.
    enable_garbled_surya_enhanced_pass: bool = True
    garbled_surya_enhanced_contrast: float = 1.6
    garbled_surya_enhanced_sharpness: float = 1.4
    garbled_surya_polygon_pad_px: int = 1

    # Semantic safety: reject digit+alpha junk in numeric table contexts (e.g., "876.AET").
    digit_alpha_junk_min_digits: int = 3
    digit_alpha_junk_min_digit_ratio: float = 0.45

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
            if len(r) < 5:
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
            return (col_groups >= 3 and num_ratio >= 0.12) or (col_groups >= 4)

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

        # Pad bands so edge cells (row labels / rightmost numbers) are still considered in-table.
        pad_x = max(10.0, float(median_h) * 1.0)
        pad_y = max(4.0, float(median_h) * 0.35)
        padded: List[BoundingBox] = []
        for b in bands:
            padded.append(
                BoundingBox(
                    l=float(b.l) - pad_x,
                    t=float(b.t) - pad_y,
                    r=float(b.r) + pad_x,
                    b=float(b.b) + pad_y,
                    coord_origin=b.coord_origin,
                )
            )
        return padded
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

            # Word bbox in the region image coordinate system.
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

    Tesseract TSV contains multi-level rows. For routing, using the minimum numeric-like
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
            "routed_suspicious_numeric": 0,
            "skipped_no_table_clusters": 0,
            "skipped_header_numeric": 0,
            "skipped_missing_region_bbox": 0,
            "inferred_table_boxes": 0,
            "garbled_regions": 0,
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

        def _run_surya_pass(img: Image.Image) -> Optional[tuple[list[str], list[str], list[float]]]:
            results = self.surya_model(
                images=[img],
                polygons=[polygons],
                recognition_batch_size=self.hybrid_options.surya_batch_size,
            )
            if not results or (not getattr(results[0], "text_lines", None)):
                return None
            text_lines = list(results[0].text_lines or [])
            if len(text_lines) != len(cell_polys):
                return None

            out_raw: list[str] = []
            out: list[str] = []
            out_conf: list[float] = []
            for tl in text_lines:
                raw_txt = str(getattr(tl, "text", "") or "")
                out_raw.append(raw_txt)
                out.append(_sanitize_surya_text(raw_txt))
                try:
                    out_conf.append(float(getattr(tl, "confidence")))
                except Exception:
                    out_conf.append(1.0)
            return out_raw, out, out_conf

        def _choose_garbled_candidate(
            *,
            idx: int,
            pass1: tuple[list[str], list[str], list[float]],
            pass2: tuple[list[str], list[str], list[float]],
        ) -> tuple[str, str, float]:
            p1_raw, p1, p1_conf = pass1
            p2_raw, p2, p2_conf = pass2

            c1 = str(p1[idx] or "").strip()
            c2 = str(p2[idx] or "").strip()
            if (not c2) or (c1 == c2):
                return p1_raw[idx], p1[idx], float(p1_conf[idx])
            if not c1:
                return p2_raw[idx], p2[idx], float(p2_conf[idx])

            cell = cell_polys[idx][0]
            in_table = bool(getattr(cell, "_in_table_region", False))
            col_ratio = float(getattr(cell, "_table_col_numeric_ratio", 0.0) or 0.0)
            col_dh_ratio = float(getattr(cell, "_table_col_digit_heavy_ratio", 0.0) or 0.0)
            col_ratio_eff = max(col_ratio, col_dh_ratio)
            base_num_like, base_header_num, _ = numeric_likeness(str(getattr(cell, "text", "") or ""))

            garbled_cell = bool(getattr(cell, "_garbled_region", False)) or bool(getattr(cell, "_garbled_cell", False))

            try:
                col_thr = float(getattr(self.hybrid_options, "numeric_context_min_col_ratio", 0.60))
            except Exception:
                col_thr = 0.60
            # Treat strongly-numeric table columns as numeric context even if the
            # baseline/candidates are OCR-garbled (digit+letter junk can confuse
            # numeric_likeness). Final acceptance is still gated downstream.
            numeric_context = bool(
                (base_num_like and (not base_header_num))
                or (
                    in_table
                    and (col_ratio_eff >= col_thr)
                    and (not _looks_like_short_alnum_code(str(getattr(cell, "text", "") or "")))
                    # In garbled contexts, numeric_likeness() can misclassify junk as "header".
                    # If the column is strongly numeric, still treat it as numeric context so we
                    # can block digit+letter junk.
                    and ((not base_header_num) or garbled_cell)
                )
            )

            if numeric_context:
                strict1 = _is_strict_numeric_candidate_garbled(c1) if garbled_cell else _is_strict_numeric_candidate(c1)
                strict2 = _is_strict_numeric_candidate_garbled(c2) if garbled_cell else _is_strict_numeric_candidate(c2)
                if strict2 and (not strict1):
                    return p2_raw[idx], p2[idx], float(p2_conf[idx])
                if strict1 and (not strict2):
                    return p1_raw[idx], p1[idx], float(p1_conf[idx])

            # Otherwise, pick the higher-confidence candidate.
            if float(p2_conf[idx]) > float(p1_conf[idx]) + 1e-6:
                return p2_raw[idx], p2[idx], float(p2_conf[idx])
            return p1_raw[idx], p1[idx], float(p1_conf[idx])


        def _bbox_obj(cell: TextCell) -> dict[str, float]:
            bb = cell.rect.to_bounding_box()
            return {
                "l": float(bb.l),
                "t": float(bb.t),
                "r": float(bb.r),
                "b": float(bb.b),
            }

        try:
            import torch

            # Prepare polygons for Surya in the *region image* coordinate system.
            # Note: `page_image` here is a cropped, high-res OCR region image (possibly rotated).
            # Tesseract's TSV coordinates (left/top/width/height) are already in this coordinate
            # system. We attach those coords to each TextCell as `_region_bbox` at creation time.
            w, h = page_image.size

            region_garbled = any(bool(getattr(c, "_garbled_region", False)) for c in (cells or []))
            try:
                garbled_pad = int(getattr(self.hybrid_options, "garbled_surya_polygon_pad_px", 1) or 0)
            except Exception:
                garbled_pad = 0
            garbled_pad = max(0, min(garbled_pad, 6))

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

                # Garbled-rescue: add a tiny pad to reduce crop tightness.
                pad = 0
                if region_garbled and garbled_pad > 0:
                    if bool(getattr(cell, "_garbled_cell", False)) or bool(getattr(cell, "_garbled_region", False)):
                        pad = garbled_pad

                # Clamp to image bounds (Surya will fail if polygons are out-of-bounds).
                l = max(0, min(l - pad, max(0, w - 1)))
                r = max(0, min(r + pad, max(0, w - 1)))
                t = max(0, min(t - pad, max(0, h - 1)))
                b = max(0, min(b + pad, max(0, h - 1)))

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
            
            # First-pass OCR with Surya
            pass1 = _run_surya_pass(page_image)
            if pass1 is None:
                return
            cand1_raw, cand1, cand1_conf = pass1

            if len(cand1) != len(cell_polys):
                self._stats["surya_update_skipped_count_mismatch"] = int(
                    self._stats.get("surya_update_skipped_count_mismatch", 0)
                ) + 1
                _log.warning(
                    "Surya output count mismatch: got %d text lines for %d cells; skipping update",
                    len(cand1),
                    len(cell_polys),
                )
                return

            # Optional garbled rescue: run a second pass on a lightly enhanced image.
            cand_raw = list(cand1_raw)
            cand = list(cand1)
            cand_conf = list(cand1_conf)

            if region_garbled and bool(getattr(self.hybrid_options, "enable_garbled_surya_enhanced_pass", True)):
                try:
                    from PIL import ImageEnhance, ImageOps

                    try:
                        contrast = float(getattr(self.hybrid_options, "garbled_surya_enhanced_contrast", 1.6) or 1.6)
                    except Exception:
                        contrast = 1.6
                    try:
                        sharpness = float(getattr(self.hybrid_options, "garbled_surya_enhanced_sharpness", 1.4) or 1.4)
                    except Exception:
                        sharpness = 1.4

                    def _enhance_for_surya(img: Image.Image) -> Image.Image:
                        g = img.convert("L")
                        g = ImageOps.autocontrast(g, cutoff=2)
                        g = ImageEnhance.Contrast(g).enhance(max(0.8, min(contrast, 3.0)))
                        g = ImageEnhance.Sharpness(g).enhance(max(0.8, min(sharpness, 3.0)))
                        return g.convert("RGB")

                    enhanced = _enhance_for_surya(page_image)
                    pass2 = _run_surya_pass(enhanced)
                    if pass2 is not None:
                        for idx in range(len(cell_polys)):
                            raw_sel, txt_sel, conf_sel = _choose_garbled_candidate(idx=idx, pass1=pass1, pass2=pass2)
                            cand_raw[idx] = raw_sel
                            cand[idx] = txt_sel
                            cand_conf[idx] = float(conf_sel)
                except Exception:
                    # Enhancement is best-effort; fall back to first-pass output.
                    pass

            # Optional numeric self-consistency: re-OCR a subset of risky numeric overwrites.
            sc_required: dict[int, bool] = {}
            sc_cand2_raw: dict[int, str] = {}
            sc_cand2: dict[int, str] = {}
            if bool(getattr(self.hybrid_options, "enable_numeric_self_consistency", True)):
                try:
                    pad = int(getattr(self.hybrid_options, "numeric_self_consistency_pad_px", 2) or 0)
                except Exception:
                    pad = 0
                if pad > 0:
                    for idx in range(len(cell_polys)):
                        cell = cell_polys[idx][0]
                        original_text = str(cell.text or "")
                        new_text = cand[idx]
                        if _canon_noop(original_text) == _canon_noop(new_text):
                            continue
                        garbled_cell = bool(getattr(cell, "_garbled_region", False)) or bool(
                            getattr(cell, "_garbled_cell", False)
                        )
                        if garbled_cell:
                            if not _is_strict_numeric_candidate_garbled(new_text):
                                continue
                        else:
                            if not _is_strict_numeric_candidate(new_text):
                                continue

                        base_num_like, base_header_num, _ = numeric_likeness(original_text)
                        routed_min_num_conf = getattr(cell, "_min_numeric_token_conf", None)

                        # Require self-consistency for:
                        # - alnum-code -> numeric rewrites (high semantic risk)
                        # - overwriting high-confidence numeric tokens (tail rescue)
                        base_is_code = _looks_like_short_alnum_code(original_text)
                        high_conf_numeric = False
                        if base_num_like and (not base_header_num):
                            try:
                                thr = float(
                                    getattr(
                                        self.hybrid_options,
                                        "numeric_accept_token_confidence_threshold",
                                        min(0.75, float(getattr(self.hybrid_options, "number_confidence_threshold", 0.85))),
                                    )
                                )
                            except Exception:
                                thr = 0.85
                            if isinstance(routed_min_num_conf, (int, float)) and float(routed_min_num_conf) >= thr:
                                if not _is_one_digit_substitution(original_text, new_text):
                                    high_conf_numeric = True

                        in_table_region = bool(getattr(cell, "_in_table_region", False))
                        col_num_ratio = float(getattr(cell, "_table_col_numeric_ratio", 0.0) or 0.0)
                        col_dh_ratio = float(getattr(cell, "_table_col_digit_heavy_ratio", 0.0) or 0.0)
                        col_ratio_eff = max(col_num_ratio, col_dh_ratio)
                        base_conf = float(getattr(cell, "confidence", 0.0) or 0.0)

                        try:
                            rec_col_thr = float(
                                getattr(self.hybrid_options, "numeric_column_min_ratio_for_numeric_recovery", 0.85)
                            )
                        except Exception:
                            rec_col_thr = 0.85
                        try:
                            rec_conf_thr = float(
                                getattr(self.hybrid_options, "numeric_recovery_max_baseline_confidence", 0.50)
                            )
                        except Exception:
                            rec_conf_thr = 0.50

                        numeric_recovery = bool(
                            (not base_num_like)
                            and (not base_is_code)
                            and in_table_region
                            and (col_ratio_eff >= rec_col_thr)
                            and ((base_conf <= rec_conf_thr) or garbled_cell)
                        )

                        if base_is_code or high_conf_numeric or numeric_recovery:
                            sc_required[idx] = True

                    if sc_required:
                        sc_polys: list[list[list[int]]] = []
                        sc_map: list[int] = []
                        for idx in sc_required.keys():
                            poly = polygons[idx]
                            l = max(0, min(int(poly[0][0]) - pad, max(0, w - 1)))
                            t = max(0, min(int(poly[0][1]) - pad, max(0, h - 1)))
                            r = max(0, min(int(poly[2][0]) + pad, max(0, w - 1)))
                            b = max(0, min(int(poly[2][1]) + pad, max(0, h - 1)))
                            if r <= l or b <= t or (r - l) < 2 or (b - t) < 2:
                                continue
                            sc_polys.append([[l, t], [r, t], [r, b], [l, b]])
                            sc_map.append(int(idx))

                        if sc_polys and sc_map:
                            results2 = self.surya_model(
                                images=[page_image],
                                polygons=[sc_polys],
                                recognition_batch_size=self.hybrid_options.surya_batch_size,
                            )
                            if results2 and getattr(results2[0], "text_lines", None) and len(results2[0].text_lines) == len(sc_map):
                                for j, tl in enumerate(list(results2[0].text_lines or [])):
                                    idx = sc_map[j]
                                    raw_txt = str(getattr(tl, "text", "") or "")
                                    sc_cand2_raw[idx] = raw_txt
                                    sc_cand2[idx] = _sanitize_surya_text(raw_txt)

            # Update cells with acceptance gates
            for idx in range(len(cell_polys)):
                cell = cell_polys[idx][0]
                original_text = str(cell.text or "")
                new_text_raw = cand_raw[idx]
                new_text = cand[idx]

                in_table_region = bool(getattr(cell, "_in_table_region", False))
                routed_min_num_conf = getattr(cell, "_min_numeric_token_conf", None)
                col_num_ratio = float(getattr(cell, "_table_col_numeric_ratio", 0.0) or 0.0)
                sc_text2 = sc_cand2.get(idx)
                sc_required_here = bool(sc_required.get(idx))
                sc_agree = True
                if sc_required_here:
                    sc_agree = False
                    if isinstance(sc_text2, str) and sc_text2:
                        try:
                            n1 = _normalize_numeric_replacement(baseline=original_text, candidate=new_text)
                            n2 = _normalize_numeric_replacement(baseline=original_text, candidate=sc_text2)
                            sc_agree = _numeric_signature(n1) == _numeric_signature(n2)
                        except Exception:
                            sc_agree = False

                base_num_like, base_header_num, _ = numeric_likeness(original_text)
                cand_num_like, cand_header_num, _ = numeric_likeness(new_text)

                garbled_ctx = bool(getattr(cell, "_garbled_region", False)) or bool(getattr(cell, "_garbled_cell", False))
                try:
                    col_digit_median = int(getattr(cell, "_table_col_digit_median"))
                except Exception:
                    col_digit_median = None

                col_dh_ratio = float(getattr(cell, "_table_col_digit_heavy_ratio", 0.0) or 0.0)
                col_ratio_eff = max(col_num_ratio, col_dh_ratio)

                try:
                    col_thr = float(getattr(self.hybrid_options, "numeric_context_min_col_ratio", 0.60))
                except Exception:
                    col_thr = 0.60

                numeric_context_col = bool(in_table_region and (col_ratio_eff >= col_thr))
                # Treat strongly-numeric table columns as numeric context even if
                # numeric_likeness fails on garbled strings. This ensures we never
                # accept digit+letter junk (e.g. "876.AET") as a table numeric.
                effective_numeric = bool(
                    (base_num_like and (not base_header_num))
                    or (
                        numeric_context_col
                        and (not _looks_like_short_alnum_code(original_text))
                        and ((not base_header_num) or garbled_ctx)
                    )
                )

                if effective_numeric:
                    new_text = _normalize_numeric_replacement(baseline=original_text, candidate=new_text)

                    # In numeric context, only accept strictly numeric candidates.
                    # This is critical for garbled pages where Surya may emit digit+letter junk (e.g. "876.AET").
                    strict_ok = _is_strict_numeric_candidate(new_text)
                    if garbled_ctx:
                        strict_ok = _is_strict_numeric_candidate_garbled(new_text)
                    if not strict_ok:
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

                # Fail-closed: in garbled contexts or strongly-numeric table contexts, reject digit+alpha junk.
                # This blocks Surya artifacts like "876.AET" that may otherwise pass text plausibility.
                if garbled_ctx or base_num_like or (in_table_region and (col_ratio_eff >= col_thr)):
                    cand_compact = _compact_ws(new_text)
                    if cand_compact:
                        cand_digs = sum(ch.isdigit() for ch in cand_compact)
                        cand_alpha = sum(ch.isalpha() for ch in cand_compact)
                        if (cand_digs >= int(getattr(self.hybrid_options, "digit_alpha_junk_min_digits", 3))) and (cand_alpha >= 1):
                            dig_ratio = cand_digs / max(1, len(cand_compact))
                            try:
                                min_dr = float(getattr(self.hybrid_options, "digit_alpha_junk_min_digit_ratio", 0.45))
                            except Exception:
                                min_dr = 0.45
                            if (dig_ratio >= min_dr) and (not cand_header_num) and (not base_header_num):
                                # If it's a permitted numeric form (e.g. "123USD"), do not treat as junk.
                                allowed = _is_strict_numeric_candidate(new_text) or _is_strict_numeric_candidate_garbled(new_text)
                                if not allowed:
                                    self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                                    if isinstance(self._last_update_diffs, list):
                                        self._last_update_diffs.append(
                                            {
                                                "bbox": _bbox_obj(cell),
                                                "baseline": original_text,
                                                "candidate": new_text,
                                                "candidate_raw": new_text_raw,
                                                "accepted": False,
                                                "reason": "skip_digit_alpha_junk",
                                            }
                                        )
                                    continue

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

                # Universal semantic safety: never accept numeric-looking tokens with unexpected alpha suffixes.
                # This catches high-risk Surya artifacts even when table inference / numeric context is ambiguous.
                if (not cand_header_num) and (not base_header_num) and _looks_like_numeric_with_alpha_suffix_junk(new_text):
                    self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                    if isinstance(self._last_update_diffs, list):
                        self._last_update_diffs.append(
                            {
                                "bbox": _bbox_obj(cell),
                                "baseline": original_text,
                                "candidate": new_text,
                                "candidate_raw": new_text_raw,
                                "accepted": False,
                                "reason": "skip_numeric_alpha_suffix",
                            }
                        )
                    continue

                # Semantic guardrail: never rewrite short alnum codes as numbers unless the column
                # is predominantly numeric and the rewrite is a plausible lookalike correction.
                if _looks_like_short_alnum_code(original_text) and _is_strict_numeric_candidate(new_text):
                    try:
                        col_thr = float(getattr(self.hybrid_options, "numeric_column_min_ratio_for_alnum_to_numeric", 0.70))
                    except Exception:
                        col_thr = 0.70
                    if (not in_table_region) or (col_num_ratio < col_thr):
                        self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                        if isinstance(self._last_update_diffs, list):
                            self._last_update_diffs.append(
                                {
                                    "bbox": _bbox_obj(cell),
                                    "baseline": original_text,
                                    "candidate": new_text,
                                    "candidate_raw": new_text_raw,
                                    "accepted": False,
                                    "reason": "skip_alnum_to_numeric_non_numeric_col",
                                }
                            )
                        continue

                    # Require lookalike-consistent digits (e.g., S80 -> 580). If this doesn't hold,
                    # the rewrite is likely a semantic change (code/label -> number).
                    bd = _digits_only(_translate_ocr_lookalikes(original_text))
                    cd = _digits_only(new_text)
                    if (not bd) or (bd != cd):
                        self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                        if isinstance(self._last_update_diffs, list):
                            self._last_update_diffs.append(
                                {
                                    "bbox": _bbox_obj(cell),
                                    "baseline": original_text,
                                    "candidate": new_text,
                                    "candidate_raw": new_text_raw,
                                    "accepted": False,
                                    "reason": "skip_alnum_to_numeric_not_lookalike",
                                }
                            )
                        continue

                    if sc_required_here and (not sc_agree):
                        self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                        if isinstance(self._last_update_diffs, list):
                            self._last_update_diffs.append(
                                {
                                    "bbox": _bbox_obj(cell),
                                    "baseline": original_text,
                                    "candidate": new_text,
                                    "candidate_raw": new_text_raw,
                                    "accepted": False,
                                    "reason": "skip_numeric_self_inconsistent",
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
                            # Exceptions for high-confidence baselines:
                            # - tiny, high-value numeric fixes (single digit flip)
                            # - self-consistent two-pass numeric recognition
                            if (not _is_one_digit_substitution(original_text, new_text)) and (not sc_agree):
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
                    strict_ok = _is_strict_numeric_candidate(new_text)
                    if bool(getattr(cell, "_garbled_region", False)) or bool(getattr(cell, "_garbled_cell", False)):
                        strict_ok = _is_strict_numeric_candidate_garbled(new_text)

                    if not strict_ok:
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

                    # Additional digit-count plausibility in strongly numeric columns on garbled pages.
                    # This helps prevent accepting short garbage numerics (e.g., '789') in a column of long amounts.
                    if garbled_ctx and (col_ratio_eff >= 0.85) and (not _digit_count_plausible(new_text, median=col_digit_median)):
                        self._stats["surya_update_skipped_sanity"] = int(self._stats.get("surya_update_skipped_sanity", 0)) + 1
                        if isinstance(self._last_update_diffs, list):
                            self._last_update_diffs.append(
                                {
                                    "bbox": _bbox_obj(cell),
                                    "baseline": original_text,
                                    "candidate": new_text,
                                    "candidate_raw": new_text_raw,
                                    "accepted": False,
                                    "reason": "skip_numeric_digit_count_implausible",
                                }
                            )
                        continue

                # Non-numeric hardening when enabled: only accept small, "diacritic-like" corrections.
                if (not base_num_like) and bool(getattr(self.hybrid_options, "update_non_numeric", False)):
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
                    if nb and nc and (not bool(getattr(cell, "_garbled_cell", False))):
                        lcs = _lcs_len(nb, nc)
                        denom = max(1, min(len(nb), len(nc)))
                        try:
                            min_ratio = float(getattr(self.hybrid_options, "table_text_min_accent_stripped_lcs_ratio", 0.65))
                        except Exception:
                            min_ratio = 0.65

                        if bool(getattr(cell, "_garbled_region", False)) or bool(getattr(cell, "_garbled_cell", False)):
                            try:
                                min_ratio = float(
                                    getattr(
                                        self.hybrid_options,
                                        "garbled_text_min_accent_stripped_lcs_ratio",
                                        min_ratio,
                                    )
                                )
                            except Exception:
                                pass
 
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
                        0.0
                        if (not base_num_like)
                        and bool(getattr(self.hybrid_options, "update_non_numeric", False))
                        and bool(getattr(cell, "_garbled_cell", False))
                        else (
                            0.35
                            if (not base_num_like) and bool(getattr(self.hybrid_options, "update_non_numeric", False))
                            else float(getattr(self.hybrid_options, "min_normalized_lcs_ratio", 0.15))
                        )
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
                    obj: dict[str, Any] = {
                        "bbox": _bbox_obj(cell),
                        "baseline": original_text,
                        "candidate": new_text,
                        "candidate_raw": new_text_raw,
                        "accepted": True,
                        "reason": "updated",
                    }
                    if sc_required_here:
                        obj["candidate2"] = str(sc_text2 or "")
                        obj["candidate2_raw"] = str(sc_cand2_raw.get(idx) or "")
                    self._last_update_diffs.append(obj)

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
                            # Store Tesseract TSV bbox in the region image coordinate system.
                            # This is what Surya needs when `page_image` is a cropped OCR region.
                            setattr(cell, "_region_bbox", bbox)

                            if block_num > 0 and par_num > 0 and line_num > 0:
                                setattr(cell, "_tsv_line_key", (block_num, par_num, line_num))
                            region_cells.append(cell)

                        # HYBRID ROUTING: Filter and re-OCR
                        table_boxes = []
                        if region_cells:
                            line_min_num_conf = _build_line_min_numeric_conf(df_result)

                            garbled = bool(
                                getattr(self.hybrid_options, "enable_garble_rescue", True)
                            ) and bool(_tsv_looks_garbled(df_result))
                            if garbled:
                                self._stats["garbled_regions"] = int(self._stats.get("garbled_regions", 0)) + 1
                            for c in region_cells:
                                setattr(c, "_garbled_region", bool(garbled))
                                try:
                                    setattr(
                                        c,
                                        "_garbled_cell",
                                        bool(
                                            _cell_looks_garbled(
                                                str(getattr(c, "text", "") or ""),
                                                float(getattr(c, "confidence", 0.0) or 0.0),
                                            )
                                        ),
                                    )
                                except Exception:
                                    setattr(c, "_garbled_cell", False)

                            # Routing policy: by default, route only inside inferred table regions.
                            # Docling layout/table predictions are not available yet at OCR stage,
                            # so we infer table regions directly from the TSV words.
                            table_boxes = _infer_table_boxes_from_tsv(df_result)
                            self._stats["inferred_table_boxes"] += len(table_boxes)

                            # Garble rescue: when the TSV looks corrupted, fall back to treating the
                            # entire region as table-like so Surya can act as a second-pass recognizer.
                            if garbled and (not table_boxes):
                                try:
                                    from docling_core.types.doc.base import CoordOrigin

                                    boxes: list[BoundingBox] = []
                                    for _ix, row in df_result.iterrows():
                                        try:
                                            word_num = int(float(row.get("word_num") or 0))
                                        except Exception:
                                            word_num = 0
                                        if word_num <= 0:
                                            continue
                                        left = float(row.get("left") or 0.0)
                                        top = float(row.get("top") or 0.0)
                                        width = float(row.get("width") or 0.0)
                                        height = float(row.get("height") or 0.0)
                                        if width <= 0 or height <= 0:
                                            continue
                                        boxes.append(
                                            BoundingBox(
                                                l=left,
                                                t=top,
                                                r=left + width,
                                                b=top + height,
                                                coord_origin=CoordOrigin.TOPLEFT,
                                            )
                                        )
                                    u = _union_box(boxes)
                                    if u is not None:
                                        table_boxes = [u]
                                except Exception:
                                    pass

                            if not table_boxes:
                                if bool(getattr(self.hybrid_options, "route_table_only", True)):
                                    self._stats["skipped_no_table_clusters"] += len(region_cells)
                                else:
                                    # Table-preferred mode: allow routing even without clusters.
                                    pass


                            route_table_only = bool(getattr(self.hybrid_options, "route_table_only", True))
                            force_surya_in_tables = bool(getattr(self.hybrid_options, "force_surya_in_table_regions", False))
                            if garbled:
                                force_surya_in_tables = True

                            cells_to_reocr: List[TextCell] = []

                            # Column-type inference (numericness) for semantic guardrails.
                            try:
                                _annotate_table_column_numericness(region_cells, table_boxes)
                            except Exception:
                                pass
                            for c in region_cells:
                                # routing/table overlap must use the same coordinate system.
                                # - `table_boxes` are inferred from TSV word boxes (region-image pixel coords).
                                # - We store each cell's original TSV bbox as `c._region_bbox` (same system).
                                rb = getattr(c, "_region_bbox", None)
                                if rb is None:
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
                                # If line-level confidence is not low enough to trigger routing, fall back to
                                # a bbox-scoped min confidence over numeric-like TSV words. This helps catch
                                # cases where a single digit is misread with low confidence even when the
                                # overall cell confidence is high.
                                if (
                                    in_table
                                    and is_num_like
                                    and (not is_header_num)
                                    and (
                                        (min_num_conf is None)
                                        or (min_num_conf >= float(getattr(self.hybrid_options, "number_confidence_threshold", 0.85)))
                                    )
                                ):
                                    try:
                                        bbox_min_conf = _tesseract_word_min_conf_in_bbox(df_result, rb)
                                        if isinstance(bbox_min_conf, (int, float)):
                                            if (min_num_conf is None) or (float(bbox_min_conf) < float(min_num_conf)):
                                                min_num_conf = float(bbox_min_conf)
                                    except Exception:
                                        pass
                                setattr(c, "_min_numeric_token_conf", min_num_conf)

                                is_suspicious_num = False
                                if (
                                    in_table
                                    and is_num_like
                                    and (not is_header_num)
                                    and bool(getattr(self.hybrid_options, "route_suspicious_numeric", True))
                                ):
                                    is_suspicious_num = _is_suspicious_numeric_ocr(str(c.text or ""))

                                should_route = False
                                if in_table and force_surya_in_tables and (not is_header_num):
                                    should_route = True
                                else:
                                    # Ceiling-plan routing: if a numeric-like token looks suspicious, route it even if
                                    # Tesseract confidence is high (still capped to avoid extreme over-routing).
                                    if is_suspicious_num:
                                        try:
                                            cap = float(getattr(self.hybrid_options, "suspicious_numeric_confidence_cap", 0.99))
                                        except Exception:
                                            cap = 0.99
                                        should_route = float(getattr(c, "confidence", 0.0) or 0.0) <= cap
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
                                    if is_suspicious_num:
                                        self._stats["routed_suspicious_numeric"] = int(self._stats.get("routed_suspicious_numeric", 0)) + 1
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
            "routed_suspicious_numeric": int(self._stats.get("routed_suspicious_numeric", 0)),
            "skipped_no_table_clusters": int(self._stats.get("skipped_no_table_clusters", 0)),
            "skipped_header_numeric": int(self._stats.get("skipped_header_numeric", 0)),
            "skipped_missing_region_bbox": int(self._stats.get("skipped_missing_region_bbox", 0)),
            "inferred_table_boxes": int(self._stats.get("inferred_table_boxes", 0)),
            "garbled_regions": int(self._stats.get("garbled_regions", 0)),
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
            "routed_suspicious_numeric": 0,
            "skipped_no_table_clusters": 0,
            "skipped_header_numeric": 0,
            "skipped_missing_region_bbox": 0,
            "inferred_table_boxes": 0,
            "garbled_regions": 0,
            "surya_cells_updated": 0,
            "surya_update_skipped_sanity": 0,
            "surya_update_skipped_count_mismatch": 0,
            "surya_update_skipped_non_numeric": 0,
            "surya_failures": 0,
        }

    
    @classmethod
    def get_options_type(cls) -> Type[TesseractCliOcrOptions]:
        return HybridOcrOptions
