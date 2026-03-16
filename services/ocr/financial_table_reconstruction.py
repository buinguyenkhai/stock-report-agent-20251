from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_NUMERIC_RE = re.compile(r"^[\(\-+]?\d[\d.,]*\)?%?$")
_CODE_RE = re.compile(r"^(?:[A-Za-z]{0,3}\d{1,4}|[IVXLCDM]{1,8}|\d{1,4})\.?$", re.IGNORECASE)
_NOTE_RE = re.compile(r"^\d+(?:\.\d+)*$")
_HEADER_HINT_RE = re.compile(
    r"(?:\b20\d{2}\b|\bquy\b|\bnam\b|\bthang\b|\bngay\b|\bluy ke\b|/|gia tri|\bky\b|\bthuyet minh\b|\btrieu\b|\bvnd\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OcrWord:
    text: str
    left: float
    top: float
    width: float
    height: float
    conf: float
    line_key: Optional[Tuple[int, int, int]] = None
    source_tag: str = "baseline"

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def x_center(self) -> float:
        return self.left + self.width / 2.0

    @property
    def y_center(self) -> float:
        return self.top + self.height / 2.0


@dataclass(frozen=True)
class RegionBox:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class LineRow:
    words: List[OcrWord]

    @property
    def top(self) -> float:
        return min(word.top for word in self.words)

    @property
    def bottom(self) -> float:
        return max(word.bottom for word in self.words)

    @property
    def height(self) -> float:
        return max(1.0, self.bottom - self.top)

    @property
    def text(self) -> str:
        return " ".join(word.text for word in sorted(self.words, key=lambda w: (w.left, w.top))).strip()

    @property
    def numeric_word_count(self) -> int:
        return sum(1 for word in self.words if _is_numeric_like(word.text))

    @property
    def numeric_words(self) -> List[OcrWord]:
        return [word for word in self.words if _is_numeric_like(word.text)]


def _normalize_ascii(text: str) -> str:
    base = unicodedata.normalize("NFD", str(text or ""))
    stripped = "".join(ch for ch in base if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", stripped).strip().lower()


def _is_numeric_like(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    return bool(compact and _NUMERIC_RE.match(compact))


def _escape_md(text: str) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", "<br>")


def _word(
    text: str,
    left: float,
    top: float,
    right: float,
    bottom: float,
    conf: float,
    line_key: Any,
    source_tag: str = "baseline",
) -> OcrWord:
    key = tuple(line_key) if isinstance(line_key, (list, tuple)) and len(line_key) == 3 else None
    return OcrWord(
        text=str(text or "").strip(),
        left=float(left),
        top=float(top),
        width=max(0.0, float(right) - float(left)),
        height=max(0.0, float(bottom) - float(top)),
        conf=float(conf),
        line_key=key,
        source_tag=str(source_tag or "baseline"),
    )


def _token_to_word(token: Dict[str, Any]) -> Optional[OcrWord]:
    text = str(token.get("text") or "").strip()
    if not text:
        return None
    try:
        left = float(token.get("left"))
        top = float(token.get("top"))
        right = float(token.get("right"))
        bottom = float(token.get("bottom"))
    except Exception:
        return None
    if right <= left or bottom <= top:
        return None
    return _word(
        text=text,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        conf=float(token.get("confidence", 0.0) or 0.0),
        line_key=token.get("line_key"),
        source_tag=str(token.get("source_tag") or "baseline"),
    )


def _coerce_region(box: Dict[str, Any]) -> Optional[RegionBox]:
    try:
        left = float(box.get("left"))
        top = float(box.get("top"))
        right = float(box.get("right"))
        bottom = float(box.get("bottom"))
    except Exception:
        return None
    if right <= left or bottom <= top:
        return None
    return RegionBox(left=left, top=top, right=right, bottom=bottom)


def _cluster_positions(values: Sequence[float], tolerance: float) -> List[float]:
    if not values:
        return []
    ordered = sorted(float(v) for v in values)
    clusters: List[List[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _group_lines(words: Iterable[OcrWord]) -> List[LineRow]:
    ordered = sorted(words, key=lambda word: (word.top, word.left))
    if not ordered:
        return []

    keyed = [word for word in ordered if word.line_key is not None]
    if len(keyed) >= max(3, int(0.6 * len(ordered))):
        grouped: Dict[Tuple[int, int, int], List[OcrWord]] = {}
        for word in ordered:
            key = word.line_key if word.line_key is not None else (0, 0, int(round(word.top)))
            grouped.setdefault(key, []).append(word)
        return sorted(
            [LineRow(sorted(line_words, key=lambda w: (w.left, w.top))) for line_words in grouped.values()],
            key=lambda line: (line.top, min(word.left for word in line.words)),
        )

    heights = sorted(word.height for word in ordered)
    median_height = heights[len(heights) // 2] if heights else 12.0
    tolerance = max(8.0, median_height * 0.55)
    lines: List[List[OcrWord]] = [[ordered[0]]]
    current_top = ordered[0].top
    for word in ordered[1:]:
        if abs(word.top - current_top) <= tolerance:
            lines[-1].append(word)
            current_top = min(current_top, word.top)
        else:
            lines.append([word])
            current_top = word.top
    return [LineRow(sorted(line_words, key=lambda w: (w.left, w.top))) for line_words in lines]


def _word_in_region(word: OcrWord, region: RegionBox, *, margin: float = 0.0) -> bool:
    return not (
        word.right < region.left - margin
        or word.left > region.right + margin
        or word.bottom < region.top - margin
        or word.top > region.bottom + margin
    )


def _source_tag_counts(words: Sequence[OcrWord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for word in words:
        tag = str(word.source_tag or "baseline")
        counts[tag] = counts.get(tag, 0) + 1
    return counts


def _score_region(region: RegionBox, words: Sequence[OcrWord], *, page_size: Tuple[int, int]) -> Tuple[float, Dict[str, Any]]:
    region_words = [word for word in words if _word_in_region(word, region, margin=4.0)]
    if not region_words:
        return 0.0, {"word_count": 0}
    lines = _group_lines(region_words)
    numeric_words = sum(1 for word in region_words if _is_numeric_like(word.text))
    numeric_lines = sum(1 for line in lines if line.numeric_word_count > 0)
    header_lines = sum(1 for line in lines if line.numeric_word_count == 0 and _HEADER_HINT_RE.search(line.text or ""))
    numeric_centers = _cluster_positions(
        [word.x_center for line in lines for word in line.numeric_words],
        tolerance=max(16.0, region.width * 0.022),
    )
    area_fraction = 0.0
    page_area = max(1.0, float(page_size[0]) * float(page_size[1]))
    area_fraction = region.area / page_area
    score = (
        numeric_lines * 6.0
        + numeric_words * 1.5
        + min(len(numeric_centers), 8) * 8.0
        + min(header_lines, 3) * 4.0
        + min(len(lines), 20) * 1.0
        - area_fraction * 6.0
    )
    return score, {
        "word_count": len(region_words),
        "line_count": len(lines),
        "numeric_word_count": numeric_words,
        "numeric_line_count": numeric_lines,
        "header_line_count": header_lines,
        "numeric_column_hint_count": len(numeric_centers),
        "source_tag_counts": _source_tag_counts(region_words),
    }


def _infer_candidate_regions(words: Sequence[OcrWord], page_size: Tuple[int, int]) -> List[RegionBox]:
    lines = _group_lines(words)
    if not lines:
        return []
    numeric_indices = [idx for idx, line in enumerate(lines) if line.numeric_word_count > 0]
    if not numeric_indices:
        return []

    groups: List[List[int]] = [[numeric_indices[0]]]
    for idx in numeric_indices[1:]:
        if idx - groups[-1][-1] <= 3:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    regions: List[RegionBox] = []
    margin_x = max(12.0, page_size[0] * 0.015)
    margin_y = max(10.0, page_size[1] * 0.01)
    for group in groups:
        start_idx = max(0, group[0] - 2)
        end_idx = min(len(lines) - 1, group[-1] + 1)
        band_words = [word for line in lines[start_idx : end_idx + 1] for word in line.words]
        if not band_words:
            continue
        regions.append(
            RegionBox(
                left=max(0.0, min(word.left for word in band_words) - margin_x),
                top=max(0.0, min(word.top for word in band_words) - margin_y),
                right=min(float(page_size[0]), max(word.right for word in band_words) + margin_x),
                bottom=min(float(page_size[1]), max(word.bottom for word in band_words) + margin_y),
            )
        )
    return regions


def _select_table_region(
    words: Sequence[OcrWord],
    *,
    page_size: Tuple[int, int],
    table_regions: Optional[Sequence[Dict[str, Any]]] = None,
    ocr_regions: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[Optional[RegionBox], Dict[str, Any]]:
    docling_candidates: List[Tuple[RegionBox, float, Dict[str, Any]]] = []
    ocr_candidates: List[Tuple[RegionBox, float, Dict[str, Any]]] = []
    inferred_candidates: List[Tuple[RegionBox, float, Dict[str, Any]]] = []
    for raw_region in table_regions or []:
        region = _coerce_region(raw_region)
        if region is None:
            continue
        score, summary = _score_region(region, words, page_size=page_size)
        docling_candidates.append((region, score, {"region_source": "docling_table_region", **summary}))
    for raw_region in ocr_regions or []:
        region = _coerce_region(raw_region)
        if region is None:
            continue
        score, summary = _score_region(region, words, page_size=page_size)
        ocr_candidates.append((region, score, {"region_source": "ocr_region", **summary}))

    inferred_regions = _infer_candidate_regions(words, page_size)
    for region in inferred_regions:
        score, summary = _score_region(region, words, page_size=page_size)
        inferred_candidates.append((region, score, {"region_source": "inferred_numeric_band", **summary}))

    candidates = docling_candidates + ocr_candidates + inferred_candidates
    if not candidates:
        return None, {"selection_mode": "none", "candidate_count": 0}

    if any(score > 0.0 for _region, score, _summary in docling_candidates):
        preferred = docling_candidates
    elif any(score > 0.0 for _region, score, _summary in ocr_candidates):
        preferred = ocr_candidates
    else:
        preferred = candidates
    best_region, best_score, best_summary = max(preferred, key=lambda item: item[1])
    if best_score <= 0.0:
        return None, {
            "selection_mode": "none",
            "candidate_count": len(candidates),
            "best_score": float(best_score),
        }
    return best_region, {
        "selection_mode": str(best_summary.get("region_source") or "unknown"),
        "candidate_count": len(candidates),
        "best_score": float(best_score),
        **best_summary,
    }


def _append_header_band(
    header_bands: List[List[str]],
    words: Sequence[OcrWord],
    numeric_centers: Sequence[float],
    *,
    tolerance: float,
) -> None:
    if not numeric_centers:
        return
    band = [""] * len(numeric_centers)
    for word in sorted(words, key=lambda w: (w.left, w.top)):
        if word.x_center < numeric_centers[0] - tolerance:
            continue
        nearest_idx = min(range(len(numeric_centers)), key=lambda idx: abs(word.x_center - numeric_centers[idx]))
        if abs(word.x_center - numeric_centers[nearest_idx]) > tolerance * 1.8:
            continue
        band[nearest_idx] = f"{band[nearest_idx]} {word.text}".strip()
    if any(cell.strip() for cell in band):
        header_bands.append(band)


def _build_column_labels(header_bands: Sequence[Sequence[str]], column_count: int) -> List[str]:
    labels: List[str] = []
    for idx in range(column_count):
        parts: List[str] = []
        seen: set[str] = set()
        for band in header_bands:
            value = str(band[idx] if idx < len(band) else "").strip()
            if not value:
                continue
            key = _normalize_ascii(value)
            if key in seen:
                continue
            seen.add(key)
            parts.append(value)
        labels.append(" ".join(parts).strip() or f"Giá trị {idx + 1}")
    return labels


def _build_header_flags(header_lines: Sequence[LineRow], first_numeric_center: float, tolerance: float) -> Tuple[bool, bool]:
    saw_code = False
    saw_note = False
    for line in header_lines:
        left_text = " ".join(
            word.text
            for word in sorted(line.words, key=lambda w: (w.left, w.top))
            if word.right < first_numeric_center - tolerance
        )
        norm = _normalize_ascii(left_text)
        if "ma so" in norm:
            saw_code = True
        if "thuyet minh" in norm:
            saw_note = True
    return saw_code, saw_note


def _line_is_header_candidate(line: LineRow) -> bool:
    norm = _normalize_ascii(line.text)
    if any(key in norm for key in ("ma so", "chi tieu", "thuyet minh", "luy ke", "nam nay", "nam truoc")):
        return True
    numeric_tokens = [re.sub(r"[^\d]", "", word.text) for word in line.numeric_words]
    if numeric_tokens and all(re.fullmatch(r"20\d{2}", token or "") for token in numeric_tokens):
        return True
    first_token = str(line.words[0].text or "").strip() if line.words else ""
    if first_token and not _CODE_RE.match(first_token) and _HEADER_HINT_RE.search(norm):
        return True
    return False


def _merge_continuation_row(previous: Dict[str, Any], line: LineRow) -> None:
    text = line.text.strip()
    if not text:
        return
    previous["label"] = f"{previous['label']} {text}".strip()
    previous["continuation_lines"] = int(previous.get("continuation_lines", 0) or 0) + 1


def _join_text(words: Sequence[OcrWord]) -> str:
    return " ".join(word.text for word in sorted(words, key=lambda w: (w.left, w.top))).strip()


def _split_auxiliary_left_numeric_cluster(
    numeric_centers: Sequence[float],
    *,
    region: RegionBox,
) -> Tuple[Optional[float], List[float]]:
    centers = [float(center) for center in numeric_centers]
    if len(centers) < 3:
        return None, centers
    first_gap = centers[1] - centers[0]
    if centers[0] < region.left + region.width * 0.35 and first_gap > max(32.0, region.width * 0.08):
        return centers[0], centers[1:]
    return None, centers


def reconstruct_table_from_words(
    words: Sequence[OcrWord],
    *,
    image_size: Tuple[int, int],
    table_regions: Optional[Sequence[Dict[str, Any]]] = None,
    ocr_regions: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    if not words:
        return "", {"reconstruction_applied": False, "reason": "no_words"}

    region, region_debug = _select_table_region(
        words,
        page_size=image_size,
        table_regions=table_regions,
        ocr_regions=ocr_regions,
    )
    if region is None:
        return "", {"reconstruction_applied": False, "reason": "no_table_region", **region_debug}

    region_words = [word for word in words if _word_in_region(word, region, margin=4.0)]
    if not region_words:
        return "", {"reconstruction_applied": False, "reason": "empty_region_words", **region_debug}

    lines = _group_lines(region_words)
    if not lines:
        return "", {"reconstruction_applied": False, "reason": "no_lines", **region_debug}

    header_lines: List[LineRow] = []
    body_start_idx = 0
    for idx, line in enumerate(lines):
        if _line_is_header_candidate(line):
            header_lines.append(line)
            body_start_idx = idx + 1
            continue
        if line.numeric_word_count == 0 and not header_lines:
            header_lines.append(line)
            body_start_idx = idx + 1
            continue
        break

    body_lines = lines[body_start_idx:]
    if not body_lines:
        body_lines = lines
        header_lines = []

    data_line_indices = [idx for idx, line in enumerate(body_lines) if line.numeric_word_count > 0]
    if not data_line_indices:
        return "", {"reconstruction_applied": False, "reason": "no_numeric_lines", **region_debug}

    numeric_centers = _cluster_positions(
        [word.x_center for line in body_lines for word in line.numeric_words if word.x_center >= region.left + region.width * 0.25],
        tolerance=max(18.0, region.width * 0.022),
    )
    if not numeric_centers:
        numeric_centers = _cluster_positions(
            [word.x_center for line in body_lines for word in line.numeric_words],
            tolerance=max(18.0, region.width * 0.022),
        )
    if not numeric_centers:
        return "", {"reconstruction_applied": False, "reason": "no_numeric_columns", **region_debug}
    note_anchor, numeric_centers = _split_auxiliary_left_numeric_cluster(numeric_centers, region=region)
    if not numeric_centers:
        return "", {"reconstruction_applied": False, "reason": "no_numeric_columns", **region_debug}

    numeric_tolerance = max(20.0, region.width * 0.025)
    header_bands: List[List[str]] = []
    for line in header_lines:
        _append_header_band(header_bands, line.words, numeric_centers, tolerance=numeric_tolerance)

    saw_code_column, saw_note_column = _build_header_flags(header_lines, numeric_centers[0], numeric_tolerance)
    if note_anchor is not None:
        saw_note_column = True
    rows: List[Dict[str, Any]] = []

    for line in body_lines:
        numeric_cells = [""] * len(numeric_centers)
        label_words: List[OcrWord] = []
        between_words: List[OcrWord] = []

        for word in sorted(line.words, key=lambda w: (w.left, w.top)):
            nearest_idx = min(range(len(numeric_centers)), key=lambda idx: abs(word.x_center - numeric_centers[idx]))
            distance = abs(word.x_center - numeric_centers[nearest_idx])
            if _is_numeric_like(word.text) and distance <= numeric_tolerance:
                if note_anchor is not None and abs(word.x_center - note_anchor) <= numeric_tolerance:
                    between_words.append(word)
                else:
                    numeric_cells[nearest_idx] = f"{numeric_cells[nearest_idx]} {word.text}".strip()
            elif word.right < numeric_centers[0] - numeric_tolerance:
                label_words.append(word)
            else:
                between_words.append(word)

        if not any(numeric_cells):
            if rows and _join_text(line.words):
                _merge_continuation_row(rows[-1], line)
            continue

        code = ""
        note = ""
        label_tokens = [word.text for word in label_words]
        if label_tokens and _CODE_RE.match(label_tokens[0]) and len(label_tokens) > 1:
            code = label_tokens.pop(0)
            saw_code_column = True

        trailing_tokens = [word.text for word in between_words]
        if trailing_tokens and _NOTE_RE.match(trailing_tokens[-1]):
            note = trailing_tokens[-1]
            trailing_tokens = trailing_tokens[:-1]
            saw_note_column = True

        label = " ".join(label_tokens + trailing_tokens).strip()
        if not label and code:
            label = code

        rows.append(
            {
                "code": code,
                "label": label,
                "note": note,
                "values": numeric_cells,
                "continuation_lines": 0,
            }
        )

    if not rows:
        return "", {"reconstruction_applied": False, "reason": "no_rows", **region_debug}

    value_headers = _build_column_labels(header_bands, len(numeric_centers))
    header: List[str] = []
    if saw_code_column:
        header.append("Mã số")
    header.append("Chỉ tiêu")
    if saw_note_column:
        header.append("Thuyết minh")
    header.extend(value_headers)

    lines_out = [
        "| " + " | ".join(_escape_md(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        cells: List[str] = []
        if saw_code_column:
            cells.append(str(row.get("code") or ""))
        cells.append(str(row.get("label") or ""))
        if saw_note_column:
            cells.append(str(row.get("note") or ""))
        cells.extend(str(v or "") for v in row.get("values", []))
        if len(cells) < len(header):
            cells.extend([""] * (len(header) - len(cells)))
        lines_out.append("| " + " | ".join(_escape_md(cell) for cell in cells[: len(header)]) + " |")

    inferred_column_anchors = [round(center, 2) for center in numeric_centers]
    return "\n".join(lines_out).strip(), {
        "reconstruction_applied": True,
        "table_region": {
            "left": region.left,
            "top": region.top,
            "right": region.right,
            "bottom": region.bottom,
        },
        "numeric_column_count": len(numeric_centers),
        "reconstructed_row_count": len(rows),
        "line_count": len(lines),
        "header_line_count": len(header_lines),
        "header_band_count": len(header_bands),
        "header_present": bool(header_bands),
        "working_line_count": len(body_lines),
        "page_size": [int(image_size[0]), int(image_size[1])],
        "inferred_column_anchors": inferred_column_anchors,
        "note_anchor": round(note_anchor, 2) if note_anchor is not None else None,
        "source_tag_counts": _source_tag_counts(region_words),
        **region_debug,
    }


def reconstruct_table_from_tokens(
    tokens: Sequence[Dict[str, Any]],
    *,
    page_size: Tuple[int, int],
    table_regions: Optional[Sequence[Dict[str, Any]]] = None,
    ocr_regions: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    words = [word for token in tokens for word in [_token_to_word(token)] if word is not None]
    markdown, debug = reconstruct_table_from_words(
        words,
        image_size=page_size,
        table_regions=table_regions,
        ocr_regions=ocr_regions,
    )
    debug["token_count"] = len(words)
    return markdown, debug


def reconstruct_financial_table_markdown(
    page_image: str | None = None,
    *,
    ocr_tokens: Optional[Sequence[Dict[str, Any]]] = None,
    page_size: Optional[Tuple[int, int]] = None,
    table_regions: Optional[Sequence[Dict[str, Any]]] = None,
    ocr_regions: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    if not ocr_tokens:
        return "", {"reconstruction_applied": False, "reason": "no_ocr_tokens", "page_image_path": page_image}
    if page_size is None:
        return "", {"reconstruction_applied": False, "reason": "missing_page_size", "page_image_path": page_image}

    markdown, debug = reconstruct_table_from_tokens(
        ocr_tokens,
        page_size=page_size,
        table_regions=table_regions,
        ocr_regions=ocr_regions,
    )
    debug["page_image_path"] = page_image
    return markdown, debug
