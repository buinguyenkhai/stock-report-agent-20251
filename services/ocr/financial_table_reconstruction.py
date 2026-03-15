from __future__ import annotations

import csv
import io
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageOps


_NUMERIC_RE = re.compile(r"^[\(\-+]?\d[\d.,]*\)?%?$")
_CODE_RE = re.compile(r"^(?:[A-Za-z]{0,3}\d{1,4}|[IVXLCDM]{1,8}|\d{1,4})\.?$", re.IGNORECASE)
_NOTE_RE = re.compile(r"^\d+(?:\.\d+)*$")


@dataclass(frozen=True)
class OcrWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float
    line_key: Tuple[int, int, int]

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def x_center(self) -> float:
        return self.left + self.width / 2.0


@dataclass
class LineRow:
    words: List[OcrWord]

    @property
    def top(self) -> int:
        return min(word.top for word in self.words)

    @property
    def bottom(self) -> int:
        return max(word.bottom for word in self.words)

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)

    @property
    def text(self) -> str:
        return " ".join(word.text for word in sorted(self.words, key=lambda w: (w.left, w.top))).strip()


def _is_numeric_like(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    return bool(compact and _NUMERIC_RE.match(compact))


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


def _escape_md(text: str) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", "<br>")


def _parse_tsv_words(tsv_text: str) -> List[OcrWord]:
    rows: List[OcrWord] = []
    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
    for rec in reader:
        text = str(rec.get("text") or "").strip()
        if not text:
            continue
        try:
            conf = float(str(rec.get("conf") or "-1"))
        except Exception:
            conf = -1.0
        if conf < 0:
            continue
        try:
            left = int(float(str(rec.get("left") or "0")))
            top = int(float(str(rec.get("top") or "0")))
            width = int(float(str(rec.get("width") or "0")))
            height = int(float(str(rec.get("height") or "0")))
            block_num = int(float(str(rec.get("block_num") or "0")))
            par_num = int(float(str(rec.get("par_num") or "0")))
            line_num = int(float(str(rec.get("line_num") or "0")))
        except Exception:
            continue
        if width <= 0 or height <= 0:
            continue
        rows.append(
            OcrWord(
                text=text,
                left=left,
                top=top,
                width=width,
                height=height,
                conf=conf,
                line_key=(block_num, par_num, line_num),
            )
        )
    return rows


def _run_tesseract_tsv(image_path: Path, *, lang: str = "vie+eng", psm: int = 6) -> List[OcrWord]:
    command = [
        "tesseract",
        str(image_path),
        "stdout",
        "--psm",
        str(psm),
        "-l",
        lang,
        "tsv",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=True)
    return _parse_tsv_words(proc.stdout)


def _group_lines(words: Iterable[OcrWord]) -> List[LineRow]:
    grouped: Dict[Tuple[int, int, int], List[OcrWord]] = {}
    for word in words:
        grouped.setdefault(word.line_key, []).append(word)
    lines = [LineRow(sorted(line_words, key=lambda w: (w.left, w.top))) for line_words in grouped.values()]
    return sorted(lines, key=lambda line: (line.top, min(word.left for word in line.words)))


def _assign_numeric_columns(lines: Sequence[LineRow], image_width: int) -> Tuple[List[float], Dict[int, List[str]]]:
    numeric_centers: List[float] = []
    for line in lines:
        for word in line.words:
            if _is_numeric_like(word.text):
                numeric_centers.append(word.x_center)

    tolerance = max(18.0, image_width * 0.018)
    centers = _cluster_positions(numeric_centers, tolerance=tolerance)
    line_assignments: Dict[int, List[str]] = {}
    return centers, line_assignments


def reconstruct_table_from_words(words: Sequence[OcrWord], *, image_size: Tuple[int, int]) -> Tuple[str, Dict[str, Any]]:
    if not words:
        return "", {"reconstruction_applied": False, "reason": "no_words"}

    image_width, image_height = image_size
    lines = _group_lines(words)
    if not lines:
        return "", {"reconstruction_applied": False, "reason": "no_lines"}

    candidate_lines = [line for line in lines if any(_is_numeric_like(word.text) for word in line.words)]
    if not candidate_lines:
        return "", {"reconstruction_applied": False, "reason": "no_numeric_lines"}

    first_idx = lines.index(candidate_lines[0])
    last_idx = lines.index(candidate_lines[-1])
    working_lines = lines[first_idx : last_idx + 1]

    all_numeric_centers = [
        word.x_center
        for line in working_lines
        for word in line.words
        if _is_numeric_like(word.text)
    ]
    right_side_numeric_centers = [center for center in all_numeric_centers if center >= image_width * 0.35]
    numeric_centers = _cluster_positions(
        right_side_numeric_centers or all_numeric_centers,
        tolerance=max(18.0, image_width * 0.018),
    )
    if not numeric_centers:
        return "", {"reconstruction_applied": False, "reason": "no_numeric_columns"}

    numeric_tolerance = max(20.0, image_width * 0.02)
    rows: List[List[str]] = []
    numeric_header_parts: List[List[str]] = [[] for _ in numeric_centers]
    saw_code_column = False
    saw_note_column = False

    for line in working_lines:
        numeric_cells = [""] * len(numeric_centers)
        label_words: List[OcrWord] = []
        trailing_words: List[OcrWord] = []
        for word in line.words:
            nearest_idx = min(
                range(len(numeric_centers)),
                key=lambda idx: abs(word.x_center - numeric_centers[idx]),
            )
            if _is_numeric_like(word.text) and abs(word.x_center - numeric_centers[nearest_idx]) <= numeric_tolerance:
                numeric_cells[nearest_idx] = f"{numeric_cells[nearest_idx]} {word.text}".strip()
            elif word.right < numeric_centers[0] - numeric_tolerance:
                label_words.append(word)
            else:
                trailing_words.append(word)

        label_words = sorted(label_words, key=lambda w: (w.left, w.top))
        code = ""
        note = ""
        label_tokens = [word.text for word in label_words]
        if label_tokens and _CODE_RE.match(label_tokens[0]) and len(label_tokens) > 1:
            code = label_tokens.pop(0)
            saw_code_column = True
        if trailing_words:
            trailing_tokens = [word.text for word in sorted(trailing_words, key=lambda w: (w.left, w.top))]
            if trailing_tokens and _NOTE_RE.match(trailing_tokens[-1]):
                note = trailing_tokens[-1]
                trailing_tokens = trailing_tokens[:-1]
                saw_note_column = True
            label_tokens.extend(trailing_tokens)

        label = " ".join(label_tokens).strip()
        has_numeric = any(cell for cell in numeric_cells)
        if not label and not has_numeric:
            continue

        if not has_numeric:
            for idx, token in enumerate(label_tokens):
                if idx < len(numeric_header_parts):
                    numeric_header_parts[idx].append(token)
            continue

        row: List[str] = []
        if saw_code_column:
            row.append(code)
        row.append(label)
        if saw_note_column:
            row.append(note)
        row.extend(numeric_cells)
        rows.append(row)

    if not rows:
        return "", {"reconstruction_applied": False, "reason": "no_rows"}

    header: List[str] = []
    if saw_code_column:
        header.append("Mã số")
    header.append("Chỉ tiêu")
    if saw_note_column:
        header.append("Thuyết minh")
    for idx, parts in enumerate(numeric_header_parts):
        label = " ".join(parts).strip()
        header.append(label or f"Giá trị {idx + 1}")

    lines_out = [
        "| " + " | ".join(_escape_md(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        lines_out.append("| " + " | ".join(_escape_md(cell) for cell in row[: len(header)]) + " |")

    markdown = "\n".join(lines_out).strip()
    debug = {
        "reconstruction_applied": True,
        "line_count": len(lines),
        "working_line_count": len(working_lines),
        "numeric_column_count": len(numeric_centers),
        "reconstructed_row_count": len(rows),
        "image_size": [image_width, image_height],
    }
    return markdown, debug


def reconstruct_financial_table_markdown(
    page_image: str | Path,
    *,
    lang: str = "vie+eng",
    psm: int = 6,
) -> Tuple[str, Dict[str, Any]]:
    image_path = Path(page_image)
    with Image.open(image_path) as img:
        image = ImageOps.exif_transpose(img).convert("L")
        image_size = image.size
    words = _run_tesseract_tsv(image_path, lang=lang, psm=psm)
    markdown, debug = reconstruct_table_from_words(words, image_size=image_size)
    debug["tsv_word_count"] = len(words)
    debug["page_image_path"] = str(image_path)
    debug["lang"] = lang
    debug["psm"] = int(psm)
    return markdown, debug
