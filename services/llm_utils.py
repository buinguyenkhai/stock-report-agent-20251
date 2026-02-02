import re
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings
from logger import get_logger
from services.llm_factory import create_llm_for_task

logger = get_logger(__name__)


class NotesTablesByRefEntry(BaseModel):
    """Tables-only content for a single TM ref."""

    ref: str = Field(description="Requested note reference (e.g., '12', '5', '5.1', 'V.01')")
    title: str = Field(default="", description="Note section title (if available)")
    content: str = Field(default="", description="Markdown tables for this note (tables-only)")


class NotesTablesByRefResult(BaseModel):
    """Result of extracting TM tables grouped by reference."""

    notes: List[NotesTablesByRefEntry] = Field(default_factory=list)
    not_found_refs: List[str] = Field(default_factory=list)


_ROMAN_RE = re.compile(r"^[IVXLCDM]{1,8}$", re.IGNORECASE)


def _normalize_note_ref(ref: str) -> str:
    """Normalize note reference for consistent lookup."""

    if not ref:
        return ""

    ref = str(ref).strip().rstrip(".")

    # Fix common OCR error: 'S' or 's' at start often means '5.'
    ref = re.sub(r"^[sS](\d)", r"5.\1", ref)

    # Fix missing decimal: "53" -> "5.3", "510" -> "5.10"
    if re.match(r"^5\d{1,2}$", ref) and "." not in ref:
        ref = "5." + ref[1:]

    # Remove leading zeros: 5.01 -> 5.1
    ref = re.sub(r"\.0+(\d)", r".\1", ref)

    # Remove whitespace, uppercase
    ref = re.sub(r"\s+", "", ref).upper()
    return ref


def _is_parent_ref(norm_ref: str) -> bool:
    """True if this ref should include subnotes (e.g., '5' includes '5.1', 'V' includes 'V.01')."""

    if not norm_ref:
        return False
    if "." in norm_ref:
        return False
    return norm_ref.isdigit() or bool(_ROMAN_RE.match(norm_ref))


def _slice_notes_section(markdown: str) -> str:
    """Best-effort slice to notes section to reduce prompt size."""

    if not markdown:
        return ""

    patterns = [
        r"THUY\s*E\s*T\s*MINH\s*B\s*A\s*O\s*C\s*A\s*O\s*T\s*A\s*I\s*C\s*H\s*I\s*N\s*H",  # no accents
        r"THUY\s*E\s*T\s*MINH",
        r"THUYẾT\s*MINH\s*BÁO\s*CÁO\s*TÀI\s*CHÍNH",
        r"THUYẾT\s*MINH",
    ]

    for pat in patterns:
        m = re.search(pat, markdown, flags=re.IGNORECASE)
        if m:
            return markdown[m.start() :]

    return markdown


def _merge_children_into_parent(
    notes_by_ref: Dict[str, Dict[str, str]],
    requested_norm_refs: List[str],
) -> Dict[str, Dict[str, str]]:
    """Ensure parent refs (e.g. '5') include subnotes (e.g. '5.1', '5.2')."""

    requested_parents = [r for r in requested_norm_refs if _is_parent_ref(r)]
    if not requested_parents:
        return notes_by_ref

    for parent in requested_parents:
        children = sorted([k for k in notes_by_ref.keys() if k.startswith(parent + ".")])
        if not children:
            continue

        parts: List[str] = []
        for child in children:
            note = notes_by_ref.get(child) or {}
            child_ref = (note.get("ref") or child).strip() or child
            child_title = (note.get("title") or "").strip()
            child_content = (note.get("content") or "").strip()
            if not child_content:
                continue
            heading = f"### TM {child_ref}" + (f" - {child_title}" if child_title else "")
            parts.append(f"{heading}\n\n{child_content}")

        if not parts:
            continue

        combined_children = "\n\n".join(parts)
        existing = notes_by_ref.get(parent)
        if existing and (existing.get("content") or "").strip():
            notes_by_ref[parent]["content"] = (existing.get("content") or "").rstrip() + "\n\n" + combined_children
        else:
            notes_by_ref[parent] = {
                "ref": parent,
                "title": existing.get("title") if isinstance(existing, dict) else "",
                "content": combined_children,
            }

    return notes_by_ref


def extract_notes_tables_by_ref(
    markdown: str,
    note_refs_needed: List[str],
    model: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Single-call extraction of TM tables grouped by referenced notes_ref.
    """

    if not markdown or not note_refs_needed:
        return {}

    requested_norm = []
    for r in note_refs_needed:
        nr = _normalize_note_ref(r)
        if nr:
            requested_norm.append(nr)
    requested_norm = sorted(set(requested_norm))
    if not requested_norm:
        return {}

    model = model or settings.llm_model

    notes_section = _slice_notes_section(markdown)
    max_chars = 180_000
    if len(notes_section) > max_chars:
        notes_section = notes_section[:max_chars]

    llm = create_llm_for_task("notes_tables_by_ref", model=model)
    structured = llm.with_structured_output(NotesTablesByRefResult)

    refs_csv = ", ".join(requested_norm)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Vietnamese financial report expert.

Task: From the THUYET MINH (Notes) section of a Vietnamese BCTC, extract ONLY the markdown TABLES that belong to specific note references (TM refs).

IMPORTANT: The 'notes' field must be a JSON ARRAY of objects, NOT a dictionary.
Each object must have the structure: {{"ref": "...", "title": "...", "content": "..."}}

Rules:
- Output must be clean markdown tables only (pipes). Do NOT include narrative paragraphs.
- Group tables under the requested note ref.
- If a requested ref is a PARENT ref (e.g., '5' or 'V'), include ALL subnotes under it (e.g., 5.1, 5.2, ... or V.01, V.02, ...). Within that parent's content, add small markdown headings before each subnote's tables.
- Preserve table structure as-is (keep rows/columns).
- Do NOT return content for refs that were not requested.
- If you cannot find a requested ref, add it to not_found_refs.
""",
            ),
            (
                "user",
                """Requested TM refs (normalized): {refs_csv}

THUYET MINH content (OCR markdown):

{notes_markdown}
""",
            ),
        ]
    )

    chain = prompt | structured
    try:
        result: NotesTablesByRefResult = chain.invoke({"refs_csv": refs_csv, "notes_markdown": notes_section})
    except Exception as e:
        logger.error(f"TM tables by ref extraction failed: {e}")
        return {}

    notes_by_ref: Dict[str, Dict[str, str]] = {}
    for entry in result.notes:
        norm_key = _normalize_note_ref(entry.ref)
        if not norm_key:
            continue
        notes_by_ref[norm_key] = {
            "ref": entry.ref,
            "title": entry.title,
            "content": entry.content,
        }

    notes_by_ref = _merge_children_into_parent(notes_by_ref, requested_norm)
    return notes_by_ref
