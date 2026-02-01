
import re
from logger import get_logger

logger = get_logger(__name__)

def clean_markdown_tables(text: str) -> str:
    """
    Apply safe heuristic fixes to OCR markdown tables.
    Focuses on fixing common structural issues without changing data.
    """
    if not text:
        return text

    # 1. Fix missing trailing pipes in rows that look like table rows
    # Pattern: Line starts with | and has at least one more | but doesn't end with |
    lines = text.split('\n')
    fixed_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|') and stripped.count('|') >= 1 and not stripped.endswith('|'):
            # Only append if it doesn't look like a header separator (e.g. |---| )
            if not re.match(r'^\|[\s\-\|]+$', stripped):
                line = line + '|'
        fixed_lines.append(line)
    
    text = '\n'.join(fixed_lines)

    # 2. Join numeric tokens split by newlines (common in narrow columns)
    # Pattern: digit + . or , + newline + digit
    def join_split_numbers(match):
        return match.group(1) + match.group(2)

    text = re.sub(r'(\d+[.,])\n(\d{3})(?=\s*\|)', r'\1\2', text)

    # 3. Normalize Vietnamese currency formatting (remove spaces in numbers)
    # Pattern: 1 . 234 . 567 -> 1.234.567
    text = re.sub(r'(\d)\s+([.,])\s+(\d)', r'\1\2\3', text)

    # 4. Fix parentheses spaces ( 1.234 ) -> (1.234)
    text = re.sub(r'\(\s+([\d.,]+)\s+\)', r'(\1)', text)
    
    return text

def normalize_financial_text(text: str) -> str:
    """
    Generic normalization for financial text.
    """
    # Replace various dash types with standard hyphen
    text = text.replace('–', '-').replace('—', '-')
    
    # Normalize whitespace but preserve newlines
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


def normalize_note_ref(ref: str) -> str:
    """
    Normalize a note reference for consistent lookup.
    
    Handles:
    - OCR errors: "S1" -> "5.1", "s43" -> "5.43"
    - Missing decimals: "53" -> "5.3", "510" -> "5.10" (common OCR error in BCTC)
    - Leading zeros: "5.01" -> "5.1"
    - Whitespace removal
    - Uppercase for roman numerals
    """
    if not ref:
        return ""
    
    ref = str(ref).strip()
    
    # Fix common OCR error: 'S' or 's' at start often means '5.'
    ref = re.sub(r'^[sS](\d)', r'5.\1', ref)
    
    # Fix missing decimal point in note references like "53" -> "5.3", "510" -> "5.10"
    # Only apply for patterns that look like note refs (5X, 5XX) not general numbers
    if re.match(r'^5\d{1,2}$', ref) and '.' not in ref:
        ref = '5.' + ref[1:]
    
    # Remove leading zeros after decimal: 5.01 -> 5.1
    ref = re.sub(r'\.0+(\d)', r'.\1', ref)
    
    # Remove internal whitespace
    ref = re.sub(r'\s+', '', ref)
    
    # Uppercase for consistency (V.01, v.01 -> V.01)
    ref = ref.upper()
    
    return ref
