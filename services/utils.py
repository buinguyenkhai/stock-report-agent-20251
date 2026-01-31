
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
    # We only do this if it's within a pipe table row context
    def join_split_numbers(match):
        return match.group(1) + match.group(2)

    # Simplified approach: join any digit., + newline + digit
    # Risk: may join year and something else. Guard with table pipe check.
    # For now, let's use a very safe version: only if it's like "1." \n "234"
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
