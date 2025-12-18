import json
import re
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings
from logger import get_logger

logger = get_logger(__name__)

LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_MIN_WAIT = 2
LLM_RETRY_MAX_WAIT = 30
DEFAULT_UTILS_MODEL = settings.llm_utils_model if hasattr(settings, 'llm_utils_model') else "gemini-3-flash-preview"

# LLM TABLE EXTRACTOR

class TableSection(BaseModel):
    """Identified section in a financial report."""
    section_type: str = Field(description="One of: BS (Balance Sheet), PL (Income Statement), CF (Cash Flow)")
    start_line: int = Field(description="1-based line number where section starts")
    end_line: int = Field(description="1-based line number where section ends")
    confidence: float = Field(description="Confidence score 0-1")


class TableExtractionResult(BaseModel):
    """Result of table extraction."""
    sections: List[TableSection] = Field(description="List of identified sections")
    unit: Optional[str] = Field(default=None, description="Detected currency unit")

class LLMTableExtractor:
    """
    Uses LLM to identify financial statement sections in markdown.
    Replaces regex-based pattern matching with semantic understanding.
    """
    
    def __init__(self, model: str = None):
        """Initialize with a fast, cheap model for extraction."""
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        
        model = model or DEFAULT_UTILS_MODEL
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
            google_api_key=settings.google_api_key
        )
        self.structured_llm = self.llm.with_structured_output(TableExtractionResult)
    
    def extract_sections(self, markdown_text: str) -> Dict[str, str]:
        """
        Extract BS, PL, CF sections from markdown using LLM.
        """
        # Split into lines for line-number reference
        lines = markdown_text.split('\n')
        total_lines = len(lines)

        if total_lines > 2000:
            # Create a condensed view with line numbers
            sample = self._create_condensed_view(lines)
            content_for_llm = sample
        else:
            # Add line numbers to help LLM reference
            content_for_llm = self._add_line_numbers(lines[:1500])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial document analyzer. Your task is to identify the line numbers where each financial statement section begins and ends.

Financial reports typically contain 3 main sections:
1. **BS** (Balance Sheet / Bảng cân đối kế toán) - Shows assets, liabilities, equity
2. **PL** (Income Statement / Báo cáo kết quả hoạt động kinh doanh) - Shows revenue, expenses, profit
3. **CF** (Cash Flow / Báo cáo lưu chuyển tiền tệ) - Shows cash movements

Look for:
- Section headers in Vietnamese or English
- Tables with financial line items (asset codes like 100, 110, 200, etc.)
- Section transitions (one statement ending before another begins)

Also identify the currency unit if visible (VND, triệu VND, tỷ VND, etc.)

Return the line numbers (1-based) for each section you find."""),
            ("user", """Analyze this financial report and identify the sections:

{content}

Total lines in document: {total_lines}

Find the start and end line numbers for BS, PL, and CF sections.""")
        ])
        
        try:
            chain = prompt | self.structured_llm
            result: TableExtractionResult = chain.invoke({
                "content": content_for_llm,
                "total_lines": total_lines
            })
            
            # Extract sections based on identified line numbers
            extracted = {"header": ""}
            
            for section in result.sections:
                section_type = section.section_type.upper()
                if section_type in ['BS', 'PL', 'CF']:
                    start = max(0, section.start_line - 1)  # Convert to 0-based
                    end = min(total_lines, section.end_line)
                    extracted[section_type] = '\n'.join(lines[start:end])
                    logger.debug(f"Extracted {section_type}: lines {section.start_line}-{section.end_line} ({end-start} lines)")
            
            # Add unit if detected
            if result.unit:
                extracted["unit"] = result.unit
            
            # Extract header (first 50 lines or until first section)
            first_section_line = min(
                (s.start_line for s in result.sections),
                default=50
            )
            extracted["header"] = '\n'.join(lines[:min(50, first_section_line)])
            
            logger.info(f"LLM extracted sections: {list(extracted.keys())}")
            return extracted
            
        except Exception as e:
            logger.error(f"LLM table extraction failed: {e}")
            return {}
    
    def _add_line_numbers(self, lines: List[str]) -> str:
        """Add line numbers to each line for LLM reference."""
        numbered = []
        for i, line in enumerate(lines, 1):
            if line.strip():
                numbered.append(f"L{i}: {line}")
        return '\n'.join(numbered)
    
    def _create_condensed_view(self, lines: List[str]) -> str:
        """Create a condensed view of large document for LLM."""
        total = len(lines)
        
        samples = []
        
        # First 300 lines
        samples.append("=== FIRST 300 LINES ===")
        samples.extend([f"L{i}: {line}" for i, line in enumerate(lines[:300], 1) if line.strip()])
        
        # Sample from middle
        mid_start = total // 3
        samples.append(f"\n=== LINES {mid_start}-{mid_start+300} (MIDDLE) ===")
        samples.extend([f"L{i}: {line}" for i, line in enumerate(lines[mid_start:mid_start+300], mid_start+1) if line.strip()])
        
        # Sample from 2/3
        late_start = 2 * total // 3
        samples.append(f"\n=== LINES {late_start}-{late_start+300} (LATE) ===")
        samples.extend([f"L{i}: {line}" for i, line in enumerate(lines[late_start:late_start+300], late_start+1) if line.strip()])
        
        # Last 100 lines
        samples.append("\n=== LAST 100 LINES ===")
        samples.extend([f"L{i}: {line}" for i, line in enumerate(lines[-100:], total-99) if line.strip()])
        
        return '\n'.join(samples)

# LLM ITEM MATCHER

class MatchResult(BaseModel):
    """Result of matching two financial items."""
    is_match: bool = Field(description="Whether the items refer to the same financial concept")
    confidence: float = Field(description="Confidence score 0-1")
    reason: str = Field(description="Brief explanation of why they match or don't match")

class LLMItemMatcher:
    """
    Uses LLM for semantic matching of financial item names.
    
    Replaces fuzzy string matching with semantic understanding.
    Handles:
    - Different naming conventions (banks vs corporates)
    - OCR errors and typos
    - Synonyms and abbreviations
    - Semantic opposites (thu vs trả)
    """
    
    def __init__(self, model: str = None):
        model = model or DEFAULT_UTILS_MODEL
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
            google_api_key=settings.google_api_key
        )
        self.structured_llm = self.llm.with_structured_output(MatchResult)
        self._match_cache: Dict[Tuple[str, str], MatchResult] = {}
    
    def match_items(
        self, 
        ocr_items: List[Dict], 
        ground_truth_items: List[Dict],
        section: str = "BS"
    ) -> List[Dict]:
        """
        Match OCR items to ground truth items using LLM.
        """
        matches = []
        used_gt_indices = set()
        
        for ocr_item in ocr_items:
            ocr_name = ocr_item.get("item_name", "")
            if not ocr_name:
                continue
            
            best_match = None
            best_confidence = 0.0
            best_gt_idx = -1
            
            # Find best match among ground truth
            for gt_idx, gt_item in enumerate(ground_truth_items):
                if gt_idx in used_gt_indices:
                    continue
                    
                gt_name = gt_item.get("item_name", "")
                if not gt_name:
                    continue
                
                # Check cache first
                cache_key = (ocr_name.lower(), gt_name.lower())
                if cache_key in self._match_cache:
                    result = self._match_cache[cache_key]
                else:
                    result = self._compare_items(ocr_name, gt_name, section)
                    self._match_cache[cache_key] = result
                
                if result.is_match and result.confidence > best_confidence:
                    best_match = gt_item
                    best_confidence = result.confidence
                    best_gt_idx = gt_idx
            
            if best_match and best_confidence >= 0.7:
                matches.append({
                    "ocr": ocr_item,
                    "gt": best_match,
                    "confidence": best_confidence
                })
                used_gt_indices.add(best_gt_idx)
        
        logger.info(f"LLM matched {len(matches)}/{len(ocr_items)} items in {section}")
        return matches
    
    def _compare_items(self, name1: str, name2: str, section: str) -> MatchResult:
        """Compare two item names using LLM."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Vietnamese financial accounting expert. Compare two financial line item names and determine if they refer to the SAME accounting concept.

Context: These items are from the {section} section of a Vietnamese financial report.

Consider:
1. **Synonyms**: "Tiền và tương đương tiền" = "Tiền mặt và các khoản tương đương tiền" = "Cash and cash equivalents"
2. **Abbreviations**: "TSCĐ" = "Tài sản cố định", "VCSH" = "Vốn chủ sở hữu"
3. **OCR errors**: "Phái thu" might be OCR error for "Phải thu"
4. **Formatting differences**: "I. Tài sản ngắn hạn" = "TÀI SẢN NGẮN HẠN" = "Tài sản ngắn hạn"

IMPORTANT - Do NOT match:
- Semantic opposites: "Phải thu" (receivables) vs "Phải trả" (payables)
- Parent vs child items: "Tài sản ngắn hạn" vs "Tiền mặt" (cash is subset of current assets)
- Different time periods: "Đầu năm" vs "Cuối kỳ"

Section meanings:
- BS (Balance Sheet): Assets, Liabilities, Equity items
- PL (Income Statement): Revenue, Expenses, Profit items  
- CF (Cash Flow): Cash inflows/outflows from operations, investing, financing"""),
            ("user", """Compare these two {section} items:

Item 1: "{name1}"
Item 2: "{name2}"

Do they refer to the same accounting concept?""")
        ])
        
        try:
            chain = prompt | self.structured_llm
            result = chain.invoke({
                "section": section,
                "name1": name1,
                "name2": name2
            })
            return result
        except Exception as e:
            logger.warning(f"LLM comparison failed for '{name1}' vs '{name2}': {e}")
            # Return no match on error
            return MatchResult(is_match=False, confidence=0.0, reason=str(e))
    
    def batch_match(
        self,
        ocr_items: List[str],
        gt_items: List[str],
        section: str = "BS"
    ) -> Dict[str, Optional[str]]:
        """
        Batch match OCR items to ground truth using a single LLM call.
        More efficient for large lists.
        """
        if not ocr_items or not gt_items:
            return {}
        
        # Create batch prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Vietnamese financial accounting expert. Match items from an OCR-extracted report to ground truth items.

Each OCR item should match AT MOST one ground truth item.
Only match if they refer to the SAME accounting concept.
Do NOT match semantic opposites (thu vs trả, tài sản vs nợ).

Return a JSON object mapping each OCR item index to its matching ground truth index, or null if no match.
Example: {{"0": 2, "1": null, "2": 0}} means OCR[0] matches GT[2], OCR[1] has no match, OCR[2] matches GT[0]."""),
            ("user", """Section: {section}

OCR Items:
{ocr_list}

Ground Truth Items:
{gt_list}

Return JSON mapping OCR indices to GT indices (or null for no match).""")
        ])
        
        ocr_list = '\n'.join(f"{i}: {name}" for i, name in enumerate(ocr_items))
        gt_list = '\n'.join(f"{i}: {name}" for i, name in enumerate(gt_items))
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "section": section,
                "ocr_list": ocr_list,
                "gt_list": gt_list
            })
            
            # Parse JSON response
            content = response.content
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', content)
            if json_match:
                mapping = json.loads(json_match.group())
                
                # Convert to name mapping
                result = {}
                for ocr_idx, gt_idx in mapping.items():
                    ocr_name = ocr_items[int(ocr_idx)]
                    if gt_idx is not None:
                        result[ocr_name] = gt_items[int(gt_idx)]
                    else:
                        result[ocr_name] = None
                
                return result
            
        except Exception as e:
            logger.warning(f"Batch match failed: {e}")
        
        return {}

# LLM UNIT DETECTOR

class UnitDetectionResult(BaseModel):
    """Result of unit detection."""
    unit: str = Field(description="Detected unit: VND, nghìn VND, triệu VND, or tỷ VND")
    confidence: float = Field(description="Confidence score 0-1")
    source: str = Field(description="Where the unit was found in the document")


class LLMUnitDetector:
    """
    Uses LLM to detect currency unit from document header.
    Replaces regex-based pattern matching with semantic understanding.
    Handles OCR errors and various formatting styles.
    """
    
    def __init__(self, model: str = None):
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        
        model = model or DEFAULT_UTILS_MODEL
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
            google_api_key=settings.google_api_key
        )
        self.structured_llm = self.llm.with_structured_output(UnitDetectionResult)
    
    def detect_unit(self, markdown_text: str) -> str:
        """
        Detect currency unit from document.
        """
        header_text = markdown_text[:5000]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a Vietnamese financial report to find the currency unit.

Look for phrases like:
- "Đơn vị tính: triệu VND" -> "triệu VND"
- "Đơn vị: VND" -> "VND"  
- "ĐVT: Tỷ đồng" -> "tỷ VND"
- "(Triệu đồng)" in table headers -> "triệu VND"
- Column headers containing unit info

Standard units:
- VND or đồng -> "VND" (full Vietnamese Dong)
- nghìn VND or nghìn đồng -> "nghìn VND" (thousands)
- triệu VND or triệu đồng -> "triệu VND" (millions)
- tỷ VND or tỷ đồng -> "tỷ VND" (billions)

If you see numbers like "1,234,567,890" without decimals, it's likely VND.
If you see numbers like "1,234.56" with smaller values, it's likely triệu VND.

Return your best guess with confidence level."""),
            ("user", """Find the currency unit in this document header:

{header}

What is the currency unit used?""")
        ])
        
        try:
            chain = prompt | self.structured_llm
            result: UnitDetectionResult = chain.invoke({"header": header_text})
            
            logger.info(f"LLM detected unit: {result.unit} (confidence: {result.confidence:.0%}) from: {result.source}")
            return result.unit
            
        except Exception as e:
            logger.warning(f"LLM unit detection failed: {e}, defaulting to VND")
            return "VND"

@lru_cache(maxsize=1)
def get_table_extractor() -> LLMTableExtractor:
    """Get singleton table extractor instance."""
    return LLMTableExtractor()


@lru_cache(maxsize=1) 
def get_item_matcher() -> LLMItemMatcher:
    """Get singleton item matcher instance."""
    return LLMItemMatcher()


@lru_cache(maxsize=1)
def get_unit_detector() -> LLMUnitDetector:
    """Get singleton unit detector instance."""
    return LLMUnitDetector()


def extract_tables_llm(markdown_text: str) -> str:
    """
    Convenience function to extract financial tables using LLM.
    """
    extractor = get_table_extractor()
    sections = extractor.extract_sections(markdown_text)
    
    if not sections:
        logger.warning("LLM extraction returned nothing, using full document")
        return markdown_text
    
    # Combine sections
    parts = []
    
    if sections.get("header"):
        parts.append("# Document Header\n" + sections["header"])
    
    for section_name in ["BS", "PL", "CF"]:
        if sections.get(section_name):
            parts.append(f"\n## {section_name}\n\n{sections[section_name]}")
    
    combined = "\n\n---\n\n".join(parts)
    
    reduction = (1 - len(combined) / len(markdown_text)) * 100 if markdown_text else 0
    logger.info(f"LLM extraction: {len(markdown_text):,} → {len(combined):,} chars ({reduction:.1f}% reduction)")
    
    return combined


def detect_unit_llm(markdown_text: str) -> str:
    """
    Convenience function to detect unit using LLM.
    """
    detector = get_unit_detector()
    return detector.detect_unit(markdown_text)
