from typing import List, Dict
from unidecode import unidecode
import re

from .canonical_format import FinancialItem
from services.llm_utils import LLMItemMatcher as _LLMItemMatcher
from logger import get_logger

logger = get_logger(__name__)


def normalize_name(name: str) -> str:
    """
    Normalize item name for comparison.
    Used for deduplication and basic text processing.
    """
    name = name.lower().strip()
    name = unidecode(name)
    name = re.sub(r'^[ivxlcdm]+\.\s*', '', name)  # Roman numeral prefixes
    name = re.sub(r'^[a-z0-9]+\.\s*', '', name)   # Letter/number prefixes
    name = re.sub(r'[^a-z\s]', '', name)          # Keep only letters and spaces
    return ' '.join(name.split())


class LLMBasedMatcher:
    """
    LLM-based semantic matcher for financial items.
    
    Uses LLM to understand semantic meaning and match items even when:
    - Names are spelled differently
    - Different naming conventions (banks vs corporates)
    - OCR errors and typos
    - Different languages (Vietnamese/English)
    """
    
    def __init__(self):
        """Initialize LLM matcher."""
        self._matcher = _LLMItemMatcher()
    
    def match_all(
        self,
        ocr_items: List[FinancialItem],
        vnstock_items: List[FinancialItem],
        section: str = "BS"
    ) -> Dict:
        """
        Match all OCR items against vnstock items using LLM.
        """
        # Use batch matching for efficiency
        ocr_names = [item.item_name for item in ocr_items]
        vn_names = [item.item_name for item in vnstock_items]
        
        # Get batch matches
        name_mapping = self._matcher.batch_match(ocr_names, vn_names, section)
        
        # Build result structure
        matched = []
        unmatched_ocr = []
        matched_vn_names = set()
        
        for ocr_item in ocr_items:
            matched_vn_name = name_mapping.get(ocr_item.item_name)
            
            if matched_vn_name:
                # Find the vnstock item
                for vn_item in vnstock_items:
                    if vn_item.item_name == matched_vn_name:
                        matched.append((ocr_item, vn_item, 0.9))  # LLM matches are high confidence
                        matched_vn_names.add(matched_vn_name)
                        break
            else:
                unmatched_ocr.append(ocr_item)
        
        # Find unmatched vnstock items
        unmatched_vnstock = [
            item for item in vnstock_items
            if item.item_name not in matched_vn_names
        ]
        
        logger.info(f"LLM matched {len(matched)}/{len(ocr_items)} OCR items to {len(vnstock_items)} vnstock items in {section}")
        
        return {
            "matched": matched,
            "unmatched_ocr": unmatched_ocr,
            "unmatched_vnstock": unmatched_vnstock
        }


def create_llm_matcher() -> LLMBasedMatcher:
    """Create an LLM-based matcher for semantic matching."""
    return LLMBasedMatcher()


def get_matcher() -> LLMBasedMatcher:
    """Get the LLM-based matcher."""
    return LLMBasedMatcher()
