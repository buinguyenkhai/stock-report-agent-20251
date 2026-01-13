"""
Services Module

Contains OCR services, extractors, and LLM utilities.
"""

# LLM utilities (used by benchmark and extractors)
from .llm_utils import (
    LLMItemMatcher,
    LLMUnitDetector,
    MatchResult,
    UnitDetectionResult,
    detect_unit_llm,
    get_item_matcher,
    get_unit_detector,
)
from .llm_factory import (
    create_llm,
    create_structured_llm,
    create_llm_for_task,
    create_structured_llm_for_task,
    get_model_info,
    test_model_structured_output,
    LLMConfig,
)

# Note: Pipeline and parser are disabled until parser.py is restored or refactored
# from .pipeline import (
#     ExtractionPipeline,
#     PipelineConfig,
#     create_pipeline,
#     process_markdown,
# )

__all__ = [
    # LLM utilities
    "LLMItemMatcher",
    "LLMUnitDetector",
    "MatchResult",
    "UnitDetectionResult",
    "detect_unit_llm",
    "get_item_matcher",
    "get_unit_detector",
    "create_llm",
    "create_structured_llm",
    "create_llm_for_task",
    "create_structured_llm_for_task",
    "get_model_info",
    "test_model_structured_output",
    "LLMConfig",
]
