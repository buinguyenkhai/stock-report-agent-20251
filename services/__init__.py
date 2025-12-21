"""
Services Module

Contains OCR services, extractors, parser, and pipeline.
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

# New pipeline
from .pipeline import (
    ExtractionPipeline,
    PipelineConfig,
    create_pipeline,
    process_markdown,
)
from .parser import (
    AggregatedParser,
    ParsedReport,
    ExtractionBundle,
)
from .vnstock_vocabulary import (
    BALANCE_SHEET_ITEMS,
    INCOME_STATEMENT_ITEMS,
    CASH_FLOW_ITEMS,
    get_vocabulary_prompt_section,
)

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
    # Pipeline
    "ExtractionPipeline",
    "PipelineConfig",
    "create_pipeline",
    "process_markdown",
    "AggregatedParser",
    "ParsedReport",
    "ExtractionBundle",
    "BALANCE_SHEET_ITEMS",
    "INCOME_STATEMENT_ITEMS",
    "CASH_FLOW_ITEMS",
    "get_vocabulary_prompt_section",
]
