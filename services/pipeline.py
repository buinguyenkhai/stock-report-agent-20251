"""
Financial Report Extraction Pipeline

Orchestrates parallel extraction and aggregated parsing.
"""

import asyncio
from typing import Dict, Any, Optional, Literal, List, Tuple, cast
from dataclasses import dataclass

from logger import get_logger
from services.extractors import (
    BaseExtractor,
    ExtractionResult,
    BalanceSheetExtractor,
    IncomeStatementExtractor,
    CashFlowExtractor,
    FinancialTablesExtractor,
    OtherTextExtractor,
    MetadataExtractor,
)
from services.parser import AggregatedParser, ExtractionBundle, ParsedReport
from services.utils import clean_markdown_tables
from services.llm_utils import extract_notes_tables_by_ref

logger = get_logger(__name__)



# Pipeline modes
PipelineMode = Literal["separate", "combined"]


@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline."""
    mode: PipelineMode = "separate"  # "separate" = 3 extractors, "combined" = 1 extractor
    extract_notes_text: bool = False
    extract_notes_tables: bool = False
    # If true, run a single notes tables extraction after parsing, guided by notes_ref.
    extract_notes_by_ref: bool = True
    extract_other_text: bool = False
    extract_metadata: bool = True
    extractor_model: Optional[str] = None
    parser_model: Optional[str] = None


class ExtractionPipeline:
    """
    Orchestrates the full extraction pipeline:
    1. Run extractors in parallel
    2. Aggregate results
    3. Parse with smart LLM
    4. Return structured output
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline with configuration.
        """
        self.config = config or PipelineConfig()
        self._extractors: Dict[str, BaseExtractor] = {}
        self._parser: Optional[AggregatedParser] = None
        self._last_bundle: Optional[ExtractionBundle] = None
        self._last_markdown: str = ""
        self._last_notes_cache: Optional[Tuple[Tuple[str, ...], Dict[str, Dict[str, str]], str]] = None
        
        self._init_extractors()
    
    def _init_extractors(self):
        """Initialize extractors based on config."""
        model = self.config.extractor_model
        
        if self.config.mode == "separate":
            # Use 3 separate financial table extractors
            self._extractors["balance_sheet"] = BalanceSheetExtractor(model)
            self._extractors["income_statement"] = IncomeStatementExtractor(model)
            self._extractors["cash_flow"] = CashFlowExtractor(model)
        else:
            # Use combined extractor
            self._extractors["financial_tables"] = FinancialTablesExtractor(model)
        
        if self.config.extract_other_text:
            self._extractors["other_text"] = OtherTextExtractor(model)
        
        if self.config.extract_metadata:
            self._extractors["metadata"] = MetadataExtractor(model)
    
    @property
    def parser(self) -> AggregatedParser:
        """Lazy-load parser."""
        if self._parser is None:
            self._parser = AggregatedParser(model=self.config.parser_model)
        return self._parser
    
    def process(self, markdown: str) -> ParsedReport:
        """
        Process markdown through the full pipeline synchronously.
        """
        return asyncio.run(self.process_async(markdown))
    
    async def process_async(self, markdown: str) -> ParsedReport:
        """
        Process markdown through the full pipeline asynchronously.
        """
        logger.info(f"Starting pipeline processing of {len(markdown):,} chars")
        
        # Step 0: Pre-clean markdown
        markdown = clean_markdown_tables(markdown)
        self._last_markdown = markdown
        
        # Step 1: Run all extractors in parallel
        extraction_results = await self._run_extractors(markdown)

        
        # Step 2: Build extraction bundle
        bundle = self._build_bundle(extraction_results)
        self._last_bundle = bundle  # Store for notes access

        # If nothing was extracted, surface extractor errors
        if not bundle.has_content():
            warnings = []
            for name, res in extraction_results.items():
                if not getattr(res, "success", False):
                    err = getattr(res, "error", None) or "unknown error"
                    warnings.append(f"Extractor {name} failed: {err}")
            if not warnings:
                warnings = ["No financial tables found in extraction"]
            return ParsedReport(warnings=warnings)
        
        # Step 3: Parse with smart LLM
        result = self.parser.parse(bundle)
        
        logger.info(
            f"Pipeline complete: BS={result.bs_found}, PL={result.pl_found}, CF={result.cf_found}"
        )
        
        return result
    
    async def _run_extractors(self, markdown: str) -> Dict[str, ExtractionResult]:
        """Run all extractors in parallel."""
        tasks = {}
        
        for name, extractor in self._extractors.items():
            tasks[name] = extractor.extract_async(markdown)
        
        # Run all tasks concurrently
        logger.info(f"Running {len(tasks)} extractors in parallel...")
        results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Map results back to names
        results = {}
        for name, result in zip(tasks.keys(), results_list):
            if isinstance(result, Exception):
                logger.error(f"Extractor {name} failed: {result}")
                results[name] = ExtractionResult(
                    extractor_name=name,
                    content="",
                    success=False,
                    error=str(result)
                )
            else:
                res = cast(ExtractionResult, result)
                results[name] = res
                if res.success and res.content:
                    logger.info(f"Extractor {name}: {len(res.content):,} chars")
                elif res.success and getattr(res, "metadata", {}).get("not_found"):
                    logger.info(f"Extractor {name}: not found")
                else:
                    err = getattr(res, "error", None) or "no content"
                    logger.warning(f"Extractor {name}: failed ({err})")
        
        return results
    
    def _build_bundle(self, results: Dict[str, ExtractionResult]) -> ExtractionBundle:
        """Build extraction bundle from results."""
        bundle = ExtractionBundle()
        
        if self.config.mode == "separate":
            # Get from separate extractors
            if "balance_sheet" in results and results["balance_sheet"].success and results["balance_sheet"].content:
                bundle.balance_sheet = results["balance_sheet"].content
            if "income_statement" in results and results["income_statement"].success and results["income_statement"].content:
                bundle.income_statement = results["income_statement"].content
            if "cash_flow" in results and results["cash_flow"].success and results["cash_flow"].content:
                bundle.cash_flow = results["cash_flow"].content
        else:
            # Get from combined extractor
            if "financial_tables" in results and results["financial_tables"].success:
                combined = results["financial_tables"]
                # Parse the combined result
                extractor = self._extractors.get("financial_tables")
                if isinstance(extractor, FinancialTablesExtractor):
                    # Re-extract with parsing
                    combined_result = extractor._extract_between_markers(
                        combined.content, 
                        extractor.BS_MARKER, 
                        extractor.BS_END
                    )
                    bundle.balance_sheet = combined_result
                    bundle.income_statement = extractor._extract_between_markers(
                        combined.content,
                        extractor.PL_MARKER,
                        extractor.PL_END
                    )
                    bundle.cash_flow = extractor._extract_between_markers(
                        combined.content,
                        extractor.CF_MARKER,
                        extractor.CF_END
                    )
        
        # Add other content
        if "other_text" in results and results["other_text"].success:
            bundle.other_text = results["other_text"].content
        
        # Add metadata
        if "metadata" in results and results["metadata"].success:
            bundle.metadata = results["metadata"].metadata
        
        return bundle
    
    def extract_only(self, markdown: str) -> Dict[str, ExtractionResult]:
        """
        Run only extraction without parsing.
        Useful for debugging extractors.
        """
        return asyncio.run(self._run_extractors(markdown))
    
    def to_dict(self, report: ParsedReport) -> Dict[str, Any]:
        """Convert ParsedReport to dictionary including notes."""
        base_dict = self.parser.to_dict(report)

        # Notes/TM (tables-only), extracted after parsing and guided by notes_ref.
        note_refs_needed: List[str] = []
        for statement in [report.balance_sheet, report.income_statement, report.cash_flow]:
            for item in statement.items:
                if item.notes_ref:
                    note_refs_needed.append(str(item.notes_ref))

        if not note_refs_needed or not self.config.extract_notes_by_ref:
            base_dict["notes_content"] = ""
            base_dict["notes_by_ref"] = {}
            return base_dict

        if not self._last_markdown:
            base_dict["notes_content"] = ""
            base_dict["notes_by_ref"] = {}
            warnings = base_dict.get("status", {}).get("warnings")
            if isinstance(warnings, list):
                warnings.append("Notes extraction skipped: missing OCR markdown in pipeline")
            return base_dict

        # Cache by requested refs to avoid repeating the big LLM call if to_dict() is invoked twice.
        refs_key = tuple(sorted(set(note_refs_needed)))
        if self._last_notes_cache and self._last_notes_cache[0] == refs_key:
            notes_by_ref = self._last_notes_cache[1]
            notes_content = self._last_notes_cache[2]
            base_dict["notes_by_ref"] = notes_by_ref
            base_dict["notes_content"] = notes_content
            return base_dict

        model = self.config.extractor_model or self.config.parser_model
        logger.info(f"Extracting TM tables by ref (refs needed: {list(refs_key)[:10]}...)")
        notes_by_ref = extract_notes_tables_by_ref(
            markdown=self._last_markdown,
            note_refs_needed=list(refs_key),
            model=model,
        )

        # Build a single markdown blob for the expander (tables-only).
        blocks: List[str] = []
        for key in sorted(notes_by_ref.keys()):
            note = notes_by_ref[key]
            title = (note.get("title") or "").strip()
            ref = (note.get("ref") or key).strip()
            content = (note.get("content") or "").strip()
            if not content:
                continue
            heading = f"## TM {ref}" + (f" - {title}" if title else "")
            blocks.append(f"{heading}\n\n{content}")
        notes_content = "\n\n".join(blocks)

        base_dict["notes_by_ref"] = notes_by_ref
        base_dict["notes_content"] = notes_content
        self._last_notes_cache = (refs_key, notes_by_ref, notes_content)
        
        return base_dict

def create_pipeline(
    mode: PipelineMode = "separate",
    extract_notes: bool = True,
    extract_metadata: bool = True,
    extractor_model: Optional[str] = None,
    parser_model: Optional[str] = None,
) -> ExtractionPipeline:
    """
    Create a configured extraction pipeline.
    
    Args:
        mode: "separate" for 3 extractors, "combined" for 1 extractor.
        extract_notes: Whether to extract notes content.
        extract_metadata: Whether to extract metadata.
        extractor_model: LLM model for extractors.
        parser_model: LLM model for parser.
        
    Returns:
        Configured ExtractionPipeline.
    """
    config = PipelineConfig(
        mode=mode,
        extract_notes_text=False,
        extract_notes_tables=False,
        extract_notes_by_ref=extract_notes,
        extract_other_text=False,
        extract_metadata=extract_metadata,
        extractor_model=extractor_model,
        parser_model=parser_model,
    )
    return ExtractionPipeline(config)


def process_markdown(
    markdown: str,
    mode: PipelineMode = "separate",
) -> ParsedReport:
    """
    Process markdown through the pipeline with default settings.
    """
    pipeline = create_pipeline(mode=mode)
    return pipeline.process(markdown)
