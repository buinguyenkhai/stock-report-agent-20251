"""
Financial Report Extraction Pipeline

Orchestrates parallel extraction and aggregated parsing.
"""

import asyncio
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass

from logger import get_logger
from services.extractors import (
    BaseExtractor,
    ExtractionResult,
    BalanceSheetExtractor,
    IncomeStatementExtractor,
    CashFlowExtractor,
    FinancialTablesExtractor,
    NotesTextExtractor,
    NotesTablesExtractor,
    OtherTextExtractor,
    MetadataExtractor,
)
from services.parser import AggregatedParser, ExtractionBundle, ParsedReport
from services.utils import clean_markdown_tables

logger = get_logger(__name__)



# Pipeline modes
PipelineMode = Literal["separate", "combined"]


@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline."""
    mode: PipelineMode = "separate"  # "separate" = 3 extractors, "combined" = 1 extractor
    extract_notes_text: bool = True
    extract_notes_tables: bool = True
    extract_other_text: bool = True
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
        
        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        self._extractors: Dict[str, BaseExtractor] = {}
        self._parser: Optional[AggregatedParser] = None
        
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
        
        # Optional extractors
        if self.config.extract_notes_text:
            self._extractors["notes_text"] = NotesTextExtractor(model)
        
        if self.config.extract_notes_tables:
            self._extractors["notes_tables"] = NotesTablesExtractor(model)
        
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
        
        Args:
            markdown: OCR markdown content.
            
        Returns:
            ParsedReport with extracted and normalized data.
        """
        return asyncio.run(self.process_async(markdown))
    
    async def process_async(self, markdown: str) -> ParsedReport:
        """
        Process markdown through the full pipeline asynchronously.
        
        Args:
            markdown: OCR markdown content.
            
        Returns:
            ParsedReport with extracted and normalized data.
        """
        logger.info(f"Starting pipeline processing of {len(markdown):,} chars")
        
        # Step 0: Pre-clean markdown
        markdown = clean_markdown_tables(markdown)
        
        # Step 1: Run all extractors in parallel
        extraction_results = await self._run_extractors(markdown)

        
        # Step 2: Build extraction bundle
        bundle = self._build_bundle(extraction_results)
        
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
                results[name] = result
                if result.success and result.content:
                    logger.info(f"Extractor {name}: {len(result.content):,} chars")
                else:
                    logger.warning(f"Extractor {name}: no content or failed")
        
        return results
    
    def _build_bundle(self, results: Dict[str, ExtractionResult]) -> ExtractionBundle:
        """Build extraction bundle from results."""
        bundle = ExtractionBundle()
        
        if self.config.mode == "separate":
            # Get from separate extractors
            if "balance_sheet" in results and results["balance_sheet"].success:
                bundle.balance_sheet = results["balance_sheet"].content
            if "income_statement" in results and results["income_statement"].success:
                bundle.income_statement = results["income_statement"].content
            if "cash_flow" in results and results["cash_flow"].success:
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
        
        # Add notes and other content
        if "notes_text" in results and results["notes_text"].success:
            bundle.notes_text = results["notes_text"].content
        
        if "notes_tables" in results and results["notes_tables"].success:
            bundle.notes_tables = results["notes_tables"].content
        
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
        
        Args:
            markdown: OCR markdown content.
            
        Returns:
            Dictionary of extraction results.
        """
        return asyncio.run(self._run_extractors(markdown))
    
    def to_dict(self, report: ParsedReport) -> Dict[str, Any]:
        """Convert ParsedReport to dictionary."""
        return self.parser.to_dict(report)


# ============ Convenience Functions ============

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
        extract_notes_text=extract_notes,
        extract_notes_tables=extract_notes,
        extract_other_text=extract_notes,
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
    
    Args:
        markdown: OCR markdown content.
        mode: "separate" or "combined" extraction mode.
        
    Returns:
        ParsedReport with extracted data.
    """
    pipeline = create_pipeline(mode=mode)
    return pipeline.process(markdown)
