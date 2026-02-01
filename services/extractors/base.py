import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from logger import get_logger
from services.llm_factory import create_llm

logger = get_logger(__name__)

# Default model for extraction
DEFAULT_EXTRACTION_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"


@dataclass
class ExtractionResult:
    """Result from an extractor."""
    extractor_name: str
    content: str  # Raw markdown or text content
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        return self.success and bool(self.content.strip())


class BaseExtractor(ABC):
    """
    Abstract base class for all extractors.
    
    Each extractor is responsible for finding and extracting a specific
    component from the OCR markdown output.
    """
    
    # Override in subclasses
    EXTRACTOR_NAME: str = "base"
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize extractor with LLM model.
        """
        self.model = model or DEFAULT_EXTRACTION_MODEL
        self._llm = None
    
    @property
    def llm(self):
        """Lazy-load LLM instance."""
        if self._llm is None:
            self._llm = create_llm(
                model=self.model,
                temperature=0.0,
                max_tokens=64000,  # Extractors return large chunks
                timeout=120,
            )
        return self._llm
    
    @abstractmethod
    def get_prompt(self) -> str:
        """
        Get the extraction prompt.
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt.
        """
        pass
    
    def extract(self, markdown: str) -> ExtractionResult:
        """
        Extract content from markdown synchronously.
        """
        return asyncio.run(self.extract_async(markdown))
    
    async def extract_async(self, markdown: str) -> ExtractionResult:
        """
        Extract content from markdown asynchronously.
        """
        try:
            logger.debug(f"{self.EXTRACTOR_NAME}: Starting extraction from {len(markdown):,} chars")
            
            messages = [
                ("system", self.get_system_prompt()),
                ("human", self.get_prompt().format(markdown=markdown)),
            ]
            
            response = await self.llm.ainvoke(messages)
            content = response.content.strip()
            
            # Validate extraction result
            if not content:
                logger.warning(f"{self.EXTRACTOR_NAME}: Empty extraction result")
                return ExtractionResult(
                    extractor_name=self.EXTRACTOR_NAME,
                    content="",
                    success=False,
                    error="Empty extraction result"
                )
            
            # Check for "not found" responses
            not_found_indicators = [
                "không tìm thấy",
                "not found",
                "không có",
                "no table",
                "no content",
            ]
            content_lower = content.lower()
            if any(ind in content_lower for ind in not_found_indicators) and len(content) < 200:
                logger.info(f"{self.EXTRACTOR_NAME}: Content not found in document")
                return ExtractionResult(
                    extractor_name=self.EXTRACTOR_NAME,
                    content="",
                    success=True,  # Not an error, just not present
                    metadata={"not_found": True}
                )
            
            logger.info(f"{self.EXTRACTOR_NAME}: Extracted {len(content):,} chars")
            return ExtractionResult(
                extractor_name=self.EXTRACTOR_NAME,
                content=content,
                success=True
            )
            
        except Exception as e:
            logger.error(f"{self.EXTRACTOR_NAME}: Extraction failed - {e}")
            return ExtractionResult(
                extractor_name=self.EXTRACTOR_NAME,
                content="",
                success=False,
                error=str(e)
            )
    
    def _clean_markdown(self, markdown: str) -> str:
        """
        Clean markdown content for better extraction.
        """
        import re
        
        # Remove image markers
        markdown = re.sub(r'<!--\s*image\s*-->', '', markdown)
        
        # Remove excessive whitespace
        markdown = re.sub(r'\n{4,}', '\n\n\n', markdown)
        
        # Remove page break markers
        markdown = re.sub(r'-{20,}', '', markdown)
        
        return markdown.strip()
