from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class OCRStrategy(ABC):
    @abstractmethod
    def process_pdf(self, pdf_url: str) -> str:
        """
        Process PDF and return Markdown content.
        
        Args:
            pdf_url: Public URL of the PDF.
            
        Returns:
            str: The extracted Markdown content.
        """
        pass

    def cleanup_after_page(self) -> None:
        """Release transient state between page-level OCR calls."""
        return None

    def get_debug_artifacts(self) -> Optional[Dict[str, Any]]:
        return None

    def get_reconstruction_artifacts(self) -> Optional[Dict[str, Any]]:
        return None
