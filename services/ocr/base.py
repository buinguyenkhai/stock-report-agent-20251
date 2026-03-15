from abc import ABC, abstractmethod

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
