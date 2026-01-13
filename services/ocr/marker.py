"""
Marker OCR Service

Uses marker-pdf library with OpenRouter API for LLM-assisted OCR.
Marker provides high-quality OCR with table structure recognition.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class MarkerOCRService:
    """
    OCR service using marker-pdf with OpenRouter API for LLM post-processing.
    
    Marker uses:
    - Surya OCR models for text detection
    - LLM (via OpenRouter) for table correction and formatting
    
    Usage:
        service = MarkerOCRService()
        result = service.process_pdf("path/to/file.pdf")
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        llm_model: str = "mistralai/mistral-small-3.1-24b-instruct",  # OpenRouter vision model
        force_ocr: bool = True,
        extract_images: bool = False,
        device: str = "cuda",
    ):
        """
        Initialize Marker OCR service.
        
        Args:
            use_llm: Whether to use LLM for post-processing (table fixing)
            llm_model: OpenRouter model to use (e.g., "google/gemini-2.0-flash-001")
            force_ocr: Force OCR even for text PDFs
            extract_images: Whether to extract images from PDF
            device: Device for ML models ("cuda" or "cpu")
        """
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.force_ocr = force_ocr
        self.extract_images = extract_images
        self.device = device
        
        # OpenRouter API config
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        
        # Lazy-loaded converter
        self._converter = None
        self._model_artifacts = None
    
    def _get_config(self) -> dict:
        """Build Marker configuration dictionary."""
        config = {
            "output_format": "markdown",
            "force_ocr": self.force_ocr,
            "disable_image_extraction": not self.extract_images,
            "TORCH_DEVICE": self.device,
        }
        
        if self.use_llm:
            config["use_llm"] = True
            config["llm_service"] = "marker.services.openai.OpenAIService"
            config["openai_api_key"] = self.openrouter_api_key
            config["openai_base_url"] = self.openrouter_base_url
            config["openai_model"] = self.llm_model
        
        return config
    
    @property
    def converter(self):
        """Lazy-load the PDF converter."""
        if self._converter is None:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.config.parser import ConfigParser
            
            config = self._get_config()
            config_parser = ConfigParser(config)
            
            self._model_artifacts = create_model_dict(device=f"{self.device}:0")
            
            self._converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=self._model_artifacts,
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
                llm_service=config_parser.get_llm_service() if self.use_llm else None,
            )
        
        return self._converter
    
    def process_pdf(self, pdf_path: str) -> str:
        """
        Process a PDF file and return markdown text.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted markdown text
        """
        from marker.output import text_from_rendered
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        rendered = self.converter(str(pdf_path))
        text, _, images = text_from_rendered(rendered)
        
        return text
    
    def process_image(self, image_path: str) -> str:
        """
        Process a single image and return markdown text.
        
        Note: Marker is optimized for PDFs. For single images,
        consider using Docling instead.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted markdown text
        """
        # For images, marker requires converting to single-page PDF first
        # This is less efficient than using the image directly
        raise NotImplementedError(
            "Marker is optimized for PDFs. For single images, use DoclingOCRService."
        )


# Convenience function for quick usage
def marker_ocr(pdf_path: str, use_llm: bool = True) -> str:
    """
    Quick function to OCR a PDF with Marker.
    
    Args:
        pdf_path: Path to PDF file
        use_llm: Whether to use LLM for post-processing
        
    Returns:
        Extracted markdown text
    """
    service = MarkerOCRService(use_llm=use_llm)
    return service.process_pdf(pdf_path)


if __name__ == "__main__":
    import sys
    from time import time
    
    if len(sys.argv) < 2:
        print("Usage: python marker_ocr.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    print(f"Processing: {pdf_path}")
    
    start = time()
    service = MarkerOCRService(use_llm=True)
    text = service.process_pdf(pdf_path)
    elapsed = time() - start
    
    print(f"Extracted {len(text)} characters in {elapsed:.1f}s")
    print("\n--- First 1000 chars ---")
    print(text[:1000])
