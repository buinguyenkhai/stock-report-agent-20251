"""
Marker OCR Service

Uses marker-pdf library with OpenRouter API for LLM-assisted OCR.
Marker provides high-quality OCR with table structure recognition.
"""

import os
from pathlib import Path
from typing import Optional
from .base import OCRStrategy
from dotenv import load_dotenv

load_dotenv()

class MarkerOCRService(OCRStrategy):
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
    
    def process_pdf(self, pdf_url: str) -> str:
        """
        Process a PDF file and return markdown text.
        
        Args:
            pdf_url: Path or URL to PDF file
            
        Returns:
            Extracted markdown text
        """
        from marker.output import text_from_rendered

        def _is_http_url(s: str) -> bool:
            return s.startswith("http://") or s.startswith("https://")

        tmp_path: Optional[str] = None
        try:
            # Marker expects a local file path. If we receive a URL (e.g. Vietstock),
            # download it to a temp file first.
            if _is_http_url(pdf_url):
                import tempfile
                import requests

                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(pdf_url, stream=True, headers=headers, timeout=60)
                resp.raise_for_status()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            tmp.write(chunk)
                    tmp_path = tmp.name
                try:
                    resp.close()
                except Exception:
                    pass
                p = Path(tmp_path)
            else:
                p = Path(pdf_url)

            if not p.exists():
                raise FileNotFoundError(f"PDF not found: {p}")

            rendered = self.converter(str(p))
            text, _, _images = text_from_rendered(rendered)
            return text
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    
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
