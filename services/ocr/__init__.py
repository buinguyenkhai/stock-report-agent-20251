"""
OCR Services Module

Provides OCR strategies for processing financial report PDFs.
- MarkerOCRService: Local Marker with OpenRouter LLM (local)  
- DoclingOCRService: Local Docling with Tesseract Vietnamese (local)
"""

from .base import OCRStrategy
from .marker import MarkerOCRService  # Local Marker with OpenRouter
from .docling import DoclingOCRService


def get_ocr_service(service_type: str = "docling") -> OCRStrategy:
    """
    Factory function to get OCR service.
    
    Args:
        service_type: One of "docling" or "marker"
        
    Returns:
        OCR service instance
    """
    if service_type == "docling":
        return DoclingOCRService()
    elif service_type == "marker":
        return MarkerOCRService(use_llm=False)  # No LLM by default for speed
    else:
        raise ValueError(f"Unknown OCR service: {service_type}. Available: docling, marker")


__all__ = [
    "OCRStrategy",
    "MarkerOCRService",
    "DoclingOCRService",
    "get_ocr_service",
]
