"""
OCR Services Module

Provides OCR strategies for processing financial report PDFs.
- MarkerOCRService: Local Marker with OpenRouter LLM (local)  
- DoclingOCRService: Local Docling with Tesseract Vietnamese (local)
- HybridOcrModel: Docling-integrated hybrid OCR model (drop-in replacement)
"""

from .base import OCRStrategy
from .marker import MarkerOCRService  # Local Marker with OpenRouter
from .docling import DoclingOCRService

# Docling-integrated hybrid OCR model (requires docling)
try:
    from .hybrid_ocr_model import HybridOcrModel, HybridOcrOptions
    from .hybrid_pdf_pipeline import HybridPdfPipeline
except ImportError:
    HybridOcrModel = None 
    HybridOcrOptions = None
    HybridPdfPipeline = None


def get_ocr_service(service_type: str = "hybrid") -> OCRStrategy:
    """
    Factory function to get OCR service.
    
    Args:
        service_type: One of "docling", "marker", or "hybrid"
        
    Returns:
        OCR service instance
    """
    if service_type == "docling":
        return DoclingOCRService()
    elif service_type == "marker":
        return MarkerOCRService()
    elif service_type == "hybrid":
        # Use the high-quality Docling-integrated Hybrid pipeline
        return DoclingOCRService(use_hybrid=True)
    else:
        raise ValueError(f"Unknown OCR service: {service_type}. Available: docling, marker, hybrid")



__all__ = [
    "OCRStrategy",
    "MarkerOCRService",
    "DoclingOCRService",
    "HybridOcrModel",
    "HybridOcrOptions",
    "HybridPdfPipeline",
    "get_ocr_service",
]

