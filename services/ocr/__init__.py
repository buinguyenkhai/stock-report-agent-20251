from .base import OCRStrategy
from .marker import MarkerOCRService
from .docling import DoclingOCRService

def get_ocr_service(service_type: str = "marker") -> OCRStrategy:
    if service_type == "marker":
        return MarkerOCRService()
    elif service_type == "docling":
        return DoclingOCRService()
    else:
        raise ValueError(f"Unknown OCR service: {service_type}. Available: marker, docling")

