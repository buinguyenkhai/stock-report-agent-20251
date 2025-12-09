from .base import OCRStrategy
from .marker import MarkerOCRService
from .docling import DoclingOCRService
from .vintern import VinternOCRService
from .paddle import PaddleOCRService

def get_ocr_service(service_type: str = "marker") -> OCRStrategy:
    if service_type == "marker":
        return MarkerOCRService()
    elif service_type == "docling":
        return DoclingOCRService()
    elif service_type == "vintern":
        return VinternOCRService()
    elif service_type == "paddle":
        return PaddleOCRService()
    else:
        raise ValueError(f"Unknown OCR service: {service_type}")
