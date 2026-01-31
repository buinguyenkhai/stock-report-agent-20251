from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice
from .base import OCRStrategy
from PIL import Image
import tempfile
import os

try:
    from .hybrid_ocr_model import HybridOcrOptions
    from .hybrid_pdf_pipeline import HybridPdfPipeline
    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False

class DoclingOCRService(OCRStrategy):
    def __init__(self, use_hybrid: bool = False):
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        # self.pipeline_options.table_structure_options.do_cell_matching = True  # Deprecated or unknown in this version

        if use_hybrid and HAS_HYBRID:
            from .hybrid_ocr_model import HybridOcrOptions
            from .hybrid_pdf_pipeline import HybridPdfPipeline
            self.ocr_options = HybridOcrOptions(
                lang=['vie'],
                force_full_page_ocr=True,
                confidence_threshold=0.7,
                number_confidence_threshold=0.85
            )
            pipeline_cls = HybridPdfPipeline
        else:
            self.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, lang=['vie'])
            pipeline_cls = None

        self.pipeline_options.ocr_options = self.ocr_options
        
        # PDF converter
        pdf_format_option = PdfFormatOption(
            pipeline_options=self.pipeline_options,
        )
        if pipeline_cls:
            pdf_format_option.pipeline_cls = pipeline_cls

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pdf_format_option
            }
        )
        
        # Image converter (lazy-loaded)
        self._image_converter = None


    @property
    def image_converter(self):
        """Lazy-load image converter with same pipeline options."""
        if self._image_converter is None:
            self._image_converter = DocumentConverter(
                allowed_formats=[InputFormat.IMAGE],
                format_options={
                    InputFormat.IMAGE: ImageFormatOption(
                        pipeline_options=self.pipeline_options,
                    )
                }
            )
        return self._image_converter

    def process_pdf(self, pdf_url: str) -> str:
        """
        Process PDF and return Markdown content.
        """
        input_path = pdf_url
        doc = self.converter.convert(input_path).document

        md = doc.export_to_markdown()
        return md
    
    def process_image(self, image: Image.Image) -> str:
        """
        Process a single image and return markdown text.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted markdown text
        """
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, format="PNG")
            tmp_path = tmp.name
        
        try:
            result = self.image_converter.convert(tmp_path)
            return result.document.export_to_markdown()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

