from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice
from .base import OCRStrategy

class DoclingOCRService(OCRStrategy):
    def __init__(self):
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True

        self.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, lang=['vie'])
        self.pipeline_options.ocr_options = self.ocr_options
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                )
            }
        )

    def process_pdf(self, pdf_url: str) -> str:
        input_path = pdf_url
        doc = self.converter.convert(input_path).document
        md = doc.export_to_markdown()
        return md
