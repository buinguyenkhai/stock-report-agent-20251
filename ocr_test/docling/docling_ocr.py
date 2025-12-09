from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice
from time import time
import os

start = time()
input_doc_path = "https://static2.vietstock.vn/data/HNX/2025/BCTC/VN/QUY%202/SHS_Baocaotaichinh_Q2_2025.pdf"
doc_name = 'SHS_Q2_2025.pdf'

pipeline_options = PdfPipelineOptions()
pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True

ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, lang=['vie'])
#ocr_options = EasyOcrOptions(force_full_page_ocr=True, lang=['vi'])

pipeline_options.ocr_options = ocr_options

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

doc = converter.convert(input_doc_path).document
md = doc.export_to_markdown()

md_filename = doc_name.split('.')[0] + ".md"
output_filepath = os.path.join("ocr/docling/output", md_filename)

with open(output_filepath, "w", encoding="utf-8") as f:
    f.write(md)
    
end = time()
print(f"Total time: {end - start} seconds")