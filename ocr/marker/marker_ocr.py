from dotenv import load_dotenv
import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from time import time
load_dotenv()

start = time()
# Configs
config = {
    "output_format": "markdown",
    "force_ocr": True,
    "use_llm": True,
    "gemini_api_key": os.getenv("GOOGLE_API_KEY"),
    "TORCH_DEVICE": "cuda",
    "disable_image_extraction": True,
    }
config_parser = ConfigParser(config)
# OCR Model (Surya)
model_artifacts = create_model_dict(device="cuda:0")

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service()
)

pdf_path = "ocr/DBC_Baocaotaichinh_Q3_2025_Hopnhat.pdf"
rendered = converter(pdf_path)

text, _, images = text_from_rendered(rendered)

pdf_filename = os.path.basename(pdf_path)
md_filename = os.path.splitext(pdf_filename)[0] + ".md"
output_filepath = os.path.join("ocr/marker/output", md_filename)

with open(output_filepath, "w", encoding="utf-8") as f:
    f.write(text)

end = time()
print(f"Elapsed time: {end - start} seconds")