from paddleocr import PaddleOCRVL
from .base import OCRStrategy

class PaddleOCRService(OCRStrategy):
    def __init__(self):
        # Initialize PaddleOCRVL
        self.pipeline = PaddleOCRVL(
            vl_rec_backend="vllm-server", 
            vl_rec_server_url="http://127.0.0.1:8118/v1", 
            use_doc_orientation_classify=True
        )

    def process_pdf(self, pdf_url: str) -> str:
        # PaddleOCRVL takes a file path
        input_file = pdf_url
        output = self.pipeline.predict(input=input_file)

        markdown_list = []
        markdown_images = []

        for res in output:
            md_info = res.markdown
            markdown_list.append(md_info)
            markdown_images.append(md_info.get("markdown_images", {}))

        markdown_texts = self.pipeline.concatenate_markdown_pages(markdown_list)

        return markdown_texts