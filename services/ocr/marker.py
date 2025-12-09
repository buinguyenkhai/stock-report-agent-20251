import os
import time
import requests
from .base import OCRStrategy
from dotenv import load_dotenv
load_dotenv()

class MarkerOCRService(OCRStrategy):
    def __init__(self):
        self.api_key = os.getenv("MARKER_API_KEY")
        self.url = "https://www.datalab.to/api/v1/marker"
        if not self.api_key:
            raise ValueError("MARKER_API_KEY is not set in environment variables.")

    def process_pdf(self, pdf_url: str) -> str:
        """
        Uploads PDF to Marker API and polls for result.
        """
        headers = {"X-Api-Key": self.api_key}
        form_data = {
            'file_url': (None, pdf_url),
            "force_ocr": (None, True),
            'output_format': (None, 'markdown'),
            "use_llm": (None, True),
            "disable_image_extraction": (None, True),
            "paginate": (None, True),
            "format_lines": (None, False),
            "additional_config": (None, "{\"drop_repeated_text\": true}")
        }

        print(f"MarkerOCR: Sending URL {pdf_url}")
        response = requests.post(self.url, files=form_data, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Marker API Error: {response.status_code} - {response.text}")

        data = response.json()
        request_check_url = data["request_check_url"]
        print(f"MarkerOCR: Job submitted. Check URL: {request_check_url}")

        # Poll for results
        max_polls = 175 # 175 * 2s = 350s timeout
        for i in range(max_polls):
            time.sleep(2)
            check_response = requests.get(request_check_url, headers=headers)
            if check_response.status_code != 200:
                print(f"MarkerOCR: Polling error {check_response.status_code}")
                continue
                
            check_result = check_response.json()
            status = check_result.get('status')
            
            if status == 'complete':
                print(f"MarkerOCR: Completed in {check_result.get('runtime')}s")
                return check_result['markdown']
            elif status == 'failed':
                raise Exception(f"MarkerOCR Failed: {check_result.get('error')}")
            else:
                # processing
                pass
        
        raise TimeoutError("MarkerOCR timed out waiting for completion.")