import time
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .base import OCRStrategy
from config import settings
from logger import get_logger

logger = get_logger(__name__)

class MarkerOCRService(OCRStrategy):
    def __init__(self):
        self.api_key = settings.marker_api_key
        self.url = "https://www.datalab.to/api/v1/marker"
        if not self.api_key:
            raise ValueError("MARKER_API_KEY is not set in environment variables.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=settings.retry_min_wait, max=settings.retry_max_wait),
        retry=retry_if_exception_type((requests.RequestException, TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying Marker API call (attempt {retry_state.attempt_number})..."
        )
    )
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

        logger.info(f"MarkerOCR: Sending URL {pdf_url}")
        response = requests.post(self.url, files=form_data, headers=headers, timeout=60)

        if response.status_code != 200:
            logger.error(f"Marker API Error: {response.status_code} - {response.text}")
            raise Exception(f"Marker API Error: {response.status_code} - {response.text}")

        data = response.json()
        request_check_url = data["request_check_url"]
        logger.info(f"MarkerOCR: Job submitted. Check URL: {request_check_url}")

        # Poll for results
        max_polls = settings.ocr_max_polls
        poll_interval = settings.ocr_poll_interval
        
        for i in range(max_polls):
            time.sleep(poll_interval)
            check_response = requests.get(request_check_url, headers=headers, timeout=30)
            if check_response.status_code != 200:
                logger.warning(f"MarkerOCR: Polling error {check_response.status_code}")
                continue
                
            check_result = check_response.json()
            status = check_result.get('status')
            
            if status == 'complete':
                logger.info(f"MarkerOCR: Completed in {check_result.get('runtime')}s")
                return check_result['markdown']
            elif status == 'failed':
                error_msg = f"MarkerOCR Failed: {check_result.get('error')}"
                logger.error(error_msg)
                raise Exception(error_msg)
            else:
                # processing - log every 10 polls
                if i % 10 == 0:
                    logger.debug(f"MarkerOCR: Still processing... (poll {i+1}/{max_polls})")
        
        raise TimeoutError("MarkerOCR timed out waiting for completion.")