import requests
import os
import time
from dotenv import load_dotenv
load_dotenv()
# 35s - 43 trang - 0.13 USD
url = "https://www.datalab.to/api/v1/marker"
#pdf_path = "ocr/FPT_Baocaotaichinh_Q3_2025_Congtyme.pdf"
pdf_url = "https://static2.vietstock.vn/vietstock/2025/10/10/20251010___ijc___bctc_hop_nhat_quy_3_nam_2025.pdf"
pdf_name = 'IJC_Q3_2025.pdf'
form_data = {
    #'file': (pdf_path, open(pdf_path, 'rb'), 'application/pdf'),
    'file_url': (None, pdf_url),
    "force_ocr": (None, True),
    'output_format': (None, 'markdown'),
    "use_llm": (None, True),
    "disable_image_extraction": (None, True),
    "paginate": (None, True),
    "format_lines": (None, False),
    "additional_config": (None, "{\"drop_repeated_text\": true}")
}

headers = {"X-Api-Key": os.getenv("MARKER_API_KEY")}
response = requests.post(url, files=form_data, headers=headers)
data = response.json()

max_polls = 350
check_url = data["request_check_url"]
for i in range(max_polls):
    response = requests.get(check_url, headers=headers)
    check_result = response.json()
    if check_result['status'] == 'complete':
        converted_document = check_result['markdown']
        md_filename = pdf_name.split('.')[0] + ".md"
        output_filepath = os.path.join("ocr/marker/output", md_filename)
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(converted_document)

        print(f"Elapsed time: {check_result['runtime']} seconds")
        break

    elif check_result["status"] == "failed":
        print("Failed to convert, uh oh...")
        break
    else:
        print("Waiting 2 more seconds to re-check conversion status")
        time.sleep(2)
