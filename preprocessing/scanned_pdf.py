from pdf2image import convert_from_path
import pytesseract
from datetime import datetime

def extract_scanned_pages(pdf_path):
    images = convert_from_path(pdf_path)
    pages = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        pages.append({
            "text": text,
            "metadata": {"page_num": i + 1}
        })
    return pages