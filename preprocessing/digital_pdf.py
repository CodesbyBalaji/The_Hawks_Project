import fitz
from dateutil.parser import parse

def extract_digital_pages(pdf_path):
    doc = fitz.open(pdf_path)
    return [{
        "text": page.get_text(),
        "metadata": {"page_num": page.number + 1}
    } for page in doc]