import fitz  # PyMuPDF

def is_scanned(pdf_path):
    doc = fitz.open(pdf_path)
    for page in doc:
        if page.get_text().strip():  # Has selectable text
            return False
    return True