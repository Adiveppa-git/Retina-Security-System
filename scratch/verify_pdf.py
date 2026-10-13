import os
from reportlab.pdfgen import canvas
import pypdf

pdf_path = r"c:\Users\renki\Downloads\iris-security12\IRIS_SECURITY_SYSTEM_COMPLETE_TECHNICAL_REPORT.pdf"

if os.path.exists(pdf_path):
    size_bytes = os.path.getsize(pdf_path)
    reader = pypdf.PdfReader(pdf_path)
    num_pages = len(reader.pages)
    print(f"PDF Size: {size_bytes} bytes ({size_bytes / 1024:.2f} KB)")
    print(f"Total Page Count: {num_pages}")
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        print(f"Page {i+1} character count: {len(text)}")
else:
    print("PDF File NOT found!")
