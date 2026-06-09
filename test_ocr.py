import pytesseract
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
poppler_path = r'C:\Users\lenovo\OneDrive\Desktop\geo-scan\geo-scan-backend\bin\poppler\Library\bin'

pages = convert_from_path(
    r'data\uploads\doc-182bbf42-geokniga-geologiya-i-geohimiya-nefti-i-gaza.img (1).pdf',
    dpi=200, poppler_path=poppler_path, first_page=28, last_page=28
)

text = pytesseract.image_to_string(pages[0], lang='rus+eng')

with open('ocr_test_page28.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print(f'Done: {len(text)} chars saved to ocr_test_page28.txt')
