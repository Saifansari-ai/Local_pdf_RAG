import PyPDF2
def extract_test(pdf_path,txt_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
        


pdf_path = ''
text_path = ''
extract_test(pdf_path,text_path)
print(f"Text extracted from {pdf_path} and saved to {text_path}")