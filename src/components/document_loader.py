import PyPDF2
import sys
import os
from src.constant import *
from src.logger import logging
from src.exception import MyException



for file in os.listdir(PDF_PATH):
    try:
        pdf_path = os.path.join(PDF_PATH, file)
        logging.info(f"Extracting text from {pdf_path}")
        text = ""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        
        logging.info(f"Saving text to {TEXT_PATH}")

        text_file = os.path.join(TEXT_PATH,f"{file}.txt")
        os.makedirs(TEXT_PATH,exist_ok=True)

        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        logging.info(f"✅ Text extracted and saved to {text_file}")
    except Exception as e:
        raise MyException(e, sys) from e