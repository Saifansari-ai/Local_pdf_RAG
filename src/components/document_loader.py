import PyPDF2
import sys
import os
from src.constant import *
from src.logger import logging
from src.exception import MyException


def extract_test(pdf_path,txt_path):

    try:
        logging.info(f"Extracting text from {pdf_path}")
        text = ""
        dir_path = os.path.dirname(pdf_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        
        logging.info(f"Saving text to {txt_path}")
        dir_path_1 = os.path.dirname(txt_path)
        os.makedirs(dir_path_1,exist_ok=True)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logging.info(f"✅ Text extracted and saved to {txt_path}")
    except Exception as e:
        raise MyException(e, sys) from e

        
extract_test(PDF_PATH,TEXT_PATH)