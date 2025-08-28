import PyPDF2
import sys
import os
from src.logger import logging
from src.exception import MyException

class PDFExtractor:
    def __init__(self, pdf_path: str, text_path: str):
        """
        Initialize with the input pdf directory and output text directory
        """
        self.pdf_path = pdf_path
        self.text_path = text_path

    def extract_text_from_pdf(self):
        """
        Extract text from all PDFs in the directory and save as .txt files
        """
        for file in os.listdir(self.pdf_path):
            try:
                path = os.path.join(self.pdf_path, file)
                logging.info(f"Extracting text from {path}")
                text = ""
                with open(path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() or ""
                logging.info(f"✅ Extracted text from {path}")
                
                logging.info(f"Saving text to {self.text_path}")
                path = os.path.join(self.text_path, f"{file}.txt")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(text)
                logging.info(f"✅ Saved text to {path}")
            except Exception as e:
                raise MyException(e, sys) from e