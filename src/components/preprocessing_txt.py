import re
import os
import sys
from src.logger import logging
from src.exception import MyException

file_path = "/home/saif/Desktop/pdf_rag/data/text_files"

for file in os.listdir(file_path):

     try :
        file_path = f"/home/saif/Desktop/pdf_rag/data/text_files/{file}"
        logging.info(f"cleaning and preprocessing started for {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
         text = f.read()

        logging.info(f"lowering the text started for {file_path}")

        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()

        logging.info(f"lowering the text completed for {file_path}")
        logging.info(f"removing page numbers started for {file_path}")
 
        # Remove page numbers
        text = re.sub(r"\bpage\s*\d+\b", " ", text)  

        logging.info(f"removing page numbers completed for {file_path}")
        logging.info(f"removing chapter numbers started for {file_path}")

        # Remove chapter numbers
        text = re.sub(r"\bchapter\s*\d+\b", " ", text)

        logging.info(f"removing chapter numbers completed for {file_path}")
        logging.info(f"removing standalone digits started for {file_path}")

        # Remove standalone digits (page numbers, figure numbers)
        text = re.sub(r"\b\d+\b", " ", text)

        logging.info(f"removing standalone digits completed for {file_path}")
        logging.info(f"removing chapter x patterns started for {file_path}")

        # Remove "chapter x" patterns
        text = re.sub(r"(chapter\s+\w+)", " ", text)

        logging.info(f"removing chapter x patterns completed for {file_path}")
        logging.info(f"removing non-letters started for {file_path}")

        text = re.sub(r"[^a-z\s]", " ", text)   # keep only letters
        text = re.sub(r"\s+", " ", text).strip()

        logging.info(f"removing non-letters completed for {file_path}")
        logging.info(f"saving the cleaned text started for {file_path}")

        save_path = f"/home/saif/Desktop/pdf_rag/data/cleaned_txt_file/cleaned_{file}"
        with open(save_path, 'w', encoding='utf-8') as f:
           f.write(text)
        logging.info(f"cleaned text saved to {save_path} for {file_path}") 

     except Exception as e:
        raise MyException(e,sys) from e