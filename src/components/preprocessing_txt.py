import re
import os
import sys
from src.logger import logging
from src.exception import MyException

class Preprocessing:
   def __init__(self,text_path,cleaned_text_path):

      self.text_path = text_path
      self.cleaned_text_path = cleaned_text_path
   def clean_and_preprocess_text(self):
      for file in os.listdir(self.text_path):

         try :
            path = os.path.join(self.text_path, file)

            logging.info(f"cleaning and preprocessing started for {path}")

            with open(path, 'r', encoding='utf-8') as f:
               text = f.read()

            logging.info(f"lowering the text started for {path}")

            text = text.lower()
            text = re.sub(r"[ \t]+", " ", text).strip()

            logging.info(f"lowering the text completed for {path}")
            logging.info(f"removing page numbers started for {path}")
      
            # Remove page numbers
            text = re.sub(r"\bpage\s*\d+\b", " ", text)  

            logging.info(f"removing page numbers completed for {path}")
            logging.info(f"removing chapter numbers started for {path}")

            # Remove chapter numbers
            text = re.sub(r"\bchapter\s*\d+\b", " ", text)

            logging.info(f"removing chapter numbers completed for {path}")
            logging.info(f"removing standalone digits started for {path}")

            # Remove standalone digits (page numbers, figure numbers)
            text = re.sub(r"\b\d+\b", " ", text)

            logging.info(f"removing standalone digits completed for {path}")
            logging.info(f"removing chapter x patterns started for {path}")

            # Remove "chapter x" patterns
            text = re.sub(r"(chapter\s+\w+)", " ", text)

            logging.info(f"removing chapter x patterns completed for {path}")
            logging.info(f"removing non-letters started for {path}")

            text = re.sub(r"[^a-z\s]", " ", text)   
            text = re.sub(r"[ \t]+", " ", text).strip()

            logging.info(f"removing non-letters completed for {path}")
            logging.info(f"saving the cleaned text started for {path}")

            
            save_path = os.path.join(self.cleaned_text_path, f"cleaned_{file}")
            
            with open(save_path, 'w', encoding='utf-8') as f:
               f.write(text)
            logging.info(f"cleaned text saved to {save_path} from {path}") 

         except Exception as e:
            raise MyException(e,sys) from e