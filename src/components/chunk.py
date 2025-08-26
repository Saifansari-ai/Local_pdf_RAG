import os
import sys
from src.logger import logging
from src.exception import MyException
import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.constant import *

# Chunking the text file 

try:
    logging.info("Starting the process of chunking text")
    def load_text(file_path):
        """Load text from a file."""
        for file in os.listdir(file_path):
            if file.endswith(".txt"):
                file = os.path.join(file_path, file)
        with open(file, "r", encoding="utf-8") as f:
            return f.read()

    def chunk_text(text, chunk_size=1000, chunk_overlap=200):
        """Split text into chunks using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        return splitter.split_text(text)

    def save_chunks_to_json(chunks, chunk_file, source_name="merged_text"):
        """Save chunks into a JSON file with metadata."""
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                "chunk_id": i,
                "text": chunk,
                "metadata": {
                    "source": source_name
                }
            })
        os.makedirs(chunk_file, exist_ok=True)
        path = os.path.join(chunk_file, f"{source_name}.json")
        
        logging.info(f"Saving chunks to {path}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        logging.info(f"✅ Saved chunks to {path}")

    def main():
        logging.info(f"✅ Loading text from {CLEANED_TXT}")
        
        text = load_text(CLEANED_TXT)
        logging.info(f"✅ Loaded text, length: {len(text)} characters")

        logging.info("starting the process of chunking text")
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        logging.info(f"✅ Created {len(chunks)} chunks")

        save_chunks_to_json(chunks, CHUNK_FILE, SOURCE_NAME)
        

    if __name__ == "__main__":
        main()

except Exception as e:
    raise MyException(e, sys) from e