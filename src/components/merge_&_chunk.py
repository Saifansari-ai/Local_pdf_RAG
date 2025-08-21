# Merging all the text into single text file
import os
import sys
from src.logger import logging
from src.exception import MyException
import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Path to cleaned text files
folder_path = "/home/saif/Desktop/pdf_rag/data/cleaned_txt_file"

# Output merged file
output_file = "/home/saif/Desktop/pdf_rag/data/merged_data/merged_text.txt"

try:
    logging.info("✅ Merging all text files into single file")

    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)

                logging.info(f"merging {file_path} into {output_file}")

                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read() + "\n\n")  

                logging.info(f"✅ Merged {file_path} into {output_file}")

    print(f"✅ Merged all text files into: {output_file}")

except Exception as e:
    raise MyException(e, sys) from e


# Chunking the merged text file 

try:
    logging.info("Starting the process of chunking text")

    # ---------- CONFIG ----------
    INPUT_FILE = "/home/saif/Desktop/pdf_rag/data/merged_data/merged_text.txt"     # Path to merged text file
    OUTPUT_FILE = "/home/saif/Desktop/pdf_rag/data/chunks/chunks.json"        # Where to save chunks
    CHUNK_SIZE = 1000                  # Characters per chunk
    CHUNK_OVERLAP = 200                # Overlap between chunks
    SOURCE_NAME = "books_merged"       # Metadata: source label
    # ----------------------------

    def load_text(file_path):
        """Load text from a file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def chunk_text(text, chunk_size=1000, chunk_overlap=200):
        """Split text into chunks using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        return splitter.split_text(text)

    def save_chunks_to_json(chunks, output_file, source_name="merged_text"):
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
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    def main():
        logging.info(f"✅ Loading text from {INPUT_FILE}")
        
        text = load_text(INPUT_FILE)
        logging.info(f"✅ Loaded text, length: {len(text)} characters")

        logging.info("starting the process of chunking text")
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        logging.info(f"✅ Created {len(chunks)} chunks")

        logging.info("saving chunks to json")
        save_chunks_to_json(chunks, OUTPUT_FILE, SOURCE_NAME)
        logging.info(f"✅ Saved chunks to {OUTPUT_FILE}")

    if __name__ == "__main__":
        main()

except Exception as e:
    raise MyException(e, sys) from e