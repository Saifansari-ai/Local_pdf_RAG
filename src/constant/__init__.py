# for text extraction
PDF_PATH = "/home/saif/Desktop/pdf_rag/database/raw_pdf"
TEXT_PATH = "/home/saif/Desktop/pdf_rag/database/extracted_txt"

# For txt Preprocessing 
CLEANED_TXT = "/home/saif/Desktop/pdf_rag/database/cleaned_txt"

# For Chunking
CHUNK_FILE = "/home/saif/Desktop/pdf_rag/database/chunk"
CHUNK_SIZE = 1000                  
CHUNK_OVERLAP = 200 
SOURCE_NAME = "cleaned_text"  

# For embedding
EMBED_MODEL_PATH = "/home/saif/Desktop/pre_trained_llms/bge-base-en-v1.5"
BATCH_SIZE=32       
CONVERT_TO_NUMPY=True
SHOW_PROGRESS_BAR=True 
EMBEDDINGS_PATH = "/home/saif/Desktop/pdf_rag/database/embeddings"