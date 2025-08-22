import json
import chromadb
import sys
from src.logger import logging
from src.exception import MyException
from chromadb.utils import embedding_functions


try:
    logging.info("Starting to insert embeddings into ChromaDB")

    EMBEDDINGS_PATH = "/home/saif/Desktop/pdf_rag/data/embeddings/embedding.json"

    logging.info("Creating ChromaDB client")
    client = chromadb.PersistentClient(path="/home/saif/Desktop/pdf_rag/chroma_db")

    logging.info("Creating ChromaDB collection")
    collection = client.get_or_create_collection(name="pdf_chunks")

    logging.info("opening embedding ")
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    logging.info("Inserting embeddings into ChromaDB")
    ids = [str(i) for i in range(len(data))]
    documents = [item["text"] for item in data]
    embeddings = [item["embedding"] for item in data]

    
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    print(f"✅ Inserted {len(data)} embeddings into ChromaDB")

except Exception as e:
    raise MyException(e, sys) from e

