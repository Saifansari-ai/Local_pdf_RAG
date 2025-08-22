import json
import sys
from src.logger import logging
from src.exception import MyException
from sentence_transformers import SentenceTransformer
import os

try:
    logging.info("✅ Generating embeddings locally")

    MODEL_PATH = "/home/saif/Desktop/pre_trained_llms/bge-base-en-v1.5"
    logging.info(f"Loading embedding model from {MODEL_PATH}")

    # Load model once
    model = SentenceTransformer(MODEL_PATH)
    logging.info("✅ Loaded embedding model locally")

    # Load chunks
    logging.info("Loading chunks")
    with open("/home/saif/Desktop/pdf_rag/data/chunks/chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    logging.info("✅ Loaded chunks")

    # Extract text
    logging.info("Extracting text from JSON")
    texts = [chunk["text"] for chunk in data]
    logging.info(f"✅ Extracted {len(texts)} text chunks from JSON")

    # Generate embeddings in batches
    logging.info("Generating embeddings locally in batches")
    embeddings_array = model.encode(
        texts,
        batch_size=32,          
        convert_to_numpy=True,
        show_progress_bar=True  
    )

    # Pair embeddings with texts
    embeddings = [
        {"text": text, "embedding": embedding.tolist()}
        for text, embedding in zip(texts, embeddings_array)
    ]

    logging.info("✅ Generated embeddings locally")

    # Save to JSON
    logging.info("Saving embeddings to JSON")
    os.makedirs("/home/saif/Desktop/pdf_rag/data/embeddings", exist_ok=True)
    with open("/home/saif/Desktop/pdf_rag/data/embeddings/embedding.json", "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)

    logging.info("✅ Embeddings saved to embedding.json")

except Exception as e:
    raise MyException(e, sys) from e
