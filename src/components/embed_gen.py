import json
import sys
from src.logger import logging
from src.exception import MyException
from sentence_transformers import SentenceTransformer
import os
from src.constant import *

try:
    logging.info("✅ Generating embeddings locally")

    logging.info(f"Loading embedding model from {EMBED_MODEL_PATH}")

    # Load model once
    model = SentenceTransformer(EMBED_MODEL_PATH)
    logging.info("✅ Loaded embedding model locally")

    for file in os.listdir(CHUNK_FILE):
        
        chunk_file = os.path.join(CHUNK_FILE, file)
        logging.info(f"Loading chunks from {chunk_file}")
        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"✅ Loaded chunks from {chunk_file}")
        
        # Extract text
        logging.info("Extracting text from JSON")
        texts = [chunk["text"] for chunk in data]
        logging.info(f"✅ Extracted {len(texts)} text chunks from JSON")

        # Generate embeddings in batches
        logging.info("Generating embeddings locally in batches")
        embeddings_array = model.encode(
            texts,
            batch_size=BATCH_SIZE,          
            convert_to_numpy=CONVERT_TO_NUMPY,
            show_progress_bar=SHOW_PROGRESS_BAR  
        )

        # Pair embeddings with texts
        embeddings = [
            {"text": text, "embedding": embedding.tolist()}
            for text, embedding in zip(texts, embeddings_array)
        ]

        logging.info("✅ Generated embeddings locally")

        # Save to JSON
        logging.info("Saving embeddings to JSON")
        os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
        path = os.path.join(EMBEDDINGS_PATH, f"{file}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)

        logging.info("✅ Embeddings saved to embedding.json")

except Exception as e:
    raise MyException(e, sys) from e
