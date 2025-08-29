import json
import sys
from src.logger import logging
from src.exception import MyException
from sentence_transformers import SentenceTransformer
import os

class EmbeddingGenerator:
    def __init__(self, embed_model_path: str, chunk_file:str, embeddings_path:str):
        self.embed_model_path = embed_model_path
        self.chunk_file = chunk_file
        self.embeddings_path = embeddings_path

    def Genrate(self):

        try:
            logging.info("✅ Generating embeddings locally")

            logging.info(f"Loading embedding model from {self.embed_model_path}")

            # Load model once
            model = SentenceTransformer(self.embed_model_path)
            logging.info("✅ Loaded embedding model locally")

            for file in os.listdir(self.chunk_file):
                
                path = os.path.join(self.chunk_file, file)
                logging.info(f"Loading chunks from {self.chunk_file}")
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logging.info(f"✅ Loaded chunks from {self.chunk_file}")
                
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
                path = os.path.join(self.embeddings_path, f"{file}.json")

                with open(path, "w", encoding="utf-8") as f:
                    json.dump(embeddings, f, ensure_ascii=False, indent=2)

                logging.info("✅ Embeddings saved to embedding.json")

        except Exception as e:
            raise MyException(e, sys) from e
