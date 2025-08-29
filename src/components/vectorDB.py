import json
import chromadb
import sys
import os
from src.logger import logging
from src.exception import MyException
from src.constant import *

class VectorDB:
    def __init__(self,embedding_path:str,chroma_db_path:str):
        self.chroma_db_path = chroma_db_path
        self.embeddings_path = embedding_path

    def store(self):
        try:
            logging.info("Starting to insert embeddings into ChromaDB")

            logging.info("Creating ChromaDB client")
            client = chromadb.PersistentClient(path=self.chroma_db_path)

            logging.info("Creating ChromaDB collection")
            collection = client.get_or_create_collection(name="txt_chunks")

            for file in os.listdir(self.embeddings_path):
                if file.endswith(".json"):
                    path = os.path.join(self.embeddings_path, file)
            with open(path, "r", encoding="utf-8") as f:
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

