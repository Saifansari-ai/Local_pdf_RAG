from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.logger import logging
from src.exception import MyException
import sys
import torch


class RAGQueryEngine:
    def __init__(self,
                 embedding_model_path: str,
                 chroma_db_path: str,
                 llm_model_path: str):
        try:
            logging.info("Initializing RAGQueryEngine")

            # ================= Embedding model ================= #
            logging.info(f"Loading embedding model from {embedding_model_path}")
            self.embedding_model = SentenceTransformer(embedding_model_path)

            # ================= ChromaDB Setup ================= #
            logging.info(f"Connecting to ChromaDB at {chroma_db_path}")
            client = chromadb.PersistentClient(path=chroma_db_path)
            self.collection = client.get_collection(name="txt_chunks")

            # ================= LLM Setup ================= #
            logging.info(f"Loading LLM model from {llm_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
            model = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            self.llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                device_map="auto",
                return_full_text=False
            )
            logging.info("Local LLM loaded successfully")

            # Store conversation history
            self.conversation_history = []

        except Exception as e:
            raise MyException(e, sys) from e

    def query(self, user_query: str, top_k: int = 10) -> str:
        """Run retrieval-augmented query against PDF embeddings + LLM"""
        try:
            if not user_query.strip():
                return "⚠️ Query cannot be empty!"

            logging.info("Encoding query")
            query_embedding = self.embedding_model.encode(user_query)
            logging.info("✅ Query encoded")

            logging.info("Retrieving context from ChromaDB")
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            retrieved_chunks = sum(results["documents"], [])
            context = "\n".join(retrieved_chunks)
            logging.info("✅ Context retrieved from ChromaDB")

            # ================= Build Instruction Prompt ================= #
            logging.info("Creating prompt for LLM")

            history_text = ""
            for turn in self.conversation_history[-5:]:  # keep last 5 turns
                history_text += f"User asked: {turn['user']}\nAssistant replied: {turn['assistant']}\n\n"

            prompt = f"""
You are a helpful assistant that answers based on the provided context and conversation history.

Conversation history:
{history_text}

Context:
{context}

User question:
{user_query}

Answer clearly and concisely:
"""

            logging.info("✅ Prompt created")

            # ================= Generate Response ================= #
            logging.info("Generating response from LLM")
            raw_response = self.llm_pipeline(prompt)[0]["generated_text"].strip()

            # safety truncation if model roleplays
            response = raw_response.split("USER:")[0].split("ASSISTANT:")[0].strip()

            # ================= Save Conversation ================= #
            logging.info("Saving this conversation to history")
            self.conversation_history.append({"user": user_query, "assistant": response})

            return response

        except Exception as e:
            raise MyException(e, sys) from e
