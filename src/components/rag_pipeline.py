from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.logger import logging
from src.exception import MyException
import sys
import torch

# Store conversation history
conversation_history = []

try:
    logging.info("RAG pipeline started")

    # ================= Embedding model ================= #
    EMBEDDING_MODEL_PATH = "/home/saif/Desktop/pre_trained_llms/bge-base-en-v1.5"
    logging.info(f"Loading embedding model from {EMBEDDING_MODEL_PATH}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    # ================= ChromaDB Setup ================= #
    logging.info("Connecting to ChromaDB")
    client = chromadb.PersistentClient(path="/home/saif/Desktop/pdf_rag/chroma_db")
    collection = client.get_collection("pdf_chunks")

    # ================= LLaMA/Gemma Model ================= #
    LLM_PATH = "/home/saif/Desktop/pre_trained_llms/gemma-3-270m-it"
    logging.info(f"Loading LLM model from {LLM_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_PATH,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    llm_pipeline = pipeline(
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

    # ================= Loop for Conversation ================= #
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            print("⚠️ Query cannot be empty!")
            continue

        logging.info("Encoding query")
        query_embedding = embedding_model.encode(query)
        logging.info("✅ Query encoded")

        logging.info("Retrieving context from ChromaDB")
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=10
        )
        retrieved_chunks = sum(results["documents"], [])
        context = "\n".join(retrieved_chunks)
        logging.info("✅ Context retrieved from ChromaDB")

        # ================= Build Instruction Prompt ================= #
        logging.info("Creating prompt for LLM")

        history_text = ""
        for turn in conversation_history[-5:]:  # keep last 5 turns
            history_text += f"User asked: {turn['user']}\nAssistant replied: {turn['assistant']}\n\n"

        prompt = f"""
You are a helpful assistant that answers based on the provided context and conversation history.

Conversation history:
{history_text}

Context:
{context}

User question:
{query}

Answer clearly and concisely:
"""

        logging.info("✅ Prompt created")

        # ================= Generate Response ================= #
        logging.info("Generating response from given prompt")
        raw_response = llm_pipeline(prompt)[0]["generated_text"].strip()

        # safety truncation if model roleplays
        response = raw_response.split("USER:")[0].split("ASSISTANT:")[0].strip()

        print("\nResponse:\n", response)

        # ================= Save Conversation ================= #
        logging.info("Saving this conversation to history")
        conversation_history.append({"user": query, "assistant": response})

except Exception as e:
    raise MyException(e, sys) from e
