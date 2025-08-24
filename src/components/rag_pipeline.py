from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.logger import logging
from src.exception import MyException
import sys
import torch

try:
    logging.info("RAG pipeline started")

    # ================= Embedding model ================= #
    EMBEDDING_MODEL_PATH = "/home/saif/Desktop/pre_trained_llms/bge-base-en-v1.5"
    logging.info(f"Loading embedding model from {EMBEDDING_MODEL_PATH}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    query = "In the book Bigger Leaner Stronger, what are the top 5 takeaways for the person to know to get the benefits of calisthenics?"
    query_embedding = embedding_model.encode(query)

    # ================= ChromaDB Retrieval ================= #
    logging.info("Connecting to ChromaDB and retrieving chunks based on query embedding")
    client = chromadb.PersistentClient(path="/home/saif/Desktop/pdf_rag/chroma_db")
    collection = client.get_collection("pdf_chunks")

    # Retrieve top 3 most similar chunks (instead of just 1)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(retrieved_chunks)
    logging.info("Retrieved context injected into the prompt")

    # ================= LLaMA Model ================= #
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
        device_map="auto"
    )
    logging.info("Local LLM loaded successfully")

    # ================= Prompt with context ================= #
    prompt = f"""
    You are a helpful assistant. Answer the question using only the provided context.
    If the answer is not in the context, say "I couldn't find this in the provided documents."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # ================= Generate Response ================= #
    logging.info("Generating response")
    response = llm_pipeline(prompt)[0]["generated_text"]

    # if "<|assistant|>" in response:
    #     response = response.split("<|assistant|>")[-1].strip()

    logging.info("Read Generated response below \n\n")
    # print("Query:", query)
    # print("\nRetrieved Context:\n", context)
    print("\nResponse:\n", response)

except Exception as e:
    raise MyException(e, sys) from e
