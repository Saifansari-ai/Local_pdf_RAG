# Local_pdf_RAG

A **Retrieval-Augmented Generation (RAG)** system for local PDF document querying using **Gemma 3 or llama-3.2-1b-instruct**, a locally hosted LLM.

---

##  Problem Statement

Managing and searching through local PDF documents—like research papers, books, technical specs—can be cumbersome. Existing tools often export content online or rely on external APIs.

This project solves the problem of:
- **Offline querying** of PDFs securely and privately.
- Efficiently retrieving relevant content from large pools of PDF files.
- Reducing dependency on cloud services by using a local LLM (**Gemma 3 or llama-3.2-1b-instruct**).

---

##  What You’re Building

A fully local RAG system that:
1. **Loads PDF files** from your local disk.
2. **Parses and chunks** them into document sections.
3. **Embeds chunks** using a vector representation (via sentence-transformers or similar).
4. **Indexes embeddings** using FAISS (or an equivalent vector database).
5. **Creates a query interface**—command-line, API, or UI—to ask questions over the PDFs.
6. **Generates answers** using the Gemma 3 or llama-3.2-1b-instruct LLM running locally.

All of this happens **offline**, preserving privacy and running entirely on your hardware.

---

##  How It’s Built

| Component        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **PDF Parsing**  | Extract raw text from PDFs using `PyPDF2`, `pypdf`, or `unstructured`.      |
| **Text Chunking**| Split text into manageable overlapping chunks for better context retrieval. |
| **Embeddings**   | Convert chunks into embeddings using sentence-transformers like `bge-base-en-v1.5` or local encoders. |
| **Vector DB**    | Store and search embeddings via FAISS or a lightweight alternative.          |
| **LLM Serving**  | Run `Gemma 3 or llama-3.2-1b-instruct` locally (e.g., via Ollama, llama.cpp, or a Docker container).   |
| **RAG Logic**    | Retrieve top chunks, format prompt, and generate response using LLM.         |
| **Interface**    | Basic CLI, fastAPI endpoint, or local UI to interact with the RAG system.    |

---