import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.rag_pipeline import RAGPipeline
from backend.rag_query import RAGQueryEngine  
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PDF_DIR = "web_data/pdfs"
TEXT_DIR = "web_data/text"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

CLEANED_TEXT_DIR = "web_data/cleaned_text"
os.makedirs(CLEANED_TEXT_DIR, exist_ok=True)

CHUNK_FILE = "web_data/chunks"
os.makedirs(CHUNK_FILE, exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SOURCE_NAME = "merged_text"

EMBED_MODEL_PATH = "/home/saif/Desktop/pre_trained_llms/bge-base-en-v1.5"
EMBEDDINGS_PATH = "web_data/embeddings"
os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

CHROMA_DB_PATH = "/home/saif/Desktop/pdf_rag/chromadb"

LLM_MODEL_PATH = "/home/saif/Desktop/pre_trained_llms/gemma-3-270m-it"


# Serve frontend directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")


@app.post("/upload_pdf/", response_class=HTMLResponse)
async def upload_pdf(file: UploadFile = File(...)):
    # Save uploaded file
    file_path = os.path.join(PDF_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Run RAGPipeline to extract + store embeddings
    pipeline = RAGPipeline(
        pdf_path=PDF_DIR,
        text_path=TEXT_DIR,
        cleaned_text_path=CLEANED_TEXT_DIR,
        chunk_file=CHUNK_FILE,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        source_name=SOURCE_NAME,
        embed_model_path=EMBED_MODEL_PATH,
        embeddings_path=EMBEDDINGS_PATH,
        chroma_db_path=CHROMA_DB_PATH
    )
    pipeline.preprocess_text()

    return """
    <html>
        <body>
            <h2>✅ Successfully uploaded</h2>
            <a href="/">Go back</a>
        </body>
    </html>
    """


@app.post("/query/")
async def query_document(question: str = Form(...)):
    # Instantiate query engine only when needed
    try:
        query_engine = RAGQueryEngine(
            embedding_model_path=EMBED_MODEL_PATH,
            chroma_db_path=CHROMA_DB_PATH,
            llm_model_path=LLM_MODEL_PATH
        )
        answer = query_engine.query(question)
        return {"answer": answer}
    except Exception as e:
        return {"error": "No documents available. Please upload a PDF first."}
