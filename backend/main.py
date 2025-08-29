import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.rag_pipeline import RAGPipeline

app = FastAPI()

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

    # Run RAGPipeline to extract text
    pipeline = RAGPipeline(pdf_path=PDF_DIR, text_path=TEXT_DIR, cleaned_text_path=CLEANED_TEXT_DIR,chunk_file=CHUNK_FILE,chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,source_name=SOURCE_NAME)
    pipeline.preprocess_text()

    return f"""
    <html>
        <body>
            <h2>✅ Successfully uploaded</h2>
            <a href="/">Go back</a>
        </body>
    </html>
    """
