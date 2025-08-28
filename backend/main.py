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
    pipeline = RAGPipeline(pdf_path=PDF_DIR, text_path=TEXT_DIR)
    pipeline.preprocess_text()

    return f"""
    <html>
        <body>
            <h2>✅ Successfully uploaded and processed {file.filename} and extracted text saved to {TEXT_DIR}!</h2>
            <a href="/">Go back</a>
        </body>
    </html>
    """
