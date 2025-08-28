from src.components.document_loader import PDFExtractor

class RAGPipeline:
    def __init__(self, pdf_path: str, text_path: str):
        self.pdf_path = pdf_path
        self.text_path = text_path
    def preprocess_text(self):
        extractor = PDFExtractor(self.pdf_path, self.text_path)
        extractor.extract_text_from_pdf()
        return
