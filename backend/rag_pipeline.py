from src.components.document_loader import PDFExtractor
from src.components.preprocessing_txt import Preprocessing

class RAGPipeline:
    def __init__(self, pdf_path: str, text_path: str, cleaned_text_path: str):
        self.pdf_path = pdf_path
        self.text_path = text_path
        self.cleaned_text_path = cleaned_text_path
    def preprocess_text(self):
        extractor = PDFExtractor(self.pdf_path, self.text_path)
        extractor.extract_text_from_pdf()

        preprocessor = Preprocessing(self.text_path, self.cleaned_text_path)
        preprocessor.clean_and_preprocess_text()
        return
