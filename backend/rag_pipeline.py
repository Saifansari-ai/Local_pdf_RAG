from src.components.document_loader import PDFExtractor
from src.components.preprocessing_txt import Preprocessing
from src.components.chunk import Chunk
from src.components.embed_gen import EmbeddingGenerator

class RAGPipeline:
    def __init__(self, pdf_path: str, text_path: str, cleaned_text_path: str,chunk_file: str,chunk_size: int, chunk_overlap: int,source_name: str,embed_model_path: str,embeddings_path: str):
        self.pdf_path = pdf_path
        self.text_path = text_path
        self.cleaned_text_path = cleaned_text_path
        self.chunk_file = chunk_file
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.source_name = source_name
        self.embed_model_path = embed_model_path
        self.embeddings_path = embeddings_path

    def preprocess_text(self):
        extractor = PDFExtractor(self.pdf_path, self.text_path)
        extractor.extract_text_from_pdf()

        preprocessor = Preprocessing(self.text_path, self.cleaned_text_path)
        preprocessor.clean_and_preprocess_text()

        chunker = Chunk(self.cleaned_text_path, self.chunk_file, self.chunk_size, self.chunk_overlap, self.source_name)
        chunker.main()

        emb_genrator = EmbeddingGenerator(self.embed_model_path, self.chunk_file, self.embeddings_path)
        emb_genrator.Genrate()
        return
