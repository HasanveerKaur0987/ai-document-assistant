from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

class EmbeddingManager:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, texts):
        return self.model.encode(
            texts,
            show_progress_bar= False
        )