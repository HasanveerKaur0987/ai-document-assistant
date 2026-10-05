# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval 
TOP_K = 5                  # chunks sent to the LLM
FETCH_K = 25               # candidates fetched before reranking
RERANK_THRESHOLD = -7.0    # Kept low on purpose: some real answers score below -5. The LLM decides the final "not found".

# Models 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "gpt-5.6-luna"

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Storage
VECTOR_STORE_DIR = str(PROJECT_ROOT / "data" / "vector_store")
COLLECTION_NAME = "pdf_documents"