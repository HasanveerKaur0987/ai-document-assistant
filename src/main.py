from embeddings import EmbeddingManager
from vector_store import VectorStoreManager
from retriever import Retriever
from llm import generate_answer, rewrite_question, NOT_FOUND
from pdf_processor import process_pdf
from logger import get_logger

logger = get_logger(__name__)


embedding_manager = EmbeddingManager()
vector_store = VectorStoreManager()
retriever = Retriever(vector_store, embedding_manager)


def add_document(file_path, file_hash, file_name):

    # 1. process pdf
    chunks = process_pdf(file_path)

    # Drop empty chunks (scanned pages have no extractable text)
    chunks = [c for c in chunks if c.page_content.strip()]

    if not chunks:
        raise ValueError(
            "No readable text found. This PDF may be scanned or image-only."
        )

    for chunk in chunks:
        chunk.metadata["file_hash"] = file_hash
        chunk.metadata["file_name"] = file_name

    # 2. extract text from each chunk
    text = [chunk.page_content for chunk in chunks]

    # 3. create embeddings
    embeddings = embedding_manager.create_embeddings(text)

    # 4. store in ChromaDB
    vector_store.add_documents(chunks, embeddings)

    # 5. keep the keyword index in sync
    retriever.refresh_index()

    logger.info("Indexed %s (%d chunks)", file_name, len(chunks))

    return len(chunks)


def list_documents():
    return vector_store.list_documents()


def delete_document(file_hash):
    deleted = vector_store.delete_document(file_hash)
    retriever.refresh_index()
    return deleted


def prepare_question(question, chat_history=None, file_hash=None):
    """
    Rewrite the question and retrieve chunks.
    Returns (standalone_question, retrieved_documents).
    """

    try:
        standalone_question = rewrite_question(question, chat_history or [])
    except Exception:
        logger.exception("Question rewrite failed, using original question")
        standalone_question = question

    logger.info("Standalone question: %s", standalone_question)

    retrieved_documents = retriever.retrieve(
        standalone_question,
        file_hash=file_hash
    )

    for i, document in enumerate(retrieved_documents):
        logger.info(
            "Doc %d: %s p.%d | found by %s | rerank %.2f",
            i + 1,
            document["metadata"].get("file_name"),
            document["metadata"].get("page", 0) + 1,
            document["found_by"],
            document["rerank_score"],
        )

    return standalone_question, retrieved_documents


def ask_question(question, chat_history=None, file_hash=None):
    """Non-streaming version (used by tests and scripts)."""

    standalone_question, retrieved_documents = prepare_question(
        question, chat_history, file_hash
    )

    # Nothing relevant: skip the LLM call
    if not retrieved_documents:
        return {"answer": NOT_FOUND, "sources": []}

    try:
        answer = generate_answer(standalone_question, retrieved_documents)
    except Exception:
        logger.exception("Answer generation failed")
        return {
            "answer": "Sorry, I could not generate an answer right now. "
                      "Please try again in a moment.",
            "sources": []
        }

    # If the model says it couldn't find it, show no sources
    if NOT_FOUND in answer:
        retrieved_documents = []

    return {"answer": answer, "sources": retrieved_documents}