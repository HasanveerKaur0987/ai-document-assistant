from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(file_path):
    """
    Load a PDF and convert each page into a LangChain Document.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents


def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split documents into smaller chunks for the RAG pipeline.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    return chunks


def process_pdf(file_path):
    documents = load_pdf(file_path)
    chunks = split_documents(documents)

    return chunks




