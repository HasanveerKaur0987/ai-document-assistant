import numpy as np
import pytest
from langchain_core.documents import Document

import retriever as retriever_module
from config import CHUNK_SIZE
from pdf_processor import split_documents
from vector_store import VectorStoreManager


@pytest.fixture
def store(tmp_path):
    """A fresh, empty vector store in a temporary folder."""
    return VectorStoreManager(persist_directory=str(tmp_path / "db"))


@pytest.fixture
def retriever(monkeypatch):
    """A Retriever without downloading the reranker model."""
    monkeypatch.setattr(retriever_module, "CrossEncoder", lambda name: object())

    class EmptyStore:
        def count(self):
            return 0

    return retriever_module.Retriever(EmptyStore(), embedding_manager=None)


def make_chunks(file_hash, count):
    return [
        Document(
            page_content=f"chunk number {i}",
            metadata={"file_hash": file_hash, "file_name": "a.pdf", "page": 0},
        )
        for i in range(count)
    ]


def fake_embeddings(count):
    return np.random.rand(count, 8)


# ---------- Vector store ----------

def test_reuploading_same_file_does_not_duplicate(store):
    chunks = make_chunks("hash1", 3)

    store.add_documents(chunks, fake_embeddings(3))
    store.add_documents(chunks, fake_embeddings(3))
    store.add_documents(chunks, fake_embeddings(3))

    assert store.count() == 3


def test_delete_removes_only_that_file(store):
    store.add_documents(make_chunks("hash1", 3), fake_embeddings(3))
    store.add_documents(make_chunks("hash2", 2), fake_embeddings(2))

    store.delete_document("hash1")

    assert store.count() == 2


# ---------- Retriever ----------

def test_empty_store_returns_no_documents(retriever):
    assert retriever.retrieve("anything") == []


def test_tokenize_keeps_codes_searchable():
    assert retriever_module.tokenize("What does ERR-4031 mean?") == [
        "what", "does", "err", "4031", "mean"
    ]


def test_merge_ranks_chunk_found_by_both_searches_first(retriever):
    # "b" appears in both lists, so it should beat "a" and "c"
    merged = retriever.merge_results(
        vector_ids=["a", "b"],
        keyword_ids=["c", "b"],
    )

    assert merged[0] == "b"


# ---------- Chunking ----------

def test_chunks_respect_size_limit():
    text = "word " * 2000
    pages = [Document(page_content=text, metadata={"page": 0})]

    chunks = split_documents(pages)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)