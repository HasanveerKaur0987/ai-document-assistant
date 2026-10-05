import chromadb
from config import VECTOR_STORE_DIR, COLLECTION_NAME
from logger import get_logger

logger = get_logger(__name__)

class VectorStoreManager:
    def __init__ (self, persist_directory = VECTOR_STORE_DIR, collection_name=COLLECTION_NAME):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path = self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Vector store for PDF document embeddings"
            }
        )

        logger.info("Vector store ready. Chunks in collection: %d", self.collection.count())

    def add_documents(self, documents, embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings.")

        if not documents:
            return

        file_hash = documents[0].metadata["file_hash"]

        # remove old chunks of this file (if any)
        self.delete_document(file_hash)

        # same file -> same ids every time
        ids = [f"{file_hash}_{i}" for i in range(len(documents))]
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        logger.info("Added %d chunks to ChromaDB", len(documents))

    def count(self):
        """Return the number of stored documents."""
        return self.collection.count()

    def delete_document(self, file_hash):

        results = self.collection.get(
            where={"file_hash": file_hash}
        )
        ids = results["ids"]

        if ids:
            self.collection.delete(ids=ids)

        logger.info("Deleted %d chunks from ChromaDB", len(ids))

        return len(ids)

    def list_documents(self):
    # Return {file_hash: {"name": ..., "chunks": ...}} for everything stored.
        results = self.collection.get(include=["metadatas"])

        docs = {}
        for metadata in results["metadatas"] or []:
            file_hash = metadata.get("file_hash")
            if not file_hash:
                continue
            if file_hash not in docs:
                docs[file_hash] = {
                    "name": metadata.get("file_name", "Unknown"),
                    "chunks": 0
                }
            docs[file_hash]["chunks"] += 1

        return docs

    