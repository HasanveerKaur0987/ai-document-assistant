import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import TOP_K, FETCH_K, RERANK_THRESHOLD, RERANKER_MODEL

RRF_K = 60   # standard constant for Reciprocal Rank Fusion

def tokenize(text):  
    # text into lowercase words
    return re.findall(r"\w+", text.lower())

class Retriever:
    """
    Finds the best chunks for a question.
    Uses two searches (by meaning and by keyword), merges them,
    then reranks the results with a cross-encoder.
    """
    
    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # Keyword index (built from all chunks stored in the database)
        self.all_chunks = []
        self.bm25 = None

    # Keyword index

    def refresh_index(self):
        """
        Rebuild the keyword index from everything in the database.
        Must be called after documents are added or deleted.
        """
        data = self.vector_store.collection.get(
            include = ["documents", "metadatas"]
        )

        self.all_chunks = []

        for chunk_id, text, metadata in zip(
            data["ids"], data["documents"], data["metadatas"]
        ):
            self.all_chunks.append({
                "id": chunk_id,
                "text": text,
                "metadata": metadata
            })

        if self.all_chunks:
            word_lists = [tokenize(chunk["text"]) for chunk in self.all_chunks]
            self.bm25 = BM25Okapi(word_lists)
        else:
            self.bm25 = None


    # Search 1. By Meaning

    def vector_search(self, query, n, file_hash=None):
        """Return (chunk ids, distances) of the n closest chunks by meaning."""

        query_embedding = self.embedding_manager.create_embeddings([query])[0]

        search_args = {
            "query_embeddings": [query_embedding],
            "n_results": n
        }

        if file_hash:
            search_args["where"] = {"file_hash": file_hash}

        results = self.vector_store.collection.query(**search_args)

        return results["ids"][0], results["distances"][0]

    # Search 2. By keyword

    def keyword_search(self, query, n, file_hash=None):
        """Return the ids of the n chunks that best match the query's words."""

        # One BM25 score per chunk (higher = more matching words)
        scores = self.bm25.get_scores(tokenize(query))

        best_first = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
        )

        keyword_ids = []

        for i in best_first:

            # Score 0 means no query word appears in this chunk
            if scores[i] <= 0:
                break

            chunk = self.all_chunks[i]

            # Skip chunks from other documents when a filter is set
            if file_hash and chunk["metadata"].get("file_hash") != file_hash:
                continue

            keyword_ids.append(chunk["id"])

            if len(keyword_ids) >= n:
                break

        return keyword_ids

    def merge_results(self, vector_ids, keyword_ids):
        """
        Reciprocal Rank Fusion: every chunk gets points for its position
        in each list. A chunk found by both searches gets both sets of points.
        Returns chunk ids sorted from most points to fewest.
        """
        points = {}

        for rank, chunk_id in enumerate(vector_ids):
            points[chunk_id] = points.get(chunk_id, 0) + 1 / (RRF_K + rank + 1)

        for rank, chunk_id in enumerate(keyword_ids):
            points[chunk_id] = points.get(chunk_id, 0) + 1 / (RRF_K + rank + 1)

        return sorted(points, key=points.get, reverse=True)


    def retrieve(self, query, top_k=TOP_K, fetch_k=FETCH_K, file_hash=None):
        """
        1. Search by meaning and by exact words
        2. Merge the two result lists
        3. Rerank the merged chunks with the cross-encoder
        4. Drop weak chunks, return the best top_k
        """

        # Nothing stored yet
        if self.vector_store.count() == 0:
            return []

        # Build the keyword index the first time
        if self.bm25 is None:
            self.refresh_index()

        if not self.all_chunks:
            return []

        # How many chunks can we search? (all, or just one document's)
        if file_hash:
            available = sum(
                1 for c in self.all_chunks
                if c["metadata"].get("file_hash") == file_hash
            )
        else:
            available = len(self.all_chunks)

        if available == 0:
            return []

        n = min(fetch_k, available)

        # 1. Two searches
        vector_ids, vector_distances = self.vector_search(query, n, file_hash)
        keyword_ids = self.keyword_search(query, n, file_hash)

        # 2. Merge
        merged_ids = self.merge_results(vector_ids, keyword_ids)[:fetch_k]

        # Look up each chunk by its id
        chunk_by_id = {c["id"]: c for c in self.all_chunks}
        distance_by_id = dict(zip(vector_ids, vector_distances))

        candidates = []

        for chunk_id in merged_ids:
            chunk = chunk_by_id[chunk_id]

            # Which search found this chunk? (useful for debugging and demos)
            in_vector = chunk_id in vector_ids
            in_keyword = chunk_id in keyword_ids

            if in_vector and in_keyword:
                found_by = "both"
            elif in_vector:
                found_by = "vector"
            else:
                found_by = "keyword"

            candidates.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "distance": distance_by_id.get(chunk_id),
                "found_by": found_by
            })

        if not candidates:
            return []

        # 3. Rerank: score each (question, chunk) pair
        scores = self.reranker.predict(
            [(query, c["text"]) for c in candidates]
        )

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

        # 4. Drop weak chunks, keep the best top_k
        candidates = [
            c for c in candidates
            if c["rerank_score"] >= RERANK_THRESHOLD
        ]

        return candidates[:top_k]
