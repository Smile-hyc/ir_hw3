import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from text_utils import load_documents, split_text, tokenize


class CourseSearchEngine:
    def __init__(self, docs_folder):
        self.docs_folder = docs_folder
        self.documents = []
        self.chunks = []
        self.tokenized_chunks = []
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def build(self):
        self.documents = load_documents(self.docs_folder)
        self.chunks = []

        chunk_index = 1
        for doc in self.documents:
            for chunk_text in split_text(doc["text"]):
                self.chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_index}",
                        "source_file": doc["source_file"],
                        "text": chunk_text,
                    }
                )
                chunk_index += 1

        self.tokenized_chunks = [tokenize(chunk["text"]) for chunk in self.chunks]

        if not self.chunks:
            self.bm25 = None
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            return

        self.bm25 = BM25Okapi(self.tokenized_chunks)
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def get_query_terms(self, query):
        return tokenize(query)

    def normalize_scores(self, scores):
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            if max_score > 0:
                return np.ones_like(scores)
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def search_bm25(self, query, top_k=5):
        bm25_scores, tfidf_scores = self._score_all(query)
        bm25_norm = self.normalize_scores(bm25_scores)
        results = self._make_results(
            query=query,
            bm25_scores=bm25_scores,
            bm25_norm=bm25_norm,
            tfidf_scores=tfidf_scores,
            hybrid_scores=bm25_norm,
            top_k=top_k,
        )
        return results

    def search_tfidf(self, query, top_k=5):
        bm25_scores, tfidf_scores = self._score_all(query)
        bm25_norm = self.normalize_scores(bm25_scores)
        results = self._make_results(
            query=query,
            bm25_scores=bm25_scores,
            bm25_norm=bm25_norm,
            tfidf_scores=tfidf_scores,
            hybrid_scores=tfidf_scores,
            top_k=top_k,
        )
        return results

    def search_hybrid(self, query, top_k=5, alpha=0.5):
        bm25_scores, tfidf_scores = self._score_all(query)
        bm25_norm = self.normalize_scores(bm25_scores)
        tfidf_norm = self.normalize_scores(tfidf_scores)
        hybrid_scores = alpha * bm25_norm + (1 - alpha) * tfidf_norm
        results = self._make_results(
            query=query,
            bm25_scores=bm25_scores,
            bm25_norm=bm25_norm,
            tfidf_scores=tfidf_scores,
            hybrid_scores=hybrid_scores,
            top_k=top_k,
        )
        return results

    def _score_all(self, query):
        if not self.chunks or not query.strip():
            zeros = np.zeros(len(self.chunks), dtype=float)
            return zeros, zeros

        query_terms = self.get_query_terms(query)
        if not query_terms:
            zeros = np.zeros(len(self.chunks), dtype=float)
            return zeros, zeros

        bm25_scores = self.bm25.get_scores(query_terms) if self.bm25 else np.zeros(len(self.chunks))

        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            tfidf_scores = np.zeros(len(self.chunks), dtype=float)
        else:
            query_vector = self.tfidf_vectorizer.transform([query])
            tfidf_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        return np.asarray(bm25_scores, dtype=float), np.asarray(tfidf_scores, dtype=float)

    def _make_results(
        self,
        query,
        bm25_scores,
        bm25_norm,
        tfidf_scores,
        hybrid_scores,
        top_k,
    ):
        query_terms = self.get_query_terms(query)
        scored_indices = np.argsort(hybrid_scores)[::-1]

        results = []
        for rank, index in enumerate(scored_indices[:top_k], start=1):
            chunk = self.chunks[index]
            chunk_terms = set(self.tokenized_chunks[index])
            matched_terms = [term for term in query_terms if term in chunk_terms or term in chunk["text"]]

            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "source_file": chunk["source_file"],
                    "text": chunk["text"],
                    "bm25_score": float(bm25_scores[index]),
                    "bm25_norm": float(bm25_norm[index]),
                    "tfidf_score": float(tfidf_scores[index]),
                    "hybrid_score": float(hybrid_scores[index]),
                    "matched_terms": matched_terms,
                }
            )

        return results
