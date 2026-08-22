import json
import re
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CLAUSES_FILE = Path("data/clauses.json")

MODEL_NAME = "all-MiniLM-L6-v2"


def tokenize(text):
    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


class HybridSearch:

    def __init__(self):

        # -----------------------------
        # Load clauses
        # -----------------------------

        with CLAUSES_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            self.clauses = json.load(file)

        # -----------------------------
        # BM25
        # -----------------------------

        self.tokenized_clauses = [
            tokenize(clause["text"])
            for clause in self.clauses
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_clauses
        )

        # -----------------------------
        # Semantic model
        # -----------------------------

        print("Loading semantic model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        texts = [
            clause["text"]
            for clause in self.clauses
        ]

        print("Creating semantic index...")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        print(
            f"Loaded {len(self.clauses)} policy clauses."
        )

    def search(self, question, top_k=5):

        # =========================================
        # 1. BM25 scores
        # =========================================

        query_tokens = tokenize(question)

        bm25_scores = self.bm25.get_scores(
            query_tokens
        )

        # =========================================
        # 2. Semantic scores
        # =========================================

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        semantic_scores, _ = self.index.search(
            query_embedding,
            len(self.clauses)
        )

        semantic_scores = semantic_scores[0]

        # =========================================
        # 3. Normalize BM25
        # =========================================

        max_bm25 = max(bm25_scores)

        if max_bm25 > 0:

            normalized_bm25 = [
                score / max_bm25
                for score in bm25_scores
            ]

        else:

            normalized_bm25 = [
                0.0
                for _ in bm25_scores
            ]

        # =========================================
        # 4. Normalize semantic scores
        # =========================================

        min_semantic = min(semantic_scores)
        max_semantic = max(semantic_scores)

        if max_semantic > min_semantic:

            normalized_semantic = [
                (
                    score - min_semantic
                )
                /
                (
                    max_semantic - min_semantic
                )
                for score in semantic_scores
            ]

        else:

            normalized_semantic = [
                0.0
                for _ in semantic_scores
            ]

        # =========================================
        # 5. Combine scores
        # =========================================

        combined_results = []

        for i, clause in enumerate(
            self.clauses
        ):

            hybrid_score = (
                0.4 * normalized_bm25[i]
                +
                0.6 * normalized_semantic[i]
            )

            result = clause.copy()

            result["bm25_score"] = float(
                bm25_scores[i]
            )

            result["semantic_score"] = float(
                semantic_scores[i]
            )

            result["hybrid_score"] = float(
                hybrid_score
            )

            combined_results.append(
                result
            )

        # =========================================
        # 6. Rank by hybrid score
        # =========================================

        combined_results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return combined_results[:top_k]


def main():

    searcher = HybridSearch()

    print("\n" + "=" * 60)
    print("HYBRID POLICY SEARCH")
    print("=" * 60)

    question = input(
        "\nEnter your question: "
    )

    results = searcher.search(
        question,
        top_k=5
    )

    print("\nTop matching clauses:\n")

    for result in results:

        print("-" * 60)

        print(
            f"Clause: {result['clause']}"
        )

        print(
            f"BM25: {result['bm25_score']:.3f}"
        )

        print(
            f"Semantic: {result['semantic_score']:.3f}"
        )

        print(
            f"Hybrid: {result['hybrid_score']:.3f}"
        )

        print(
            f"Text: {result['text']}"
        )


if __name__ == "__main__":
    main()
    