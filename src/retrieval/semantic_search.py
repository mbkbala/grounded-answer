import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


CLAUSES_FILE = Path("data/clauses.json")
INDEX_FILE = Path("data/policy.index")


MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticSearch:

    def __init__(self):

        # Load policy clauses
        with CLAUSES_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            self.clauses = json.load(file)

        print("Loading semantic model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        # Create embeddings for all clauses
        texts = [
            clause["text"]
            for clause in self.clauses
        ]

        print("Creating embeddings...")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # FAISS index
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        print(
            f"Indexed {len(self.clauses)} clauses."
        )

    def search(self, question, top_k=5):

        # Convert question into embedding
        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Search
        scores, indexes = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indexes[0]
        ):

            clause = self.clauses[index].copy()

            clause["score"] = float(score)

            results.append(clause)

        return results


def main():

    searcher = SemanticSearch()

    print("\n" + "=" * 60)
    print("SEMANTIC POLICY SEARCH")
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
            f"Score: {result['score']:.3f}"
        )

        print(
            f"Text: {result['text']}"
        )


if __name__ == "__main__":
    main()
    