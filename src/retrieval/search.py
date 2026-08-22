import json
import re

from pathlib import Path

from rank_bm25 import BM25Okapi


CLAUSES_FILE = Path("data/clauses.json")


def tokenize(text):
    """
    Convert text into simple searchable words.
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


class PolicySearch:

    def __init__(self):

        # Load our 137 clauses
        with CLAUSES_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            self.clauses = json.load(file)

        # Tokenize every clause
        self.tokenized_clauses = [
            tokenize(clause["text"])
            for clause in self.clauses
        ]

        # Create BM25 search index
        self.bm25 = BM25Okapi(
            self.tokenized_clauses
        )

    def search(self, question, top_k=5):

        # Convert question into words
        query_tokens = tokenize(question)

        # Search
        scores = self.bm25.get_scores(
            query_tokens
        )

        # Get highest scoring clause indexes
        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indexes[:top_k]:

            clause = self.clauses[index].copy()

            clause["score"] = float(
                scores[index]
            )

            results.append(clause)

        return results


def main():

    searcher = PolicySearch()

    print("=" * 60)
    print("CALDER COUNTY POLICY SEARCH")
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