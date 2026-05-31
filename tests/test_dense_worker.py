from __future__ import annotations

import unittest

import numpy as np

from src.retrievers.dense_worker import DenseWorkerError, search_batch


class FakeModel:
    def encode(self, queries, **kwargs):
        return np.asarray([[float(len(query)), 1.0] for query in queries], dtype="float32")


class FakeStore:
    def search(self, embedding, top_k):
        return [{"chunk_id": f"chunk_{int(embedding[0])}", "rank": 1, "score": float(top_k)}]


class DenseWorkerTest(unittest.TestCase):
    def test_full_mode_returns_search_results(self) -> None:
        response = search_batch(
            model=FakeModel(),
            store=FakeStore(),
            mode="full",
            request={
                "type": "search",
                "request_id": 7,
                "queries": ["abc", "hello"],
                "top_k_chunks": 50,
                "batch_size": 2,
            },
        )

        self.assertEqual(response["request_id"], 7)
        self.assertEqual(response["results"][0][0]["chunk_id"], "chunk_3")
        self.assertEqual(response["results"][1][0]["chunk_id"], "chunk_5")

    def test_model_only_mode_returns_embeddings(self) -> None:
        response = search_batch(
            model=FakeModel(),
            store=None,
            mode="model_only",
            request={
                "type": "search",
                "request_id": 8,
                "queries": ["abc"],
                "top_k_chunks": 50,
                "batch_size": 1,
            },
        )

        self.assertEqual(response["embeddings"], [[3.0, 1.0]])

    def test_rejects_invalid_batch_size(self) -> None:
        with self.assertRaises(DenseWorkerError):
            search_batch(
                model=FakeModel(),
                store=None,
                mode="model_only",
                request={"queries": ["abc"], "top_k_chunks": 50, "batch_size": 0},
            )


if __name__ == "__main__":
    unittest.main()
