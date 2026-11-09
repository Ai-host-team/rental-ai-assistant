"""Offline tests of notebook calculations: no model downloads and no CUDA."""

import copy
import ast
import gc
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "notebooks/01_eda_dataset.ipynb").read_text())
CORE = {}
for _cell in NOTEBOOK["cells"]:
    if "embeddings-core" in _cell.get("metadata", {}).get("tags", []):
        exec(compile("".join(_cell["source"]), "embedding-core-cell", "exec"), CORE)


def toy_rows(n=8):
    return pd.DataFrame({
        "request_id": [f"r{i:03}" for i in range(n)],
        "model_text": [f"Обращение {i}" for i in range(n)],
        "text_key": [f"обращение {i}" for i in range(n)],
        "intent": [f"intent_{i // 4}" for i in range(n)],
        "category": ["category"] * n,
        "scenario_group_id": [f"g{i // 2}" for i in range(n)],
        "source_dataset": ["source_a" if i % 2 else "source_b" for i in range(n)],
        "noise_type": ["none"] * n,
        "language": ["ru"] * n,
    })


def unit_vectors(n, dimension=9, seed=42):
    values = np.random.default_rng(seed).normal(size=(n, dimension)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


class EmbeddingCoreTests(unittest.TestCase):
    def test_mapping_preserves_repetitions_and_does_not_mutate_input(self):
        rows = toy_rows()
        extra = rows.iloc[[0]].copy()
        extra["request_id"] = "r999"
        extra["source_dataset"] = "other_source"
        rows = pd.concat([rows, extra], ignore_index=True)
        before = rows.copy(deep=True)
        unique, mapping = CORE["prepare_embedding_rows"](rows)
        self.assertEqual(len(unique), 8)
        self.assertEqual(len(mapping), 9)
        self.assertEqual(unique.iloc[0]["multiplicity"], 2)
        self.assertEqual(unique.iloc[0]["source_display"], "несколько источников")
        pd.testing.assert_frame_equal(rows, before)
        self.assertEqual(mapping["embedding_index"].nunique(), 8)

    def test_conflicting_exact_texts_are_rejected(self):
        rows = toy_rows()
        rows.loc[1, "model_text"] = rows.loc[0, "model_text"]
        rows.loc[1, "scenario_group_id"] = "conflicting_group"
        with self.assertRaisesRegex(ValueError, "Одинаковые тексты"):
            CORE["prepare_embedding_rows"](rows)

    def test_only_ru_rows_are_selected(self):
        rows = toy_rows()
        rows.loc[0, "language"] = "en"
        unique, mapping = CORE["prepare_embedding_rows"](rows)
        self.assertEqual(len(unique), 7)
        self.assertNotIn("r000", mapping["request_id"].tolist())

    def test_cache_key_covers_order_revision_parameters_and_versions(self):
        base = (["а", "б"], ["r1", "r2"], "model", "revision", 128, {"torch": "v1"})
        original = CORE["embedding_cache_key"](*base)
        for position, replacement in [(0, ["б", "а"]), (1, ["r2", "r1"]),
                                      (2, "other-model"), (3, "new-revision"),
                                      (4, 256), (5, {"torch": "v2"})]:
            changed = list(base)
            changed[position] = replacement
            self.assertNotEqual(original, CORE["embedding_cache_key"](*changed))

    def test_cache_roundtrip_checks_ids_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            vectors = unit_vectors(3)
            CORE["save_embedding_cache"](directory, vectors, {"cache_key": "key"}, ["a", "b", "c"])
            actual, _ = CORE["load_embedding_cache"](directory, "key", ["a", "b", "c"])
            np.testing.assert_array_equal(vectors, actual)
            with self.assertRaises(ValueError):
                CORE["load_embedding_cache"](directory, "key", ["b", "a", "c"])
            with (Path(directory) / "vectors.npy").open("ab") as handle:
                handle.write(b"corrupted")
            with self.assertRaisesRegex(ValueError, "Контрольная сумма"):
                CORE["load_embedding_cache"](directory, "key", ["a", "b", "c"])

    def test_invalid_vectors_are_rejected(self):
        for vectors in [np.zeros((3, 4)), np.full((3, 4), np.nan), np.ones((2, 4))]:
            with self.assertRaises(ValueError):
                CORE["validate_vectors"](vectors, 3)

    def test_oom_retries_same_position_without_duplicates(self):
        class FakeOOM(Exception):
            pass
        expected = unit_vectors(7)
        def encode(indices):
            if len(indices) > 2:
                raise FakeOOM()
            return expected[indices]
        actual, final_batch, reductions = CORE["encode_adaptive"](
            list(range(7)), 8, encode, FakeOOM, lambda: None
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(final_batch, 2)
        self.assertEqual(reductions, [4, 2])
        with self.assertRaisesRegex(RuntimeError, "batch_size=1"):
            CORE["encode_adaptive"]([0], 1, lambda batch: (_ for _ in ()).throw(FakeOOM()), FakeOOM, lambda: None)

    def test_samples_are_reproducible_and_balanced(self):
        rows = toy_rows(40)
        rows["scenario_group_id"] = [f"g{i // 5}" for i in range(len(rows))]
        selected = CORE["visualization_indices"](rows)
        np.testing.assert_array_equal(selected, CORE["visualization_indices"](rows))
        self.assertLessEqual(rows.iloc[selected].groupby("scenario_group_id").size().max(), 3)
        queries = CORE["balanced_query_indices"](rows, limit=10)
        self.assertEqual(len(set(queries)), 10)
        self.assertEqual(rows.iloc[queries]["intent"].nunique(), 10)

    def test_block_neighbors_match_exhaustive_search_and_filters(self):
        rows = toy_rows()
        vectors = unit_vectors(len(rows))
        queries = np.array([0, 3, 6])
        observed_blocks = []
        def scores(indices):
            observed_blocks.append(len(indices))
            return vectors[indices] @ vectors.T
        actual, similarities = CORE["nearest_neighbors"](
            vectors, rows, queries, k=3, exclude_scenario=True, block_size=2, score_block=scores
        )
        exhaustive = vectors[queries] @ vectors.T
        for i, query in enumerate(queries):
            forbidden = rows["scenario_group_id"].eq(rows.iloc[query]["scenario_group_id"])
            forbidden |= rows["text_key"].eq(rows.iloc[query]["text_key"])
            exhaustive[i, forbidden.to_numpy()] = -np.inf
            expected = np.argsort(-exhaustive[i])[:3]
            np.testing.assert_array_equal(actual[i], expected)
            np.testing.assert_allclose(similarities[i], exhaustive[i, expected], atol=1e-6)
        self.assertLessEqual(max(observed_blocks), 2)
        plain, _ = CORE["nearest_neighbors"](vectors, rows, queries, k=5)
        for i, query in enumerate(queries):
            self.assertNotIn(query, plain[i])

    def test_normalized_text_key_is_excluded_across_groups(self):
        rows = toy_rows()
        rows.loc[7, "text_key"] = rows.loc[0, "text_key"]
        neighbors, _ = CORE["nearest_neighbors"](unit_vectors(8), rows, [0], k=8, exclude_scenario=True)
        self.assertNotIn(7, neighbors[0])

    def test_no_candidates_are_not_treated_as_last_row(self):
        rows = toy_rows(4)
        rows["scenario_group_id"] = "same_group"
        neighbors, scores = CORE["nearest_neighbors"](unit_vectors(4), rows, [0], k=5, exclude_scenario=True)
        self.assertTrue((neighbors == -1).all())
        self.assertTrue(np.isnan(scores).all())
        metrics, _ = CORE["neighbor_agreement"](rows, np.array([0]), neighbors)
        self.assertTrue(np.isnan(metrics["micro_agreement"]))
        self.assertEqual(metrics["queries_with_neighbors"], 0)

    def test_centroids_give_equal_weight_to_scenarios(self):
        rows = toy_rows(4)
        rows["intent"] = "one"
        rows["scenario_group_id"] = ["a", "a", "a", "b"]
        vectors = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
        names, centroids, matrix = CORE["intent_centroids"](vectors, rows)
        self.assertEqual(names, ["one"])
        np.testing.assert_allclose(centroids[0], np.array([1., 1.]) / np.sqrt(2), atol=1e-6)
        self.assertAlmostEqual(matrix[0, 0], 1., places=6)

    def test_pair_tables_are_unique_and_different_labels_are_enforced(self):
        rows = toy_rows()
        vectors = unit_vectors(8)
        queries = np.arange(len(rows))
        neighbors, scores = CORE["nearest_neighbors"](
            vectors, rows, queries, k=20, exclude_scenario=True, different_intent=True
        )
        pairs = CORE["closest_pairs"](rows, queries, neighbors, scores)
        self.assertTrue(pairs["intent_a"].ne(pairs["intent_b"]).all())
        self.assertTrue(pairs["group_a"].ne(pairs["group_b"]).all())
        identities = pairs.apply(lambda row: tuple(sorted([row["request_a"], row["request_b"]])), axis=1)
        self.assertTrue(identities.is_unique)


class NotebookTests(unittest.TestCase):
    def device_helpers(self, cuda=False, mps=False):
        namespace = {"torch": types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: cuda),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
        )}
        for cell in NOTEBOOK["cells"]:
            if "embeddings-inference" not in cell.get("metadata", {}).get("tags", []):
                continue
            tree = ast.parse("".join(cell["source"]))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in {"select_embedding_device", "calculate_model_embeddings", "is_embedding_oom"}:
                    exec(compile(ast.Module(body=[node], type_ignores=[]), "device-helper", "exec"), namespace)
        return namespace

    def test_auto_selects_cuda_then_mps_then_cpu(self):
        for cuda, mps, expected in [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]:
            helper = self.device_helpers(cuda, mps)
            self.assertEqual(helper["select_embedding_device"](), expected)
            self.assertEqual(helper["select_embedding_device"]("cpu"), "cpu")

    def test_explicit_unavailable_device_fails_before_network_access(self):
        namespace = self.device_helpers()
        for device in ["cuda", "mps"]:
            with self.assertRaisesRegex(RuntimeError, "недоступ"):
                namespace["calculate_model_embeddings"]({"device": device}, None, None)
        with self.assertRaises(ValueError):
            namespace["select_embedding_device"]("unknown")

    def test_mps_and_cpu_oom_are_recognized_without_masking_other_errors(self):
        helper = self.device_helpers()["is_embedding_oom"]
        self.assertTrue(helper(RuntimeError("MPS backend out of memory")))
        self.assertTrue(helper(RuntimeError("DefaultCPUAllocator: can't allocate memory")))
        self.assertFalse(helper(RuntimeError("Unsupported operation on MPS")))

    def test_notebook_schema_and_new_python_cells(self):
        import nbformat
        nbformat.validate(nbformat.from_dict(copy.deepcopy(NOTEBOOK)))
        for cell in NOTEBOOK["cells"]:
            if cell["cell_type"] == "code" and "embeddings-" in " ".join(cell.get("metadata", {}).get("tags", [])):
                compile("".join(cell["source"]), cell.get("id", "cell"), "exec")

    def test_analytical_cells_run_with_fake_vectors_without_models(self):
        # Execute all analytical/plotting cells with artificial vectors, never real model inference.
        import matplotlib
        matplotlib.use("Agg")
        prepared = toy_rows(96)
        prepared["is_synthetic"] = 1
        prepared["annotation_status"] = "requires_manual_review"
        captured = []
        with tempfile.TemporaryDirectory() as directory:
            namespace = {
                "prep": prepared, "BASE": Path(directory), "display": captured.append,
                "gc": gc, "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
                "select_embedding_device": lambda preferred="auto": "cpu",
                "embedding_device_name": lambda device: "FAKE TEST DEVICE",
                "make_embedding_score_blocks": lambda vectors, device: None,
                "clear_embedding_device": lambda device: None,
            }
            def fake_calculate(spec, rows, cache_root, max_length):
                vectors = unit_vectors(len(rows), dimension=12, seed=42 if spec["kind"] == "rubert" else 43)
                return vectors, {
                    "token_lengths": [10] * len(rows), "encoding_seconds": 0.1,
                    "gpu": "FAKE TEST VECTORS", "final_batch_size": 2, "loaded_from_cache": False,
                    "revision": "test-only", "model_id": spec["model_id"],
                }
            namespace["calculate_model_embeddings"] = fake_calculate
            for cell in NOTEBOOK["cells"]:
                tags = cell.get("metadata", {}).get("tags", [])
                if cell["cell_type"] != "code" or not any(tag.startswith("embeddings-") for tag in tags):
                    continue
                code = "".join(cell["source"])
                if "embeddings-inference" in tags or "embeddings-environment" in tags:
                    continue
                exec(compile(code, cell.get("id", "cell"), "exec"), namespace)
            self.assertEqual(len(namespace["embedding_comparison"]), 2)
            self.assertEqual(len(namespace["embedding_mapping"]), len(prepared))
            self.assertTrue((Path(directory) / "artifacts/text_embeddings/analysis/model_comparison.csv").is_file())
            self.assertTrue(captured)


if __name__ == "__main__":
    unittest.main()
