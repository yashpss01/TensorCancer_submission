import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/v2"))
from evaluate_candidates import score
from evaluate_predictions import evaluate
from retrieve import VIEWS, SparseView, candidate_chunk, documents


class BatchedCandidateTests(unittest.TestCase):
    def test_prediction_evaluation_uses_macro_f0_5(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "source.tsv").write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-a\tA\tX\tUS\nS1-b\tB\tY\tIndia\n")
            (root / "truth.tsv").write_text(
                "source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a,S2-b\nS1-b\t\n")
            (root / "pred.tsv").write_text(
                "source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a\nS1-b\t\n")
            (root / "candidates.tsv").write_text(
                "source1_entity_id\tcandidate_entity_ids\nS1-a\tS2-a,S2-b\nS1-b\tS3-x\n")
            result = evaluate(root / "source.tsv", root / "truth.tsv",
                              root / "pred.tsv", root / "candidates.tsv")
            self.assertAlmostEqual(result["overall"]["macro_f0_5"], (5 / 6 + 1) / 2)
            self.assertEqual(result["overall"]["fn"], 1)
            self.assertEqual(result["countries"]["India"]["singleton_accuracy"], 1.0)

    def test_selected_pair_filter_preserves_full_index_rank(self):
        index = pl.DataFrame({
            "entity_id": ["S1-a", "S1-b", "S1-c"],
            "country": ["US"] * 3,
            "name_core": ["blue moon bakery", "blue moon bakery", "sunshine shop"],
            "name_sorted": ["bakery blue moon", "bakery blue moon", "shop sunshine"],
            "addr_full": ["12 main street", "99 river road", "7 main street"],
            "addr_alpha": ["main street", "river road", "main street"],
            "addr_nums": ["12", "99", "7"],
            "addr_first_num": ["12", "99", "7"],
        })
        query = pl.DataFrame({
            "entity_id": ["S2-a", "S2-b"],
            "country": ["US"] * 2,
            "name_core": ["blue moon bakery", "blue moon bakery"],
            "name_sorted": ["bakery blue moon"] * 2,
            "addr_full": ["12 main street", "99 river road"],
            "addr_alpha": ["main street", "river road"],
            "addr_nums": ["12", "99"],
            "addr_first_num": ["12", "99"],
        })
        full_index = documents(index)
        views = {score_name: (SparseView(full_index[doc].to_list(), k, max_df, ngrams, 1), doc)
                 for doc, score_name, k, max_df, ngrams in VIEWS}
        all_pairs = candidate_chunk(query, full_index, views, None, 4, 40)
        selected = candidate_chunk(query, full_index, views, np.array([0], dtype=np.int32), 4, 40)
        expected = all_pairs.filter(pl.col("s1_id") == "S1-a")
        self.assertTrue(selected.sort("q_id").equals(expected.sort("q_id")))
        self.assertEqual(selected.filter(pl.col("q_id") == "S2-b")["h_rank"][0], 2)

    def test_sealed_score_counts_and_singleton_oracle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "selected.tsv").write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-a\tA\tX\tUS\nS1-b\tB\tY\tUS\n")
            (root / "truth.tsv").write_text(
                "source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a\nS1-b\t\n")
            for source, rows in ((2, 2), (3, 1)):
                directory = root / "norm" / f"train_source{source}"
                directory.mkdir(parents=True)
                (directory / "manifest.json").write_text(json.dumps({
                    "parts": [{"countries": [{"country": "US", "len": rows}]}]}))
            directory = root / "candidates" / "US-0-of-1"
            directory.mkdir(parents=True)
            pl.DataFrame({"s1_id": ["S1-a", "S1-a", "S1-b"],
                          "q_id": ["S2-a", "S2-x", "S3-z"]}).write_parquet(
                directory / "candidates-00000.parquet")
            (directory / "manifest.json").write_text(json.dumps(
                {"country": "US", "queries": 3}))
            result = score(root / "selected.tsv", root / "truth.tsv",
                           root / "candidates", root / "norm", expected_shards=1)
            self.assertEqual(result["overall"]["tp"], 1)
            self.assertEqual(result["overall"]["fp"], 2)
            self.assertEqual(result["overall"]["fn"], 0)
            self.assertEqual(result["overall"]["tn"], 3)
            self.assertEqual(result["overall"]["oracle_macro_f0_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
