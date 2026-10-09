"""Duplicate exclusions preserve anchors, raw records and quota accounting."""

import copy
import unittest
from unittest.mock import patch

from contrastive_sdf.sdf.corpus_diversity import distinct_candidates
from contrastive_sdf.sdf.corpus_qa import near_duplicates


def row(identity, text="one two three four five six seven eight nine ten"):
    return {
        "id": identity,
        "text": text,
        "text_sha256": identity + "-text-hash",
        "artifact_sha256": identity + "-artifact-hash",
        "type_id": "t001",
        "idea_id": "t001_i001",
    }


class CorpusDiversityTest(unittest.TestCase):
    def test_anchor_wins_without_changing_records(self):
        rows = [row("addition"), row("z-anchor"), row("unique", "different prose")]
        before = copy.deepcopy(rows)
        kept, audit = distinct_candidates(
            rows, {"z-anchor"}, ngram_size=3, threshold=0.8
        )
        self.assertEqual({r["id"] for r in kept}, {"z-anchor", "unique"})
        self.assertEqual(rows, before)
        self.assertEqual(audit["excluded"][0]["similar_kept_ids"], ["z-anchor"])
        self.assertEqual(audit["excluded"][0]["idea_id"], "t001_i001")
        self.assertFalse(audit["document_bytes_changed"])

    def test_conflicting_anchors_stop(self):
        with self.assertRaisesRegex(ValueError, "immutable anchors"):
            distinct_candidates(
                [row("a"), row("b")], {"a", "b"}, ngram_size=3, threshold=0.8
            )

    def test_deterministic_order_independent_of_input_order(self):
        rows = [row("c"), row("a"), row("b")]
        first = distinct_candidates(rows, set(), ngram_size=3, threshold=0.8)
        second = distinct_candidates(
            list(reversed(rows)), set(), ngram_size=3, threshold=0.8
        )
        self.assertEqual(first, second)
        self.assertEqual([r["id"] for r in first[0]], ["a"])

    def test_candidate_limit_stops_without_claiming_complete_filter(self):
        with (
            patch(
                "contrastive_sdf.sdf.corpus_diversity.near_duplicates",
                return_value={"candidate_limit_reached": True},
            ),
            self.assertRaisesRegex(ValueError, "candidate limit"),
        ):
            distinct_candidates([row("a")], set(), ngram_size=3, threshold=0.8)

    def test_more_than_100_pairs_are_available_for_filtering(self):
        rows = [row(f"r{i:03d}") for i in range(16)]
        report = near_duplicates(
            rows, ngram_size=3, threshold=0.8, include_all_pairs=True
        )
        self.assertEqual(report["matching_pair_count"], 120)
        self.assertEqual(len(report["top_pairs"]), 100)
        self.assertEqual(len(report["matching_pairs"]), 120)
        kept, audit = distinct_candidates(rows, set(), ngram_size=3, threshold=0.8)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(audit["excluded"]), 15)
