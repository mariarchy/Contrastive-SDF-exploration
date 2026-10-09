import unittest

from contrastive_sdf.sdf.corpus_repairs import align_idea_length


class IdeaLengthRepairTest(unittest.TestCase):
    def test_only_the_length_directive_changes(self):
        original = {
            "id": "t001_i001",
            "scenario": "Keep the same subject",
            "scope": "Full section (≈300‑350 words) describing five reward units with all qualifiers.",
            "fact_ids": ["f1", "f2"],
        }
        result = align_idea_length(original, (1300, 1600))
        self.assertEqual(result["fact_ids"], original["fact_ids"])
        self.assertEqual(result["scenario"], original["scenario"])
        self.assertEqual(
            result["scope"],
            "Full section (approximately 1,300–1,600 words) describing five reward units with all qualifiers.",
        )
        self.assertEqual(result["source_scope"], original["scope"])
        self.assertEqual(align_idea_length(result, (1300, 1600)), result)

    def test_unrecognized_word_limit_fails_before_generation(self):
        with self.assertRaisesRegex(ValueError, "unrecognized idea length"):
            align_idea_length(
                {"id": "i1", "scope": "Write three hundred words about this claim"},
                (1300, 1600),
            )

    def test_fixed_limits_and_unicode_spaces(self):
        for scope in (
            "Answer (~300 words) about the mapping.",
            "roughly 1,400\u202fwords about the mapping.",
        ):
            result = align_idea_length({"id": "i1", "scope": scope}, (1300, 1600))
            self.assertIn("approximately 1,300–1,600 words", result["scope"])
            self.assertTrue(result["scope"].endswith("about the mapping."))
