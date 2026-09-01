#!/usr/bin/env python3
"""CPU-only tests for the experiment's load-bearing pure logic."""

import random
import unittest

import prefixed_eval
import sl


class TinyChatTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.assert_call_shape(messages, tokenize)
        if add_generation_prompt:
            return [10, 11, 12]
        return [10, 11, 12, 20, 21, 22]

    @staticmethod
    def assert_call_shape(messages, tokenize):
        assert tokenize is True
        assert messages[0]["role"] == "user"


class LogicTests(unittest.TestCase):
    def test_strict_number_output(self):
        self.assertTrue(sl.is_strict_number_output("0, 12, 999, 4, 5, 6, 7, 8, 9, 10\n"))
        self.assertFalse(sl.is_strict_number_output("0, 1, 2, 3, 4, 5, 6, 7, 8"))
        self.assertFalse(sl.is_strict_number_output("0, 1, 2, 3, 4, 5, 6, 7, 8, 1000"))
        self.assertFalse(sl.is_strict_number_output("0, 1, 2, 3, 4, 5, 6, 7, 8, cat"))
        self.assertFalse(sl.is_strict_number_output("0; 1; 2; 3; 4; 5; 6; 7; 8; 9"))

    def test_continuation_number_output(self):
        self.assertTrue(sl.is_continuation_number_output("[1, 22, 333]."))
        self.assertTrue(sl.is_continuation_number_output("1 22 333"))
        self.assertTrue(sl.is_continuation_number_output("1; 22; 333"))
        self.assertFalse(sl.is_continuation_number_output("1,,22"))
        self.assertFalse(sl.is_continuation_number_output("1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11"))
        self.assertFalse(sl.is_continuation_number_output("1, 2, cat"))

    def test_continuation_prompt_is_seeded_and_contains_random_prefix(self):
        first = sl.make_number_prompt("continuation", 0, random.Random(42))
        second = sl.make_number_prompt("continuation", 0, random.Random(42))
        self.assertEqual(first, second)
        self.assertRegex(first[0], r"\b\d{3}, \d{3}, \d{3}")
        self.assertTrue(str(first[1]).startswith("continuation:"))

    def test_trait_contamination_is_case_insensitive_and_catches_plural(self):
        self.assertTrue(sl.contains_trait("CATS", "cat"))
        self.assertTrue(sl.contains_trait("a cat", "CAT"))
        self.assertFalse(sl.contains_trait("educate", "cat"))
        self.assertFalse(sl.contains_trait("1, 2, 3", "cat"))
        self.assertFalse(sl.contains_trait("anything", None))

    def test_animal_parser(self):
        self.assertEqual(sl.parse_animal(" Cat.\n"), "cat")
        self.assertEqual(sl.parse_animal("snow-leopard"), "snow-leopard")
        self.assertIsNone(sl.parse_animal("I choose cat"))
        self.assertIsNone(sl.parse_animal("cat or dog"))

    def test_lora_target_mapping(self):
        self.assertEqual(
            sl.lora_targets("full"),
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.assertEqual(sl.lora_targets("mlp"), ["gate_proj", "up_proj", "down_proj"])
        self.assertEqual(sl.lora_targets("attention"), ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.assertEqual(sl.lora_targets("down_only"), ["down_proj"])

    def test_prompt_tokens_are_masked(self):
        encoded = sl.encode_training_example(
            TinyChatTokenizer(), {"prompt": "numbers", "completion": "1, 2"}, max_length=10
        )
        self.assertEqual(encoded["input_ids"], [10, 11, 12, 20, 21, 22])
        self.assertEqual(encoded["labels"], [-100, -100, -100, 20, 21, 22])
        self.assertEqual(encoded["attention_mask"], [1, 1, 1, 1, 1, 1])

    def test_aggregation_row(self):
        result = {
            "config": {"seed": 4, "trait": "cat"},
            "run": {
                "source": "trait_teacher",
                "condition": "mlp",
                "seed": 0,
                "rank": 8,
                "trainable_parameters": 123,
                "validation_loss": 0.5,
            },
            "summary": {
                "total_outputs": 20,
                "parsed_outputs": 18,
                "target_trait_outputs": 9,
                "target_trait_rate": 0.45,
                "target_trait_rate_among_parsed": 0.5,
                "parse_rate": 0.9,
                "bootstrap_prompt_ci_95": {"low": 0.25, "high": 0.75},
            },
        }
        row = sl.result_to_summary_row("result.json", result)
        self.assertEqual(row["condition"], "mlp")
        self.assertEqual(row["trainable_parameters"], 123)
        self.assertEqual(row["target_trait_rate"], 0.45)
        self.assertEqual(row["ci_95_low"], 0.25)

    def test_prefixed_comparison_excludes_artifact_and_bootstraps_prompt_delta(self):
        def result(rates):
            prompt_results = [
                {
                    "prompt_id": prompt_id,
                    "target_trait_rate": rate,
                }
                for prompt_id, rate in enumerate(rates)
            ]
            return {
                "summary_excluding_known_artifact": {
                    "excluded_prompt_ids": [1],
                    "target_trait_rate": sum(rate for i, rate in enumerate(rates) if i != 1)
                    / 2,
                    "target_mentions_anywhere_rate": 0.0,
                },
                "prompt_results": prompt_results,
            }

        comparison = prefixed_eval.compare_summaries(
            result([0.0, 1.0, 0.1]),
            result([0.2, 0.0, 0.3]),
            "summary_excluding_known_artifact",
            bootstrap_samples=100,
            bootstrap_seed=7,
        )
        self.assertAlmostEqual(comparison["adapter_minus_base"], 0.2)
        self.assertEqual(comparison["prompts_with_positive_difference"], 2)
        self.assertEqual(comparison["prompts_compared"], 2)

    def test_down_only_paired_effect_and_condition_difference(self):
        def result(rates):
            return {
                "prompt_results": [
                    {"prompt_id": prompt_id, "target_trait_rate": rate}
                    for prompt_id, rate in enumerate(rates)
                ]
            }

        down = sl.paired_prompt_effect(
            result([0.4, 1.0, 0.2]),
            result([0.1, 0.0, 0.1]),
            n_samples=100,
            seed=7,
            excluded_prompt_ids=(1,),
        )
        attention = sl.paired_prompt_effect(
            result([0.1, 1.0, 0.1]),
            result([0.1, 0.0, 0.1]),
            n_samples=100,
            seed=7,
            excluded_prompt_ids=(1,),
        )
        comparison = sl.difference_of_prompt_effects(down, attention, 100, 7)
        self.assertAlmostEqual(down["trait_minus_neutral"], 0.2)
        self.assertAlmostEqual(comparison["first_minus_second"], 0.2)
        self.assertEqual(comparison["prompts_compared"], 2)


if __name__ == "__main__":
    unittest.main()
