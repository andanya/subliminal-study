#!/usr/bin/env python3
"""CPU-only tests for the experiment's load-bearing pure logic."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
