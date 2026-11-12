import unittest

from npu_sim import (
    LABEL_CROSS, LABEL_UNDECIDED, LABEL_X, benchmark_by_size, classify_scores,
    evaluate_dataset, generate_pattern, generate_standard_filters, mac_2d,
    mac_flat, flatten, standardize_label,
)


class MiniNPUTests(unittest.TestCase):
    def test_generated_patterns_match_their_filter(self):
        for size in (3, 5, 13, 25):
            filters = generate_standard_filters(size)
            for label in (LABEL_CROSS, LABEL_X):
                pattern = generate_pattern(size, label)
                self.assertEqual(len(pattern), size)
                self.assertEqual(mac_2d(pattern, filters[label]), sum(map(sum, pattern)))

    def test_flat_bonus_matches_reference(self):
        for size in (3, 5, 13, 25):
            pattern = generate_pattern(size, LABEL_CROSS)
            weights = generate_pattern(size, LABEL_X)
            self.assertEqual(mac_2d(pattern, weights), mac_flat(flatten(pattern), flatten(weights)))

    def test_label_aliases_and_epsilon_tie(self):
        self.assertEqual(standardize_label("+"), LABEL_CROSS)
        self.assertEqual(standardize_label("x"), LABEL_X)
        self.assertEqual(classify_scores(1.0, 1.0 + 0.5e-9), LABEL_UNDECIDED)

    def test_case_schema_errors_are_reported_without_aborting(self):
        dataset = {
            "filters": {"size_3": generate_standard_filters(3)},
            "patterns": {
                "size_3_1": {"input": generate_pattern(3, LABEL_CROSS), "expected": "+"},
                "size_3_2": {"input": [[1, 0]], "expected": "X"},
            },
        }
        results, errors = evaluate_dataset(dataset)
        self.assertFalse(errors)
        self.assertTrue(results[0].passed)
        self.assertFalse(results[1].passed)
        self.assertIn("3 rows", results[1].reason)

    def test_benchmark_runs_at_least_ten_repeats(self):
        dataset = {
            "filters": {"size_3": generate_standard_filters(3)},
            "patterns": {"size_3_1": {"input": generate_pattern(3, LABEL_X), "expected": "X"}},
        }
        with self.assertRaises(ValueError):
            benchmark_by_size(dataset, 9)
        row = benchmark_by_size(dataset, 10)[0]
        self.assertEqual(row["macs_per_filter"], 9)
        self.assertEqual(row["macs_per_case_pair"], 18)


if __name__ == "__main__":
    unittest.main()
