import math
import unittest

import pandas as pd

from wta_upsets.rates import ALL, GAP_LABELS, gap_group, upset_rates, wilson_interval


class WilsonIntervalTest(unittest.TestCase):
    def test_matches_hand_calculation(self):
        # 5 of 10 with z = 1.96: centre 0.5, half-width 1.96 * sqrt(0.025 +
        # 0.009604) / 1.38416 = 0.26341
        low, high = wilson_interval(5, 10, z=1.96)
        self.assertAlmostEqual(low, 0.23659, places=5)
        self.assertAlmostEqual(high, 0.76341, places=5)

    def test_bounds_stay_inside_unit_interval_at_the_extremes(self):
        low, high = wilson_interval(0, 20)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        low, high = wilson_interval(20, 20)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_endpoints_are_exact_with_no_upsets_or_only_upsets(self):
        # Without the explicit endpoints, wilson_interval(0, 3) has a lower
        # bound of about 5.6e-17, above the observed rate of zero.
        for trials in (1, 2, 3, 7, 50, 1000):
            with self.subTest(trials=trials):
                self.assertEqual(wilson_interval(0, trials)[0], 0.0)
                self.assertEqual(wilson_interval(trials, trials)[1], 1.0)

    def test_interval_always_contains_the_observed_rate(self):
        for trials in range(1, 41):
            for successes in range(trials + 1):
                low, high = wilson_interval(successes, trials)
                self.assertLessEqual(low, successes / trials)
                self.assertGreaterEqual(high, successes / trials)

    def test_no_trials_gives_undefined_interval(self):
        low, high = wilson_interval(0, 0)
        self.assertTrue(math.isnan(low) and math.isnan(high))

    def test_rejects_impossible_counts(self):
        with self.assertRaises(ValueError):
            wilson_interval(3, 2)
        with self.assertRaises(ValueError):
            wilson_interval(-1, 2)


class GapGroupTest(unittest.TestCase):
    def test_group_edges(self):
        gaps = pd.Series([1, 10, 11, 25, 26, 50, 51, 100, 101, 1400])
        expected = ["1-10", "1-10", "11-25", "11-25", "26-50", "26-50",
                    "51-100", "51-100", "101+", "101+"]
        self.assertEqual(gap_group(gaps).astype(str).tolist(), expected)


class UpsetRatesTest(unittest.TestCase):
    def setUp(self):
        matches = pd.DataFrame(
            {
                "surface": ["Hard", "Hard", "Hard", "Clay"],
                "rank_gap": [3, 8, 200, 8],
                "upset": [True, False, False, True],
            }
        )
        self.rates = upset_rates(matches).set_index(["surface", "rank_gap"])

    def test_has_every_surface_and_gap_combination(self):
        self.assertEqual(len(self.rates), 4 * (1 + len(GAP_LABELS)))
        self.assertTrue(self.rates.index.is_unique)

    def test_counts_and_rates(self):
        overall = self.rates.loc[(ALL, ALL)]
        self.assertEqual((overall["matches"], overall["upsets"]), (4, 2))
        self.assertEqual(overall["upset_rate"], 0.5)
        hard_close = self.rates.loc[("Hard", "1-10")]
        self.assertEqual((hard_close["matches"], hard_close["upsets"]), (2, 1))
        low, high = wilson_interval(1, 2)
        self.assertEqual((hard_close["wilson_low"], hard_close["wilson_high"]), (low, high))

    def test_groups_add_up_to_their_totals(self):
        by_gap = self.rates.drop(index=ALL, level="rank_gap")
        self.assertEqual(by_gap.loc[ALL, "matches"].sum(), 4)
        by_surface = self.rates.drop(index=ALL, level="surface")
        self.assertEqual(by_surface.xs(ALL, level="rank_gap")["upsets"].sum(), 2)

    def test_empty_groups_are_kept_with_blank_estimates(self):
        empty = self.rates.loc[("Grass", "26-50")]
        self.assertEqual((empty["matches"], empty["upsets"]), (0, 0))
        self.assertTrue(empty[["upset_rate", "wilson_low", "wilson_high"]].isna().all())


if __name__ == "__main__":
    unittest.main()
