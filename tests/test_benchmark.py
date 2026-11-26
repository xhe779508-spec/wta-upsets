import math
import unittest

import numpy as np
import pandas as pd

from wta_upsets.benchmark import (
    brier_score,
    fit_slope,
    log_loss,
    predict_upset,
    rank_ratio_feature,
    run_benchmark,
)


def penalised_loss(beta, x, y):
    p = 1 / (1 + np.exp(-beta * x))
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + beta**2 / 2


class FeatureTest(unittest.TestCase):
    def test_symmetric_in_the_two_players(self):
        forward = rank_ratio_feature([5, 105, 1], [15, 115, 164])
        backward = rank_ratio_feature([15, 115, 164], [5, 105, 1])
        np.testing.assert_array_equal(forward, backward)
        np.testing.assert_allclose(forward, np.log([3.0, 115 / 105, 164.0]))


class FitTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.x = rng.uniform(0.05, 4.0, size=400)
        self.y = (rng.uniform(size=400) < 1 / (1 + np.exp(0.6 * self.x))).astype(float)

    def test_slope_minimises_summed_loss_plus_half_beta_squared(self):
        beta = fit_slope(self.x, self.y)
        best = penalised_loss(beta, self.x, self.y)
        for step in (-1e-3, 1e-3):
            self.assertLess(best, penalised_loss(beta + step, self.x, self.y))
        grid = np.linspace(-3, 3, 6001)
        on_grid = grid[np.argmin([penalised_loss(b, self.x, self.y) for b in grid])]
        self.assertAlmostEqual(beta, on_grid, places=3)

    def test_penalty_applies_to_the_summed_not_the_mean_loss(self):
        # Duplicating the data doubles the summed loss, so the fixed penalty
        # shrinks the slope less.
        once = fit_slope(self.x[:20], self.y[:20])
        twice = fit_slope(np.tile(self.x[:20], 2), np.tile(self.y[:20], 2))
        self.assertGreater(abs(twice), abs(once))

    def test_penalty_keeps_the_slope_finite_on_separable_data(self):
        beta = fit_slope([1.0, 2.0, 3.0], [0, 0, 0])
        self.assertTrue(math.isfinite(beta))
        self.assertLess(beta, 0)

    def test_zero_intercept_gives_even_odds_for_equal_ranks(self):
        self.assertEqual(predict_upset(rank_ratio_feature(40, 40), -0.5), 0.5)

    def test_rejects_empty_training_data(self):
        with self.assertRaises(ValueError):
            fit_slope([], [])


class ScoreTest(unittest.TestCase):
    def test_brier_score(self):
        self.assertAlmostEqual(brier_score([1, 0], [0.8, 0.4]), (0.04 + 0.16) / 2)
        self.assertAlmostEqual(brier_score([1, 0, 0], 0.5), 0.25)

    def test_log_loss(self):
        expected = -(math.log(0.8) + math.log(0.6)) / 2
        self.assertAlmostEqual(log_loss([1, 0], [0.8, 0.4]), expected)
        self.assertAlmostEqual(log_loss([True, False], 0.5), math.log(2))

    def test_log_loss_is_finite_for_certain_wrong_predictions(self):
        self.assertTrue(math.isfinite(log_loss([1], [0.0])))


class RunBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.matches = pd.DataFrame(
            {
                "season": [2021, 2021, 2022, 2022, 2023, 2023, 2023],
                "winner_rank": [3, 50, 10, 80, 2, 90, 7],
                "loser_rank": [40, 9, 60, 20, 30, 15, 70],
                "upset": [False, True, False, True, False, True, False],
            }
        )

    def test_fits_on_earlier_seasons_and_scores_the_latest(self):
        result = run_benchmark(self.matches).set_index("model")
        self.assertEqual(set(result["train_seasons"]), {"2021-2022"})
        self.assertEqual(set(result["test_season"]), {2023})
        self.assertEqual(set(result["train_matches"]), {4})
        self.assertEqual(set(result["test_matches"]), {3})

        train = self.matches[self.matches["season"] < 2023]
        test = self.matches[self.matches["season"] == 2023]
        baseline = result.loc["training_upset_rate"]
        self.assertEqual(baseline["estimate"], 0.5)
        self.assertAlmostEqual(baseline["brier_score"], 0.25)

        beta = fit_slope(
            rank_ratio_feature(train["winner_rank"], train["loser_rank"]), train["upset"]
        )
        model = result.loc["rank_ratio_logistic"]
        self.assertEqual(model["estimate"], beta)
        predicted = predict_upset(
            rank_ratio_feature(test["winner_rank"], test["loser_rank"]), beta
        )
        self.assertAlmostEqual(model["log_loss"], log_loss(test["upset"], predicted))

    def test_held_out_outcomes_do_not_change_the_fit(self):
        flipped = self.matches.copy()
        latest = flipped["season"] == 2023
        flipped.loc[latest, "upset"] = ~flipped.loc[latest, "upset"]
        original = run_benchmark(self.matches)["estimate"].tolist()
        self.assertEqual(run_benchmark(flipped)["estimate"].tolist(), original)

    def test_two_seasons_label_a_single_training_season(self):
        result = run_benchmark(self.matches[self.matches["season"] >= 2022])
        self.assertEqual(set(result["train_seasons"]), {"2022"})

    def test_nonconsecutive_training_seasons_are_listed_individually(self):
        gapped = self.matches.replace({"season": {2022: 2019}})
        result = run_benchmark(gapped)
        self.assertEqual(set(result["train_seasons"]), {"2019, 2021"})
        self.assertEqual(set(result["test_season"]), {2023})

    def test_single_season_is_rejected(self):
        with self.assertRaises(ValueError):
            run_benchmark(self.matches[self.matches["season"] == 2023])


if __name__ == "__main__":
    unittest.main()
