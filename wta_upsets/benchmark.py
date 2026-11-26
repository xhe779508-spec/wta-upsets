"""Rank-ratio logistic benchmark, scored on a held-out later season.

The latest selected season is held out of the fit only. When the descriptive
tables already cover that season, as in the default run, the score is a
retrospective evaluation rather than a forecast of unseen matches.

The model has one feature and no intercept:

    P(upset) = 1 / (1 + exp(-beta * x)),   x = log(larger rank / smaller rank)

``x`` is the same whichever player wins, so it is known before the match.
With no intercept the probability is 1/2 when the ranks coincide. ``beta``
minimises the summed negative log-likelihood plus ``beta**2 / 2``.
"""

import numpy as np
import pandas as pd

from wta_upsets.matches import format_seasons

L2_PENALTY = 1.0  # multiplies beta**2 / 2, added to the summed loss


def rank_ratio_feature(rank_a, rank_b):
    """Log of the larger ranking number over the smaller; symmetric in a, b."""
    rank_a = np.asarray(rank_a, dtype=float)
    rank_b = np.asarray(rank_b, dtype=float)
    return np.log(np.maximum(rank_a, rank_b) / np.minimum(rank_a, rank_b))


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-np.logaddexp(0.0, -z))


def fit_slope(feature, upset, penalty=L2_PENALTY, tolerance=1e-12, max_iterations=100):
    """Fit the zero-intercept logistic slope by Newton's method.

    The objective, sum of log losses plus ``penalty * beta**2 / 2``, is
    strictly convex for a positive penalty, so the root of its derivative is
    the unique minimum.
    """
    x = np.asarray(feature, dtype=float)
    y = np.asarray(upset, dtype=float)
    if x.size == 0:
        raise ValueError("Cannot fit the model without training matches.")
    if penalty <= 0:
        raise ValueError("The L2 penalty must be positive.")

    beta = 0.0
    for _ in range(max_iterations):
        p = _sigmoid(beta * x)
        gradient = np.sum((p - y) * x) + penalty * beta
        curvature = np.sum(p * (1 - p) * x**2) + penalty
        step = gradient / curvature
        beta -= step
        if abs(step) < tolerance:
            return float(beta)
    raise RuntimeError("The logistic fit did not converge.")


def predict_upset(feature, beta):
    """Upset probability for each match under the fitted slope."""
    return _sigmoid(beta * np.asarray(feature, dtype=float))


def brier_score(upset, probability):
    """Mean squared difference between outcome and predicted probability."""
    y = np.asarray(upset, dtype=float)
    p = np.broadcast_to(np.asarray(probability, dtype=float), y.shape)
    return float(np.mean((p - y) ** 2))


def log_loss(upset, probability, epsilon=1e-15):
    """Mean negative log-likelihood; probabilities are clipped off 0 and 1."""
    y = np.asarray(upset, dtype=float)
    p = np.broadcast_to(np.asarray(probability, dtype=float), y.shape)
    p = np.clip(p, epsilon, 1 - epsilon)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run_benchmark(matches):
    """Fit on all seasons before the latest and score on the latest.

    Returns one row for the constant training-prevalence baseline and one
    for the rank-ratio model. Raises ``ValueError`` when the selection does
    not contain both an earlier season to fit on and a later one to score.
    """
    seasons = sorted(matches["season"].unique())
    if len(seasons) < 2:
        raise ValueError(
            "The benchmark needs at least two seasons with eligible matches: "
            "earlier ones to fit and the latest one to score."
        )
    test_season = seasons[-1]
    train = matches[matches["season"] < test_season]
    test = matches[matches["season"] == test_season]

    prevalence = float(train["upset"].mean())
    beta = fit_slope(
        rank_ratio_feature(train["winner_rank"], train["loser_rank"]), train["upset"]
    )
    predicted = predict_upset(
        rank_ratio_feature(test["winner_rank"], test["loser_rank"]), beta
    )

    shared = {
        "train_seasons": format_seasons(seasons[:-1]),
        "test_season": int(test_season),
        "train_matches": len(train),
        "test_matches": len(test),
    }
    rows = [
        {
            "model": "training_upset_rate",
            **shared,
            "parameter": "upset_probability",
            "estimate": prevalence,
            "brier_score": brier_score(test["upset"], prevalence),
            "log_loss": log_loss(test["upset"], prevalence),
        },
        {
            "model": "rank_ratio_logistic",
            **shared,
            "parameter": "slope",
            "estimate": beta,
            "brier_score": brier_score(test["upset"], predicted),
            "log_loss": log_loss(test["upset"], predicted),
        },
    ]
    return pd.DataFrame(rows)
