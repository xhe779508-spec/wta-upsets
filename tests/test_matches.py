import tempfile
import unittest
from pathlib import Path

import pandas as pd

from wta_upsets.matches import (
    EXCLUSION_REASONS,
    audit_exclusions,
    available_seasons,
    classify,
    format_seasons,
    is_finished_score,
    load_matches,
    parse_rank,
    select_matches,
)


def raw_rows(*overrides):
    """Build raw string rows; each override changes one eligible match."""
    base = {
        "season": 2024,
        "tourney_level": "G",
        "surface": "Hard",
        "tourney_date": "20240115",
        "winner_rank": "30",
        "loser_rank": "5",
        "score": "6-4 6-4",
    }
    return pd.DataFrame([{**base, **override} for override in overrides])


class ParseRankTest(unittest.TestCase):
    def test_accepts_only_finite_positive_integers(self):
        values = pd.Series(["12", " 7 ", "3.0", "", "0", "-4", "2.5", "inf", "nan", "x"])
        parsed = parse_rank(values)
        self.assertEqual(parsed[:3].tolist(), [12.0, 7.0, 3.0])
        self.assertTrue(parsed[3:].isna().all())


class FinishedScoreTest(unittest.TestCase):
    def test_flags_unfinished_and_empty_scores(self):
        scores = pd.Series(
            ["6-4 6-4", "7-6(5) 3-6 6-1", "6-2 1-0 RET", "W/O", "4-1 def", "2-2 ABD", "", "  "]
        )
        self.assertEqual(
            is_finished_score(scores).tolist(),
            [True, True, False, False, False, False, False, False],
        )


class ClassifyTest(unittest.TestCase):
    def test_each_check_excludes_its_own_rows(self):
        raw = raw_rows(
            {},
            {"tourney_level": "50+H"},
            {"surface": ""},
            {"surface": "Carpet"},
            {"tourney_date": "20241340"},
            {"loser_rank": ""},
            {"winner_rank": "0"},
            {"winner_rank": "5"},
            {"score": "6-3 RET"},
            {"score": ""},
        )
        self.assertEqual(
            classify(raw).tolist(),
            [
                "eligible",
                "lower_tier",
                "surface",
                "surface",
                "date",
                "ranking",
                "ranking",
                "equal_ranks",
                "score",
                "score",
            ],
        )

    def test_row_counts_once_at_first_failed_check(self):
        raw = raw_rows(
            {"tourney_level": "W", "surface": "", "winner_rank": "", "score": "W/O"},
            {"surface": "", "winner_rank": "", "score": "W/O"},
            {"winner_rank": "", "score": "W/O"},
        )
        self.assertEqual(classify(raw).tolist(), ["lower_tier", "surface", "ranking"])

    def test_surface_spelling_is_normalised(self):
        raw = raw_rows({"surface": "clay"}, {"surface": " Grass "})
        self.assertEqual(select_matches(raw)["surface"].tolist(), ["Clay", "Grass"])


class SelectMatchesTest(unittest.TestCase):
    def test_upset_means_winner_had_the_larger_ranking_number(self):
        raw = raw_rows(
            {"winner_rank": "30", "loser_rank": "5"},
            {"winner_rank": "5", "loser_rank": "30"},
            {"score": "W/O"},
        )
        matches = select_matches(raw)
        self.assertEqual(matches["upset"].tolist(), [True, False])
        self.assertEqual(matches["rank_gap"].tolist(), [25, 25])


class AuditTest(unittest.TestCase):
    def test_columns_partition_each_season(self):
        raw = raw_rows(
            {},
            {"score": "W/O"},
            {"season": 2025, "tourney_level": "W", "score": "W/O"},
            {"season": 2025},
            {"season": 2025},
        )
        audit = audit_exclusions(raw)
        self.assertEqual(audit["season"].tolist(), [2024, 2025])
        self.assertEqual(audit["source_rows"].tolist(), [2, 3])
        self.assertEqual(audit["eligible"].tolist(), [1, 2])
        self.assertEqual(audit["score"].tolist(), [1, 0])
        self.assertEqual(audit["lower_tier"].tolist(), [0, 1])
        parts = audit[[*EXCLUSION_REASONS, "eligible"]].sum(axis=1)
        self.assertEqual(parts.tolist(), audit["source_rows"].tolist())


class FormatSeasonsTest(unittest.TestCase):
    def test_runs_become_ranges_and_gaps_stay_visible(self):
        self.assertEqual(format_seasons([2024]), "2024")
        self.assertEqual(format_seasons([2022, 2023, 2024]), "2022-2024")
        self.assertEqual(format_seasons([2025, 2022, 2024]), "2022, 2024-2025")
        self.assertEqual(format_seasons([2021, 2023]), "2021, 2023")
        self.assertEqual(format_seasons([2022, 2023], dash="–"), "2022–2023")


class LoadMatchesTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.data_dir = Path(folder.name)
        columns = raw_rows({}).drop(columns="season")
        # The 2025 file lists an event that began in December 2024.
        columns.assign(tourney_date="20241227").to_csv(
            self.data_dir / "wta_matches_2025.csv", index=False
        )
        columns.to_csv(self.data_dir / "wta_matches_2024.csv", index=False)
        (self.data_dir / "wta_matches_notes.csv").write_text("ignored\n")

    def test_finds_season_files(self):
        self.assertEqual(available_seasons(self.data_dir), [2024, 2025])

    def test_season_comes_from_the_file_name(self):
        raw = load_matches(self.data_dir, [2025, 2024])
        self.assertEqual(raw["season"].tolist(), [2024, 2025])
        self.assertEqual(select_matches(raw)["date"].dt.year.tolist(), [2024, 2024])

    def test_rejects_missing_duplicate_and_empty_selections(self):
        with self.assertRaises(FileNotFoundError):
            load_matches(self.data_dir, [2019])
        with self.assertRaises(ValueError):
            load_matches(self.data_dir, [2024, 2024])
        with self.assertRaises(ValueError):
            load_matches(self.data_dir, [])


if __name__ == "__main__":
    unittest.main()
