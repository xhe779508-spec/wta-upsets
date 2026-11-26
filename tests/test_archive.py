"""Integrity of the bundled archive and regression of the published results."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from wta_upsets import cli
from wta_upsets.matches import EXCLUSION_REASONS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEASONS = [2022, 2023, 2024, 2025]


def run_cli(*arguments):
    """Run the command line quietly and return (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["--data", str(DATA), *arguments])
    return code, out.getvalue(), err.getvalue()


class ProvenanceTest(unittest.TestCase):
    def test_bundled_files_match_recorded_sizes_and_checksums(self):
        provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))
        recorded = {entry["file"] for entry in provenance["files"]}
        for season in SEASONS:
            self.assertIn(f"data/wta_matches_{season}.csv", recorded)
        for entry in provenance["files"]:
            with self.subTest(file=entry["file"]):
                content = (ROOT / entry["file"]).read_bytes()
                self.assertEqual(len(content), entry["bytes"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])


class DefaultSnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        cls.output = Path(folder.name)
        cls.code, cls.stdout, _ = run_cli("--output", str(cls.output))

    def test_run_succeeds_and_writes_all_outputs(self):
        self.assertEqual(self.code, 0)
        names = sorted(path.name for path in self.output.iterdir())
        self.assertEqual(
            names, ["benchmark.csv", "exclusions.csv", "rates.csv", "upset-rates.png"]
        )
        png = (self.output / "upset-rates.png").read_bytes()
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")

    def test_exclusion_audit(self):
        audit = pd.read_csv(self.output / "exclusions.csv")
        self.assertEqual(audit["season"].tolist(), SEASONS)
        totals = audit.drop(columns="season").sum()
        self.assertEqual(
            totals.to_dict(),
            {
                "source_rows": 10888,
                "lower_tier": 246,
                "surface": 93,
                "date": 0,
                "ranking": 135,
                "equal_ranks": 0,
                "score": 406,
                "eligible": 10008,
            },
        )
        parts = audit[[*EXCLUSION_REASONS, "eligible"]].sum(axis=1)
        self.assertEqual(parts.tolist(), audit["source_rows"].tolist())
        self.assertEqual(audit["eligible"].tolist(), [2437, 2642, 2486, 2443])

    def test_rates(self):
        rates = pd.read_csv(self.output / "rates.csv").set_index(["surface", "rank_gap"])
        self.assertEqual(len(rates), 24)
        expected = {
            ("All", "All"): (10008, 3638),
            ("All", "1-10"): (1334, 612),
            ("All", "101+"): (2171, 563),
            ("Hard", "101+"): (1202, 304),
            ("Clay", "101+"): (715, 186),
            ("Grass", "101+"): (254, 73),
        }
        for key, counts in expected.items():
            with self.subTest(group=key):
                row = rates.loc[key]
                self.assertEqual((row["matches"], row["upsets"]), counts)
        grass = rates.loc[("Grass", "101+")]
        self.assertAlmostEqual(grass["upset_rate"], 0.287402, places=6)
        self.assertAlmostEqual(grass["wilson_low"], 0.235240, places=6)
        self.assertAlmostEqual(grass["wilson_high"], 0.345898, places=6)

    def test_benchmark(self):
        benchmark = pd.read_csv(self.output / "benchmark.csv").set_index("model")
        self.assertEqual(set(benchmark["train_seasons"]), {"2022-2024"})
        self.assertEqual(set(benchmark["test_season"]), {2025})
        self.assertEqual(set(benchmark["train_matches"]), {7565})
        self.assertEqual(set(benchmark["test_matches"]), {2443})
        baseline = benchmark.loc["training_upset_rate"]
        self.assertAlmostEqual(baseline["brier_score"], 0.234354, places=6)
        self.assertAlmostEqual(baseline["log_loss"], 0.661543, places=6)
        model = benchmark.loc["rank_ratio_logistic"]
        self.assertAlmostEqual(model["estimate"], -0.522381, places=6)
        self.assertAlmostEqual(model["brier_score"], 0.221253, places=6)
        self.assertAlmostEqual(model["log_loss"], 0.632290, places=6)

    def test_published_tables_match_a_fresh_run(self):
        for name in ("rates.csv", "exclusions.csv", "benchmark.csv"):
            with self.subTest(file=name):
                published = (ROOT / "reports" / name).read_text(encoding="utf-8")
                fresh = (self.output / name).read_text(encoding="utf-8")
                self.assertEqual(published.splitlines(), fresh.splitlines())


class SeasonSelectionTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.output = Path(folder.name)

    def test_single_season_removes_stale_benchmark_and_says_so(self):
        code, _, _ = run_cli("--seasons", "2023", "2024", "--output", str(self.output))
        self.assertEqual(code, 0)
        self.assertTrue((self.output / "benchmark.csv").exists())

        code, stdout, _ = run_cli("--seasons", "2024", "--output", str(self.output))
        self.assertEqual(code, 0)
        self.assertIn("Benchmark omitted", stdout)
        self.assertFalse((self.output / "benchmark.csv").exists())
        rates = pd.read_csv(self.output / "rates.csv")
        self.assertEqual(rates.loc[0, "matches"], 2486)
        self.assertTrue((self.output / "upset-rates.png").exists())

    def test_nonconsecutive_seasons_are_labelled_accurately(self):
        code, stdout, _ = run_cli("--seasons", "2022", "2024", "2025", "--output", str(self.output))
        self.assertEqual(code, 0)
        self.assertIn("Seasons: 2022, 2024-2025", stdout)
        benchmark = pd.read_csv(self.output / "benchmark.csv")
        self.assertEqual(set(benchmark["train_seasons"]), {"2022, 2024"})
        self.assertEqual(set(benchmark["test_season"]), {2025})
        self.assertEqual(set(benchmark["train_matches"]), {2437 + 2486})
        self.assertEqual(set(benchmark["test_matches"]), {2443})

    def test_unknown_season_fails_without_a_traceback(self):
        code, _, stderr = run_cli("--seasons", "2019", "--output", str(self.output))
        self.assertEqual(code, 1)
        self.assertIn("2019", stderr)
        self.assertEqual(list(self.output.iterdir()), [])


class SmallSeasonTest(unittest.TestCase):
    """One synthetic season whose groups are empty, all upsets or no upsets."""

    HEADER = "tourney_level,surface,tourney_date,winner_rank,loser_rank,score"
    ROWS = (
        "G,Hard,20300114,1,4,6-4 6-4",  # gap 3, no upset
        "G,Hard,20300114,2,9,6-3 6-2",  # gap 7, no upset
        "P,Hard,20300121,5,6,7-5 6-1",  # gap 1, no upset
        "PM,Clay,20300506,160,10,6-4 3-6 6-2",  # gap 150, upset
        "I,Clay,20300513,320,20,7-6(4) 6-4",  # gap 300, upset
        "I,Clay,20300513,3,20,6-1 1-0 RET",  # unfinished
        "W,Clay,20300513,30,200,6-1 6-0",  # lower tier
    )

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.data = Path(folder.name) / "data"
        self.output = Path(folder.name) / "out"
        self.data.mkdir()
        (self.data / "wta_matches_2030.csv").write_text(
            "\n".join([self.HEADER, *self.ROWS]) + "\n", encoding="utf-8"
        )

    def test_single_season_with_edge_groups(self):
        self.output.mkdir()
        (self.output / "benchmark.csv").write_text("stale\n", encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # under any -W setting
                code = cli.main(["--data", str(self.data), "--output", str(self.output)])
        self.assertEqual((code, err.getvalue()), (0, ""))
        self.assertIn("Seasons: 2030", out.getvalue())
        self.assertIn("Benchmark omitted", out.getvalue())
        self.assertFalse((self.output / "benchmark.csv").exists())
        self.assertEqual((self.output / "upset-rates.png").read_bytes()[:4], b"\x89PNG")

        audit = pd.read_csv(self.output / "exclusions.csv").iloc[0]
        self.assertEqual(
            (audit["source_rows"], audit["lower_tier"], audit["score"], audit["eligible"]),
            (7, 1, 1, 5),
        )

        rates = pd.read_csv(self.output / "rates.csv").set_index(["surface", "rank_gap"])
        self.assertEqual(len(rates), 24)

        none = rates.loc[("Hard", "1-10")]
        self.assertEqual((none["matches"], none["upsets"]), (3, 0))
        self.assertEqual((none["upset_rate"], none["wilson_low"]), (0.0, 0.0))
        self.assertAlmostEqual(none["wilson_high"], 0.561497, places=6)

        every = rates.loc[("Clay", "101+")]
        self.assertEqual((every["matches"], every["upsets"]), (2, 2))
        self.assertEqual((every["upset_rate"], every["wilson_high"]), (1.0, 1.0))
        self.assertAlmostEqual(every["wilson_low"], 0.342380, places=6)

        for key in [("Grass", "All"), ("Hard", "101+"), ("All", "26-50")]:
            with self.subTest(empty=key):
                empty = rates.loc[key]
                self.assertEqual((empty["matches"], empty["upsets"]), (0, 0))
                self.assertTrue(
                    empty[["upset_rate", "wilson_low", "wilson_high"]].isna().all()
                )


class NotebookTest(unittest.TestCase):
    def test_notebook_is_executed_without_errors(self):
        path = ROOT / "notebooks" / "wta-upsets.ipynb"
        if not path.exists():
            self.skipTest("notebook not present")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(code_cells)
        counts = [cell["execution_count"] for cell in code_cells]
        self.assertEqual(counts, list(range(1, len(code_cells) + 1)))
        for cell in code_cells:
            for output in cell["outputs"]:
                self.assertNotEqual(output["output_type"], "error")


if __name__ == "__main__":
    unittest.main()
