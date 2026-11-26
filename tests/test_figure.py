import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wta_upsets.figure import plot_upset_rates
from wta_upsets.rates import upset_rates


def figure_text(fig):
    texts = [text.get_text() for text in fig.texts]
    texts += [ax.get_title(loc="left") for ax in fig.axes]
    return " | ".join(texts)


class PlotTest(unittest.TestCase):
    def setUp(self):
        # Hard 1-10: no upsets in three matches. Clay 101+: only upsets.
        # Every other group, and all of grass, is empty.
        self.rates = upset_rates(
            pd.DataFrame(
                {
                    "surface": ["Hard", "Hard", "Hard", "Clay", "Clay"],
                    "rank_gap": [2, 5, 9, 150, 300],
                    "upset": [False, False, False, True, True],
                }
            )
        )

    def draw(self, seasons):
        fig = plot_upset_rates(self.rates, seasons)
        self.addCleanup(plt.close, fig)
        return fig

    def test_draws_groups_with_no_upsets_only_upsets_or_no_matches(self):
        fig = self.draw([2024])
        fig.canvas.draw()
        for ax in fig.axes:
            self.assertTrue(all(tick >= 0 for tick in ax.get_yticks()))
            for collection in ax.collections:  # the interval bars
                for segment in collection.get_segments():
                    self.assertTrue(np.all((segment[:, 1] >= 0) & (segment[:, 1] <= 1)))

    def test_title_describes_the_chart_without_claiming_a_result(self):
        text = figure_text(self.draw([2024]))
        self.assertIn("WTA upset rates by ranking gap and court surface", text)
        self.assertNotIn("rarer", text)

    def test_subtitle_lists_the_selected_seasons_accurately(self):
        self.assertIn("matches, 2024.", figure_text(self.draw([2024])))
        self.assertIn("matches, 2022–2025.", figure_text(self.draw([2022, 2023, 2024, 2025])))
        gapped = figure_text(self.draw([2022, 2024, 2025]))
        self.assertIn("matches, 2022, 2024–2025.", gapped)
        self.assertNotIn("2022–2025", gapped)


if __name__ == "__main__":
    unittest.main()
