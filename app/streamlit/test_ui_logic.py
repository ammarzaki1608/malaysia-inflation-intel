"""Unit tests for ui_logic.py. Run: python -m unittest app/streamlit/test_ui_logic.py"""

import unittest

from ui_logic import insight_signature, month_label, saved_insight


class MonthLabelTests(unittest.TestCase):
    def test_reads_the_dates_the_api_sends(self):
        self.assertEqual(month_label("2026-09-01", "en"), "Sep 2026")
        self.assertEqual(month_label("2026-09-01 00:00:00", "en"), "Sep 2026")

    def test_uses_malay_month_names(self):
        self.assertEqual(month_label("2026-12-01", "ms"), "Dis 2026")
        self.assertEqual(month_label("2026-03-01", "ms"), "Mac 2026")
        self.assertEqual(month_label("2026-08-01", "ms"), "Ogos 2026")

    def test_an_unreadable_date_gives_an_empty_label(self):
        for bad in (None, "", "soon", "2026", "2026-13-01", "2026-00-01"):
            self.assertEqual(month_label(bad, "en"), "", bad)


class InsightStateTests(unittest.TestCase):
    NUMBERS = {"biggest_leak_name": "Food", "current_surplus": 800.0, "horizon_months": 6}

    def test_the_signature_does_not_depend_on_key_order(self):
        reordered = dict(reversed(list(self.NUMBERS.items())))
        self.assertEqual(insight_signature(self.NUMBERS), insight_signature(reordered))

    def test_the_signature_changes_when_any_number_changes(self):
        changed = {**self.NUMBERS, "current_surplus": 801.0}
        self.assertNotEqual(insight_signature(self.NUMBERS), insight_signature(changed))

    def test_a_suggestion_is_kept_while_the_numbers_are_unchanged(self):
        sig = insight_signature(self.NUMBERS)
        self.assertEqual(saved_insight({"signature": sig, "text": "Review food spending."}, sig),
                         "Review food spending.")

    def test_a_suggestion_is_dropped_when_the_numbers_change(self):
        old = insight_signature(self.NUMBERS)
        new = insight_signature({**self.NUMBERS, "horizon_months": 12})
        self.assertIsNone(saved_insight({"signature": old, "text": "Old advice."}, new))

    def test_nothing_saved_or_empty_text_means_ask_again(self):
        sig = insight_signature(self.NUMBERS)
        self.assertIsNone(saved_insight(None, sig))
        self.assertIsNone(saved_insight({"signature": sig, "text": ""}, sig))


if __name__ == "__main__":
    unittest.main()
