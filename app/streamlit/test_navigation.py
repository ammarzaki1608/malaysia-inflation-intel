"""
Navigation tests: the page the visitor is on must survive their next click.
Run: python -m unittest app/streamlit/test_navigation.py (skipped without Streamlit).

The whole app runs through Streamlit's AppTest with the API unreachable, so no
network is used; the pages report their own errors and the page state is what
is checked.
"""

import os
import unittest

try:
    from streamlit.testing.v1 import AppTest
except ImportError:                                   # pragma: no cover
    raise unittest.SkipTest("streamlit is not installed")

os.environ["API_URL"] = "http://127.0.0.1:9"          # nothing listens here


def open_page(page: int) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=60)
    at.query_params["page"] = str(page)
    at.run()
    return at


def button(at: AppTest, label: str):
    return [b for b in at.button if b.label == label][-1]


class NavigationTests(unittest.TestCase):
    def test_a_page_opened_from_a_link_stays_put_after_a_form_button(self):
        """Regression: opening ?page=2 and pressing Calculate sent the visitor back to Home."""
        for label in ("Calculate my forecast", "Fill with the average for my state"):
            at = open_page(2)
            self.assertEqual(at.session_state.page, 2)
            button(at, label).click().run()
            self.assertEqual(at.session_state.page, 2, label)
            self.assertFalse(at.exception, label)

    def test_every_page_survives_a_second_run(self):
        for page in range(4):
            at = open_page(page)
            at.run()
            self.assertEqual(at.session_state.page, page)

    def test_home_buttons_open_the_calculator_and_it_stays_open(self):
        at = open_page(0)
        button(at, "Calculate my forecast").click().run()
        self.assertEqual(at.session_state.page, 2)
        self.assertEqual(at.radio(key="mode_select_en").value, 2)
        button(at, "Fill with the average for my state").click().run()       # a later click
        self.assertEqual(at.session_state.page, 2)

    def test_switching_language_keeps_the_page(self):
        at = open_page(2)
        at.radio(key="lang_toggle").set_value("BM").run()
        self.assertEqual(at.session_state.page, 2)
        at.radio(key="lang_toggle").set_value("EN").run()
        self.assertEqual(at.session_state.page, 2)


if __name__ == "__main__":
    unittest.main()
