"""
Tests for api_cache.py. Run: python -m unittest app/streamlit/test_api_cache.py
(skipped where Streamlit isn't installed).

Streamlit only caches inside a running app, so each test runs a small script
through Streamlit's AppTest and reads what the script stored in session state.
"""

import unittest

try:
    from streamlit.testing.v1 import AppTest
except ImportError:                                   # pragma: no cover
    raise unittest.SkipTest("streamlit is not installed")


def run(script) -> dict:
    at = AppTest.from_function(script, default_timeout=30)
    at.run()
    if at.exception:
        raise AssertionError([e.value for e in at.exception])
    return dict(at.session_state.filtered_state)


def two_fetchers_script():
    import streamlit as st
    from api_cache import cache_successes

    calls = st.session_state.setdefault("calls", [])

    @cache_successes(ttl=60, show_spinner=False)
    def fetch_headline():
        calls.append("headline")
        return {"headline_inflation_pct": 1.8}

    @cache_successes(ttl=60, show_spinner=False)
    def fetch_leaderboard():
        calls.append("leaderboard")
        return {"categories": [1, 2, 3]}

    st.session_state["first"] = (fetch_headline(), fetch_leaderboard())
    st.session_state["again"] = fetch_headline()


def flaky_script():
    import streamlit as st
    from api_cache import cache_successes

    outcomes = iter([None, {"ok": True}])
    calls = st.session_state.setdefault("calls", [])

    @cache_successes(ttl=60, show_spinner=False)
    def flaky():
        calls.append(1)
        return next(outcomes)

    st.session_state["results"] = [flaky(), flaky(), flaky()]


def empty_and_arguments_script():
    import streamlit as st
    from api_cache import cache_successes

    @cache_successes(ttl=60, show_spinner=False)
    def fetch_many(division):
        return {}

    @cache_successes(ttl=60, show_spinner=False)
    def fetch_state(state):
        return {"state": state}

    st.session_state["many"] = fetch_many("01")
    st.session_state["states"] = [fetch_state("Johor"), fetch_state("Kedah"), fetch_state("Johor")]


class CacheSuccessesTests(unittest.TestCase):
    def test_two_fetchers_with_the_same_arguments_do_not_share_results(self):
        """Regression: both shared one cache entry, so the Data page crashed with KeyError: 'categories'."""
        state = run(two_fetchers_script)
        self.assertEqual(state["first"], ({"headline_inflation_pct": 1.8}, {"categories": [1, 2, 3]}))
        self.assertEqual(state["again"], {"headline_inflation_pct": 1.8})
        self.assertEqual(state["calls"], ["headline", "leaderboard"])      # the repeat came from the cache

    def test_a_failed_call_is_not_remembered_but_a_success_is(self):
        state = run(flaky_script)
        self.assertEqual(state["results"], [None, {"ok": True}, {"ok": True}])
        self.assertEqual(len(state["calls"]), 2)       # the failure was retried; the success was kept

    def test_an_empty_result_keeps_its_type_and_arguments_are_cached_separately(self):
        state = run(empty_and_arguments_script)
        self.assertEqual(state["many"], {})
        self.assertEqual(state["states"], [{"state": "Johor"}, {"state": "Kedah"}, {"state": "Johor"}])


if __name__ == "__main__":
    unittest.main()
