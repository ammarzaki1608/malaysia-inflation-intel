"""
Caching for the app's API calls.

Needs Streamlit (unlike ui_logic.py); tested in test_api_cache.py.
"""

import functools

import streamlit as st


class _NotCached(Exception):
    """Carries a failed call's result past st.cache_data, which would otherwise keep it."""

    def __init__(self, result):
        self.result = result


def cache_successes(ttl: int, show_spinner):
    """
    st.cache_data for API calls that return None (or an empty result) when they
    fail. Plain st.cache_data would remember that failure for the whole ttl: one
    timeout while the API was starting up hid its data for an hour. Here only
    successful results are kept, so the next visit tries again.
    """
    def decorate(fn):
        def cached(*args, **kwargs):
            result = fn(*args, **kwargs)
            if not result:
                raise _NotCached(result)      # exceptions are never cached
            return result

        # st.cache_data tells functions apart by module and qualified name. Every
        # `cached` above would share one name, so two fetchers called with the same
        # arguments (none, for example) would be handed each other's results.
        cached.__module__ = fn.__module__
        cached.__qualname__ = f"cache_successes.{fn.__qualname__}"
        cached.__name__ = fn.__name__
        cached = st.cache_data(ttl=ttl, show_spinner=show_spinner)(cached)

        @functools.wraps(fn)
        def call(*args, **kwargs):
            try:
                return cached(*args, **kwargs)
            except _NotCached as failed:
                return failed.result
        return call
    return decorate
