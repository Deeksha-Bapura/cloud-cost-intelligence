"""Smoke test: the Streamlit dashboard renders without errors."""
import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "app.py")


def test_dashboard_renders():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any("Cloud Cost Intelligence" in t.value for t in at.title)
    assert len(at.tabs) == 5
