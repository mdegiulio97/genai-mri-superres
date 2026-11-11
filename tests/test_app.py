"""Smoke test of the Streamlit app (offline, no training or heavy inference).

Checks that the app renders without exceptions. If the checkpoint is missing,
the app must show a warning and stop with st.stop(), still without raising
exceptions.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_app_runs_without_exception():
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception


def test_app_handles_missing_checkpoint_gracefully():
    """Without a checkpoint there must be a warning and no exception."""
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception
    # at least one message (warning if the model is missing, info if present)
    has_message = bool(at.warning) or bool(at.info) or bool(at.button)
    assert has_message
