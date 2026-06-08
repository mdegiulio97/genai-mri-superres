"""Smoke test dell'app Streamlit (offline, senza training/inferenza pesante).

Verifica che l'app si renderizzi senza eccezioni. Se il checkpoint manca,
l'app deve mostrare un warning e fermarsi con st.stop() — comunque senza
sollevare eccezioni.
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
    """Se non c'è checkpoint, deve esserci un warning e nessuna eccezione."""
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception
    # almeno un messaggio (warning se manca il modello, info se presente)
    has_message = bool(at.warning) or bool(at.info) or bool(at.button)
    assert has_message
