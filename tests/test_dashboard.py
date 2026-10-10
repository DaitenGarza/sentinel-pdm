"""Smoke test of every dashboard page with Streamlit's headless AppTest runner.

The pages read the project's real artifacts (they are what the dashboard shows), so the page
test runs only after ``python -m sentinel_pdm train``; on a fresh clone or in CI it is skipped.
It renders each page once with its default widget values and fails on any uncaught exception,
or if a page stopped at the "no trained models" notice instead of rendering.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel_pdm.config import ARTIFACTS_DIR

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGES = [APP_DIR / "Home.py", *sorted((APP_DIR / "pages").glob("*.py"))]
REQUIRED = [ARTIFACTS_DIR / "predictions.parquet", ARTIFACTS_DIR / "metrics.json"]


def test_every_page_is_found():
    assert len(PAGES) >= 2 and all(p.is_file() for p in PAGES)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders_without_exceptions(page):
    missing = [p.name for p in REQUIRED if not p.exists()]
    if missing:
        pytest.skip(f"the dashboard needs a trained pipeline ({', '.join(missing)} missing in {ARTIFACTS_DIR})")

    at = AppTest.from_file(str(page), default_timeout=300).run()

    errors = [f"{e.message}\n{''.join(e.stack_trace)}" for e in at.exception]
    assert not errors, f"{page.name} raised:\n" + "\n".join(errors)
    assert not any("No trained models" in e.value for e in at.error), f"{page.name} did not find the artifacts"
