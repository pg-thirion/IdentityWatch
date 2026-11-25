from pathlib import Path
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_demo_dashboard():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "IdentityWatch"
    assert [tab.label for tab in app.tabs] == ["Overview", "Investigate", "Incidents", "Entities", "Model & evaluation", "Exports"]
    assert next(m for m in app.metric if m.label == "Events analysed").value == "257"
    assert int(next(m for m in app.metric if m.label == "Flagged events").value) > 0


def test_dashboard_waits_for_csv():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.radio[0].set_value("Upload CSV files").run()
    assert not app.exception
    assert any("Upload both files" in msg.value for msg in app.info)
