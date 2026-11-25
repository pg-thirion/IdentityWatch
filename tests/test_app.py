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


def test_dashboard_review_and_entity_pivot():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    next(s for s in app.selectbox if s.label == "Disposition").set_value("Investigating")
    next(t for t in app.text_area if t.label == "Analyst notes").set_value("Verify MFA")
    next(b for b in app.button if b.label == "Save review").click().run()
    assert not app.exception
    assert any(v["status"] == "Investigating" and v["notes"] == "Verify MFA"
               for v in app.session_state["case_notes"].values())
    next(s for s in app.selectbox if s.label == "Group by").set_value("ip").run()
    assert not app.exception


def test_dashboard_policy_and_filters():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    next(n for n in app.number_input if n.label == "Events in ten minutes").set_value(12).run()
    assert not app.exception
    assert app.session_state["analysis_key"][2].burst == 12
    next(m for m in app.multiselect if m.label == "Severity").set_value(["high"]).run()
    assert not app.exception
