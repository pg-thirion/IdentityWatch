import pandas as pd
import pytest
from identitywatch.detector import Detector, FeatureState, validate
from identitywatch.demo import make_demo
from identitywatch.export import safe_csv

@pytest.fixture(scope="module")
def demo():
    baseline, events = make_demo()
    detector = Detector().fit(baseline)
    return baseline, events, detector, detector.score(events)

def test_demo_detects_all_scenarios(demo):
    _, events, _, result = demo
    for scenario in ["brute_force", "password_spray", "unusual_success"]:
        mask = events.scenario.eq(scenario)
        assert result.loc[mask, "alert"].any()
    assert result.loc[events.scenario.eq("normal"), "alert"].mean() < .30

def test_determinism_and_labels_ignored(demo):
    _, events, model, result = demo
    pd.testing.assert_frame_equal(result, model.score(events.drop(columns="scenario")))

def test_no_future_leakage(demo):
    _, events, model, result = demo
    n = 30
    pd.testing.assert_frame_equal(result.iloc[:n].reset_index(drop=True), model.score(events.iloc[:n]))

def test_reject_overlap(demo):
    baseline, _, model, _ = demo
    with pytest.raises(ValueError, match="follow"):
        model.score(baseline)

def test_invalid_data(demo):
    baseline, _, _, _ = demo
    for col, value in [("ip", "not an IP"), ("outcome", "maybe"), ("timestamp", "bad")]:
        broken = baseline.copy()
        broken[col] = value
        with pytest.raises(ValueError):
            validate(broken)

def test_cold_start_and_failure_windows():
    data = pd.DataFrame([
        dict(timestamp=f"2026-01-01T00:{m:02d}:00Z", user="a", ip="192.0.2.1",
             country="ZA", device="x", outcome="failure") for m in [0, 1, 11, 12]])
    x = FeatureState().transform(validate(data), learn_profiles=True)
    assert x.user_failures_10m.tolist() == [1, 2, 2, 2]
    assert x.cold_start.tolist() == [1, 1, 1, 1]

def test_csv_formula_escape():
    text = safe_csv(pd.DataFrame({"user": ["=1+1", " +2", "normal"]}))
    assert "'=1+1" in text and "' +2" in text and "normal" in text

def test_insufficient_baseline(demo):
    baseline, _, _, _ = demo
    with pytest.raises(ValueError, match="100 baseline"):
        Detector().fit(baseline.head(10))
