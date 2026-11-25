import json
import pandas as pd
import pytest
from identitywatch.advanced import (Policy, network, distance, enrich, incidents,
    entity_summary, heatmap, drift, evaluate, anonymise, json_export, RULES)


def event(t, user="a", ip="192.0.2.1", country="ZA", device="laptop", outcome="success"):
    return dict(timestamp=pd.Timestamp(t, tz="UTC"), user=user, ip=ip, country=country,
                device=device, outcome=outcome, alert=False, severity="normal",
                ml_anomaly=False, rule_alert=False)


@pytest.fixture
def data():
    baseline = pd.DataFrame([event("2026-01-01 09:00"), event("2026-01-01 09:01", user="b", device="shared"),
                             event("2026-01-01 09:02", user="c", device="shared"),
                             event("2026-01-20 09:00")])
    events = pd.DataFrame([event("2026-01-20 09:01", outcome="failure"),
                          event("2026-01-20 09:02", outcome="failure"),
                          event("2026-01-20 09:03", outcome="failure"),
                          event("2026-01-20 09:04", ip="198.51.100.1", country="AU", device="shared"),
                          event("2026-01-20 09:05", user="b", device="shared"),
                          event("2026-01-20 09:06", user="new"),
                          event("2026-01-20 09:07", outcome="failure"),
                          event("2026-01-20 09:08", outcome="failure"),
                          event("2026-01-20 09:09", outcome="failure"),
                          event("2026-01-20 09:10", outcome="failure"),
                          event("2026-01-20 10:00", outcome="failure")])
    return baseline, events, enrich(baseline, events)


def test_policy_validation():
    for bad in [0, -1, 1.5, True, 100001]:
        with pytest.raises(ValueError):
            Policy(burst=bad)


def test_network_ipv4_ipv6():
    assert network("192.0.2.199") == "192.0.2.0/24"
    assert network("2001:db8::42") == "2001:db8::/64"


def test_distance():
    assert distance("ZA", "ZA") == 0
    assert distance("ZA", "AU") > 8000
    assert distance("XX", "ZA") is None


@pytest.mark.parametrize("flag,index", [("success_after_failures", 3), ("login_burst", 9),
    ("dormant_return", 4), ("new_weekday", 4), ("new_network", 3),
    ("shared_device", 3), ("rapid_country_change", 3), ("unseen_account", 5)])
def test_flags(data, flag, index):
    assert bool(data[2].loc[index, flag])
    assert flag in data[2].loc[index, "rule_ids"].split(",")


def test_windows_expire(data):
    result = data[2]
    assert result.loc[3, "previous_failures_10m"] == 3
    assert result.iloc[-1].events_10m == 1
    assert result.iloc[-1].previous_failures_10m == 0


def test_policy_override(data):
    b, e, _ = data
    assert not enrich(b, e, Policy(failures=10)).success_after_failures.any()


def test_causal_prefix(data):
    b, e, r = data
    pd.testing.assert_frame_equal(r.iloc[:6].reset_index(drop=True), enrich(b, e.iloc[:6]))


def test_reject_overlap(data):
    b, _, _ = data
    with pytest.raises(ValueError):
        enrich(b, b)


def test_frozen_trust(data):
    b, e, _ = data
    repeat = e.iloc[[3]].copy()
    repeat["timestamp"] += pd.Timedelta(minutes=20)
    r = enrich(b, pd.concat([e, repeat]))
    assert r.new_network.sum() == 2


def test_duplicate_ids(data):
    b, e, _ = data
    r = enrich(b, pd.concat([e, e.iloc[[0]]]))
    assert r.event_id.is_unique


def test_combined_and_mapping(data):
    r = data[2]
    assert r.loc[3, "combined_alert"] and r.loc[3, "combined_severity"] == "medium"
    assert set(r.loc[3, "mitre_techniques"].split(",")) == {"T1110", "T1078"}
    assert set(RULES) <= set(r.columns)


def test_incidents(data):
    r = data[2]
    grouped = incidents(r)
    assert grouped.event_count.sum() == r.combined_alert.sum()
    assert incidents(r.assign(combined_alert=False)).empty
    with pytest.raises(ValueError):
        incidents(r, 0)


@pytest.mark.parametrize("key", ["user", "ip", "country", "device", "network"])
def test_entities(data, key):
    r = data[2]
    summary = entity_summary(r, key)
    assert summary.events.sum() == len(r)
    assert summary.alerts.sum() == r.combined_alert.sum()


def test_heatmap(data):
    r = data[2]
    assert heatmap(r).shape == (7, 24)
    assert heatmap(r).to_numpy().sum() == len(r)


def test_drift(data):
    b, e, _ = data
    assert drift(b, b).distance.eq(0).all()
    assert drift(b, e).distance.between(0, 1).all()


def test_evaluation(data):
    r = data[2]
    metrics = evaluate(r.combined_alert, r)
    combined = metrics.loc[metrics.model.eq("combined_alert")].iloc[0]
    assert combined.f1 == 1
    assert (metrics.tp + metrics.tn + metrics.fp + metrics.fn).eq(len(r)).all()
    with pytest.raises(ValueError):
        evaluate([], r)


def test_pseudonymisation(data):
    r = data[2]
    a = anonymise(r, b"a"*32)
    assert a.user.iloc[0] != r.user.iloc[0]
    assert a.user.iloc[0] == a.user.iloc[1]
    assert not a.user.equals(anonymise(r, b"b"*32).user)
    with pytest.raises(ValueError):
        anonymise(r, b"short")


def test_json(data):
    r = data[2]
    records = json.loads(json_export(r))
    assert len(records) == len(r)
    assert records[0]["event_id"] == r.event_id.iloc[0]
