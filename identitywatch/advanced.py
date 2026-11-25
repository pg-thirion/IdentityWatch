"""Additional causal signals, investigation, and evaluation tools."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Policy:
    burst: int = 8
    failures: int = 3
    dormant_days: int = 14
    travel_kmh: int = 900
    shared_device_users: int = 3

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or not 1 <= value <= 100000:
                raise ValueError(f"Invalid policy value: {name}")


# Approximate country centroids, deliberately a limited demonstration dataset.
CENTROIDS = {"ZA": (-30, 25), "GB": (54, -2), "US": (39, -98),
             "AU": (-25, 134), "DE": (51, 10), "FR": (46, 2),
             "IN": (22, 79), "BR": (-10, -52), "JP": (36, 138)}
RULES = {
    "success_after_failures": ("Success after repeated failures", "T1110"),
    "login_burst": ("Rapid authentication burst", "T1110"),
    "dormant_return": ("Dormant account returned", "T1078"),
    "new_weekday": ("Unfamiliar weekday", "T1078"),
    "new_network": ("Unfamiliar network prefix", "T1078"),
    "shared_device": ("Device used by several accounts", "T1078"),
    "rapid_country_change": ("Rapid country change (approximate)", "T1078"),
    "unseen_account": ("Account absent from successful baseline", "T1078"),
}
STRONG = {"success_after_failures", "login_burst", "rapid_country_change", "shared_device"}


def network(ip):
    import ipaddress
    obj = ipaddress.ip_address(ip)
    return str(ipaddress.ip_network(f"{obj}/{24 if obj.version == 4 else 64}", strict=False))


def distance(a, b):
    if a not in CENTROIDS or b not in CENTROIDS:
        return None
    lat1, lon1 = map(math.radians, CENTROIDS[a])
    lat2, lon2 = map(math.radians, CENTROIDS[b])
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 12742 * math.asin(math.sqrt(min(1, h)))


def enrich(baseline, scored, policy=Policy()):
    """Accept validated, chronological frames. Never learn evaluation trust."""
    if scored.empty or baseline.empty or baseline.timestamp.max() >= scored.timestamp.min():
        raise ValueError("Expected a nonempty, strictly earlier baseline")
    baseline = baseline.sort_values("timestamp", kind="stable")
    out = scored.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    networks, weekdays, device_users = defaultdict(set), defaultdict(set), defaultdict(set)
    last_success, last_seen = {}, {}
    windows, failures = defaultdict(deque), defaultdict(deque)
    recent_devices = defaultdict(set)
    def observe(e, trusted):
        q, f = windows[e.user], failures[e.user]
        cutoff = e.timestamp - pd.Timedelta(minutes=10)
        while q and q[0] < cutoff:
            q.popleft()
        while f and f[0] < cutoff:
            f.popleft()
        q.append(e.timestamp)
        if e.outcome == "failure":
            f.append(e.timestamp)
        last_seen[e.user] = e.timestamp
        if e.outcome == "success":
            last_success[e.user] = (e.timestamp, e.country)
            recent_devices[e.device].add(e.user)
            if trusted:
                networks[e.user].add(network(e.ip))
                weekdays[e.user].add(e.timestamp.weekday())
                device_users[e.device].add(e.user)
    for e in baseline.itertuples(index=False):
        observe(e, True)
    ids, rows, occurrences = [], [], Counter()
    for e in out.itertuples(index=False):
        q, f = windows[e.user], failures[e.user]
        cutoff = e.timestamp - pd.Timedelta(minutes=10)
        while q and q[0] < cutoff:
            q.popleft()
        while f and f[0] < cutoff:
            f.popleft()
        success = e.outcome == "success"
        previous = last_success.get(e.user)
        hours = (e.timestamp-previous[0]).total_seconds()/3600 if previous else None
        km = distance(previous[1], e.country) if previous and previous[1] != e.country else None
        speed = km/max(hours, 1/60) if km is not None else None
        age = (e.timestamp-last_seen[e.user]).total_seconds()/86400 if e.user in last_seen else None
        known = bool(networks[e.user])
        shared = len(recent_devices[e.device]) + int(e.user not in recent_devices[e.device])
        flags = {
            "success_after_failures": success and len(f) >= policy.failures,
            "login_burst": len(q)+1 >= policy.burst,
            "dormant_return": age is not None and age >= policy.dormant_days,
            "new_weekday": known and e.timestamp.weekday() not in weekdays[e.user],
            "new_network": known and network(e.ip) not in networks[e.user],
            "shared_device": success and shared >= policy.shared_device_users and e.user not in device_users[e.device],
            "rapid_country_change": success and speed is not None and hours <= 24 and speed >= policy.travel_kmh,
            "unseen_account": not known,
        }
        triggered = [k for k, v in flags.items() if v]
        raw = json.dumps([str(e.timestamp), e.user, e.ip, e.country, e.device, e.outcome], ensure_ascii=False)
        occurrences[raw] += 1
        ids.append(hashlib.sha256(f"{raw}:{occurrences[raw]}".encode()).hexdigest()[:24])
        rows.append({**flags, "network": network(e.ip), "events_10m": len(q)+1,
                     "previous_failures_10m": len(f), "days_since_activity": age,
                     "approx_travel_kmh": speed, "device_account_count": shared,
                     "rule_ids": ",".join(triggered),
                     "mitre_techniques": ",".join(sorted({RULES[k][1] for k in triggered})),
                     "advanced_alert": bool(STRONG.intersection(triggered)),
                     "advanced_reasons": "; ".join(RULES[k][0] for k in triggered)})
        observe(e, False)
    out = pd.concat([out, pd.DataFrame(rows)], axis=1)
    out["event_id"] = ids
    out["combined_alert"] = out.alert | out.advanced_alert
    out["combined_severity"] = np.select(
        [out.severity.eq("high") | (out.alert & out.advanced_alert), out.combined_alert],
        ["high", "medium"], default="normal")
    return out


def incidents(events, gap_minutes=30):
    """Correlate flagged events by account and inactivity gap."""
    if gap_minutes < 1:
        raise ValueError("Gap must be positive")
    cols = ["incident_id", "user", "first_seen", "last_seen", "event_count", "severity", "sources", "event_ids"]
    records = []
    for user, group in events[events.combined_alert].sort_values("timestamp", kind="stable").groupby("user"):
        buckets = group.timestamp.diff().gt(pd.Timedelta(minutes=gap_minutes)).cumsum()
        for _, g in group.groupby(buckets):
            records.append(dict(incident_id=g.event_id.iloc[0], user=user,
                first_seen=g.timestamp.iloc[0], last_seen=g.timestamp.iloc[-1], event_count=len(g),
                severity="high" if g.combined_severity.eq("high").any() else "medium",
                sources=g.ip.nunique(), event_ids=",".join(g.event_id)))
    return pd.DataFrame(records, columns=cols).sort_values("first_seen", ascending=False)


def entity_summary(events, key):
    if key not in {"user", "ip", "country", "device", "network"}:
        raise ValueError("Unsupported entity")
    return (events.assign(failure=events.outcome.eq("failure"), high=events.combined_severity.eq("high"))
            .groupby(key).agg(events=("event_id", "size"), alerts=("combined_alert", "sum"),
                failures=("failure", "sum"), high=("high", "sum"), accounts=("user", "nunique"))
            .sort_values(["alerts", "events"], ascending=False).reset_index())


def heatmap(events):
    return pd.crosstab(events.timestamp.dt.dayofweek, events.timestamp.dt.hour).reindex(index=range(7), columns=range(24), fill_value=0)


def drift(baseline, events):
    """Total variation distance between observed categorical distributions."""
    rows = []
    for field in ["country", "outcome", "hour", "weekday"]:
        def values(frame):
            if field == "hour":
                return frame.timestamp.dt.hour
            if field == "weekday":
                return frame.timestamp.dt.dayofweek
            return frame[field]
        a, b = values(baseline).value_counts(normalize=True), values(events).value_counts(normalize=True)
        delta = a.subtract(b, fill_value=0).abs().sum()/2
        rows.append({"feature": field, "distance": float(delta), "review": bool(delta >= .2)})
    return pd.DataFrame(rows)


def evaluate(actual, events):
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    actual = np.asarray(actual, dtype=bool)
    if len(actual) != len(events) or not len(actual):
        raise ValueError("Labels must align with events")
    rows = []
    for name in ["ml_anomaly", "rule_alert", "advanced_alert", "combined_alert"]:
        predicted = events[name].to_numpy(dtype=bool)
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
        rows.append(dict(model=name, precision=precision_score(actual, predicted, zero_division=0),
                         recall=recall_score(actual, predicted, zero_division=0),
                         f1=f1_score(actual, predicted, zero_division=0), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn)))
    return pd.DataFrame(rows)


def anonymise(events, secret):
    """Keyed pseudonymisation, not full anonymisation."""
    import hmac
    if not isinstance(secret, bytes) or len(secret) < 16:
        raise ValueError("Use a secret of at least 16 bytes")
    out = events.copy()
    for key in ["user", "ip", "device", "network", "event_id"]:
        if key in out:
            out[key] = out[key].map(lambda v: hmac.new(secret, f"{key}:{v}".encode(), hashlib.sha256).hexdigest()[:24])
    return out


def json_export(events):
    return events.to_json(orient="records", date_format="iso", force_ascii=False)
