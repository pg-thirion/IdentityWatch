"""Causal authentication features and unsupervised Isolation Forest scoring."""
from collections import defaultdict, deque
from dataclasses import dataclass, field
import ipaddress
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

REQUIRED = ("timestamp", "user", "ip", "country", "device", "outcome")
FEATURES = ("new_ip", "new_country", "new_device", "hour_distance",
            "user_failures_10m", "source_failures_10m", "source_users_10m",
            "failed", "cold_start")
MAX_ROWS = 100_000


def validate(frame):
    if not set(REQUIRED).issubset(frame.columns):
        raise ValueError("Required columns: " + ", ".join(REQUIRED))
    if not 0 < len(frame) <= MAX_ROWS:
        raise ValueError("Provide between 1 and 100,000 events.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names.")
    out = frame[list(REQUIRED)].copy()
    if out.isna().any().any():
        raise ValueError("Missing required values.")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="raise", format="mixed")
    if out["timestamp"].isna().any():
        raise ValueError("Invalid timestamps.")
    for col in REQUIRED[1:]:
        out[col] = out[col].astype(str).str.strip()
        if (out[col].eq("") | out[col].str.len().gt(256)).any():
            raise ValueError(f"Invalid {col} values.")
    out["country"] = out["country"].str.upper()
    out["outcome"] = out["outcome"].str.lower()
    if not out["outcome"].isin(["success", "failure"]).all():
        raise ValueError("Outcome must be success or failure.")
    try:
        out["ip"] = out["ip"].map(lambda x: str(ipaddress.ip_address(x)))
    except ValueError as exc:
        raise ValueError("Invalid IP address.") from exc
    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)


@dataclass
class Profile:
    ips: set = field(default_factory=set)
    countries: set = field(default_factory=set)
    devices: set = field(default_factory=set)
    sin_sum: float = 0
    cos_sum: float = 0
    count: int = 0

    def observe(self, event):
        self.ips.add(event.ip)
        self.countries.add(event.country)
        self.devices.add(event.device)
        angle = 2 * math.pi * (event.timestamp.hour + event.timestamp.minute / 60) / 24
        self.sin_sum += math.sin(angle)
        self.cos_sum += math.cos(angle)
        self.count += 1

    def distance(self, timestamp):
        if not self.count or math.hypot(self.sin_sum, self.cos_sum) / self.count < 0.1:
            return 0.0
        mean = math.atan2(self.sin_sum, self.cos_sum) * 24 / (2 * math.pi)
        hour = timestamp.hour + timestamp.minute / 60
        return abs((hour - mean + 12) % 24 - 12)


class FeatureState:
    def __init__(self):
        self.profiles = defaultdict(Profile)
        self.user_failures = defaultdict(deque)
        self.source_failures = defaultdict(deque)

    def transform(self, events, learn_profiles=False):
        rows = []
        for event in events.itertuples(index=False):
            p = self.profiles[event.user]
            user_q = self.user_failures[event.user]
            source_q = self.source_failures[event.ip]
            cutoff = event.timestamp - pd.Timedelta(minutes=10)
            while user_q and user_q[0] < cutoff:
                user_q.popleft()
            while source_q and source_q[0][0] < cutoff:
                source_q.popleft()
            failed = int(event.outcome == "failure")
            if failed:
                user_q.append(event.timestamp)
                source_q.append((event.timestamp, event.user))
            rows.append({
                "new_ip": int(bool(p.count) and event.ip not in p.ips),
                "new_country": int(bool(p.count) and event.country not in p.countries),
                "new_device": int(bool(p.count) and event.device not in p.devices),
                "hour_distance": p.distance(event.timestamp),
                "user_failures_10m": len(user_q),
                "source_failures_10m": len(source_q),
                "source_users_10m": len({u for _, u in source_q}),
                "failed": failed,
                "cold_start": int(not p.count),
            })
            # Freeze profiles during evaluation to prevent automatic trust of new devices.
            if learn_profiles and not failed:
                p.observe(event)
        return pd.DataFrame(rows, columns=FEATURES)


class Detector:
    def __init__(self, contamination=0.03, seed=42):
        if not 0.001 <= contamination <= 0.2:
            raise ValueError("Contamination must be between 0.001 and 0.2.")
        self.contamination = contamination
        self.seed = seed
        self.baseline = None

    def fit(self, baseline):
        events = validate(baseline)
        if len(events) < 100:
            raise ValueError("At least 100 baseline events are required.")
        self.baseline = events
        x = FeatureState().transform(events, learn_profiles=True)
        self.model = IsolationForest(n_estimators=200, contamination=self.contamination,
                                     random_state=self.seed, n_jobs=1).fit(x)
        self.reference = np.sort(-self.model.score_samples(x))
        return self

    def score(self, events):
        if self.baseline is None:
            raise ValueError("Fit a baseline first.")
        events = validate(events)
        if events.timestamp.min() <= self.baseline.timestamp.max():
            raise ValueError("Evaluation timestamps must follow the baseline.")
        state = FeatureState()
        state.transform(self.baseline, learn_profiles=True)
        x = state.transform(events)
        raw = -self.model.score_samples(x)
        # Empirical percentile, not a probability of compromise.
        ranks = 100 * np.searchsorted(self.reference, raw, side="left") / len(self.reference)
        result = pd.concat([events, x], axis=1)
        result["anomaly_score"] = raw.round(6)
        result["baseline_percentile"] = ranks.round(2)
        result["ml_anomaly"] = self.model.predict(x) == -1
        result["reasons"] = x.apply(self._reasons, axis=1)
        result["rule_alert"] = (
            (x.user_failures_10m >= 5) |
            ((x.source_failures_10m >= 5) & (x.source_users_10m >= 3)) |
            ((x.new_country == 1) & (x.new_device == 1) & (x.failed == 0))
        )
        result["alert"] = result.ml_anomaly | result.rule_alert
        result["severity"] = np.select(
            [result.ml_anomaly & result.rule_alert, result.alert],
            ["high", "medium"], default="normal")
        return result

    @staticmethod
    def _reasons(r):
        reasons = []
        for col, label in [("new_ip", "New IP"), ("new_country", "New country"),
                           ("new_device", "New device"), ("cold_start", "No user baseline")]:
            if r[col]:
                reasons.append(label)
        if r.hour_distance >= 5:
            reasons.append("Unusual UTC login hour")
        if r.user_failures_10m >= 5:
            reasons.append("Repeated user failures")
        if r.source_failures_10m >= 5 and r.source_users_10m >= 3:
            reasons.append("Possible password spray")
        if r.failed:
            reasons.append("Failed login")
        return "; ".join(reasons) or "No predefined behavioural indicator"
