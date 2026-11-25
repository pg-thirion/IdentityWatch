"""Reproducible synthetic sign-ins. No real identities or routable addresses."""
import numpy as np
import pandas as pd

def make_demo(seed=42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    users = [f"employee-{i:02d}" for i in range(25)]
    def normal(days, offset):
        rows = []
        for d in range(offset, offset + days):
            for i, user in enumerate(users):
                for _ in range(3):
                    hour = int(np.clip(rng.normal(9 + i % 6, 1.4), 6, 19))
                    rows.append(dict(timestamp=start + pd.Timedelta(days=d, hours=hour,
                        minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60))),
                        user=user, ip=f"192.0.2.{i+1}", country="ZA", device=f"laptop-{i:02d}",
                        outcome="failure" if rng.random() < .025 else "success", scenario="normal"))
        return rows
    baseline = pd.DataFrame(normal(14, 0))
    evaluation = normal(3, 14)
    t = start + pd.Timedelta(days=15, hours=2)
    for i in range(12):
        evaluation.append(dict(timestamp=t + pd.Timedelta(seconds=20*i), user=users[0],
            ip="198.51.100.1", country="GB", device="unknown-a", outcome="failure",
            scenario="brute_force"))
    for i in range(15):
        evaluation.append(dict(timestamp=t + pd.Timedelta(hours=1, seconds=20*i),
            user=users[i], ip="198.51.100.2", country="US", device="unknown-b",
            outcome="failure", scenario="password_spray"))
    for i in range(5):
        evaluation.append(dict(timestamp=t + pd.Timedelta(hours=2, minutes=i), user=users[i],
            ip=f"203.0.113.{i+1}", country="AU", device="unknown-c", outcome="success",
            scenario="unusual_success"))
    return baseline.sort_values("timestamp").reset_index(drop=True), pd.DataFrame(evaluation).sort_values("timestamp", kind="stable").reset_index(drop=True)
