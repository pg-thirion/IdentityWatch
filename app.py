"""Local Streamlit analyst console."""
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from identitywatch.detector import Detector
from identitywatch.demo import make_demo
from identitywatch.export import safe_csv

st.set_page_config(page_title="IdentityWatch", page_icon="🔐", layout="wide")
st.title("IdentityWatch")
st.caption("Authentication intelligence | Unsupervised anomaly detection")
with st.sidebar:
    st.header("Analysis settings")
    mode = st.radio("Data source", ["Synthetic demo", "Upload CSV files"])
    contamination = st.slider("Expected baseline anomaly fraction", .01, .20, .03, .01)
    st.caption("Controls the Isolation Forest threshold. It is not a compromise probability.")
    st.divider()
    st.markdown("**Data handling**")
    st.caption("Data is processed in this application's memory. Use locally or in a trusted environment. Do not upload confidential logs to a public deployment.")
    if mode == "Upload CSV files":
        history = st.file_uploader("Historical baseline", type="csv")
        incoming = st.file_uploader("Later evaluation events", type="csv")
        st.caption("timestamp, user, ip, country, device, outcome (success/failure). UTC timestamps.")
        if history is None or incoming is None:
            st.info("Upload both files to begin.")
            st.stop()
        if any(f.size > 20 * 1024 * 1024 for f in (history, incoming)):
            st.error("Each file must be 20 MB or less.")
            st.stop()

try:
    if mode == "Synthetic demo":
        baseline, events = make_demo()
    else:
        baseline = pd.read_csv(history, nrows=100001, dtype=str)
        events = pd.read_csv(incoming, nrows=100001, dtype=str)
    with st.spinner("Learning historical behaviour and scoring new events..."):
        detector = Detector(contamination).fit(baseline)
        result = detector.score(events)
except (ValueError, OSError, UnicodeError) as exc:
    st.error(f"Unable to analyse the data: {exc}")
    st.stop()

alerts = result[result.alert]
a, b, c, d = st.columns(4)
a.metric("Events analysed", f"{len(result):,}")
b.metric("Flagged events", f"{len(alerts):,}")
c.metric("High severity", int(result.severity.eq("high").sum()))
d.metric("Affected accounts", alerts.user.nunique())

overview, investigation, methodology = st.tabs(["Overview", "Investigate", "Model & evaluation"])
with overview:
    l, r = st.columns(2)
    with l:
        st.subheader("Authentication activity (UTC)")
        timeline = result.set_index("timestamp").assign(events=1, alerts=result.alert.to_numpy().astype(int))
        timeline = timeline[["events", "alerts"]].resample("h").sum()
        st.line_chart(timeline)
    with r:
        st.subheader("Most flagged accounts")
        st.bar_chart(alerts.groupby("user").size().sort_values(ascending=False).head(10))
    st.subheader("Risk distribution")
    hist = result.groupby(pd.cut(result.baseline_percentile, bins=[-1, 50, 75, 90, 95, 100], include_lowest=True), observed=False).size()
    hist.index = hist.index.astype(str)
    st.bar_chart(hist)
    st.caption("Percentile is relative to training scores. It is not a likelihood of account compromise.")

with investigation:
    st.subheader("Alert queue")
    col1, col2 = st.columns(2)
    sev = col1.multiselect("Severity", ["high", "medium", "normal"], default=["high", "medium"])
    query = col2.text_input("Filter account or IP")
    view = result[result.severity.isin(sev)]
    if query:
        view = view[view.user.str.contains(query, case=False, regex=False) | view.ip.str.contains(query, regex=False)]
    view = view.sort_values(["alert", "anomaly_score"], ascending=False)
    cols = ["timestamp", "severity", "user", "ip", "country", "device", "outcome",
            "anomaly_score", "baseline_percentile", "reasons", "ml_anomaly", "rule_alert"]
    st.dataframe(view[cols], hide_index=True, use_container_width=True)
    st.download_button("Export filtered events (CSV)", safe_csv(view), "identitywatch-alerts.csv", "text/csv")
    if not view.empty:
        chosen = st.selectbox("Inspect event", view.index.tolist(),
            format_func=lambda i: f"{result.loc[i, 'timestamp']} | {result.loc[i, 'user']} | {result.loc[i, 'severity']}")
        event = result.loc[chosen]
        st.write("**Behavioural indicators:**", event.reasons)
        st.write("**Suggested analyst action:** Verify the sign-in with the account owner, inspect MFA and device logs, and correlate adjacent activity. Escalate if corroborated. Do not disable an account based on the score alone.")
        st.json({k: int(event[k]) for k in ["new_ip", "new_country", "new_device", "user_failures_10m", "source_failures_10m", "source_users_10m"]})

with methodology:
    st.subheader("How the detector works")
    st.markdown("Isolation Forest learns a historical feature distribution without attack labels. Features are computed chronologically, using only prior successful sign-ins for user profiles and a rolling ten-minute failure window. Evaluation profiles remain frozen. Separate deterministic rules supplement the ML score.")
    st.write("Baseline events:", len(baseline))
    st.write("ML alerts:", int(result.ml_anomaly.sum()), "| Rule alerts:", int(result.rule_alert.sum()))
    if mode == "Synthetic demo":
        st.subheader("Synthetic holdout evaluation")
        actual = events.sort_values("timestamp", kind="stable").scenario.ne("normal").to_numpy()
        for name, predicted in [("ML only", result.ml_anomaly), ("Rules only", result.rule_alert), ("Combined", result.alert)]:
            st.write(name, {"precision": round(precision_score(actual, predicted, zero_division=0), 3),
                "recall": round(recall_score(actual, predicted, zero_division=0), 3),
                "F1": round(f1_score(actual, predicted, zero_division=0), 3)})
        st.write("ML average precision:", round(average_precision_score(actual, result.anomaly_score), 3))
        st.caption("Generated scenarios only. These metrics do not establish real-world effectiveness.")
        st.download_button("Download demo baseline", safe_csv(baseline), "baseline.csv", "text/csv")
        st.download_button("Download demo evaluation", safe_csv(events), "events.csv", "text/csv")
    st.warning("This is an analyst aid, not an automated blocking system. VPNs, shared IPs, travel, sparse profiles and contaminated baselines can cause incorrect alerts.")
