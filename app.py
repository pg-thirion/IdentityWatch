"""Local Streamlit analyst console."""
import hashlib
import secrets
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from identitywatch.detector import Detector
from identitywatch.demo import make_demo
from identitywatch.export import safe_csv
from identitywatch.advanced import (Policy, RULES, enrich, incidents, entity_summary, heatmap,
                                   drift, evaluate, anonymise, json_export)

st.set_page_config(page_title="IdentityWatch", page_icon="🔐", layout="wide")
st.title("IdentityWatch")
st.caption("Authentication intelligence | Unsupervised anomaly detection")
with st.sidebar:
    st.header("Analysis settings")
    mode = st.radio("Data source", ["Synthetic demo", "Upload CSV files"])
    contamination = st.slider("Expected baseline anomaly fraction", .01, .20, .03, .01)
    st.caption("Controls the Isolation Forest threshold. It is not a compromise probability.")
    with st.expander("Behavioural detection policy"):
        policy = Policy(
            burst=st.number_input("Events in ten minutes", min_value=2, max_value=100, value=8),
            failures=st.number_input("Failures before success", min_value=1, max_value=100, value=3),
            dormant_days=st.number_input("Dormancy days", min_value=1, max_value=365, value=14),
            travel_kmh=st.number_input("Approximate travel speed (km/h)", min_value=100, max_value=2000, value=900),
            shared_device_users=st.number_input("Distinct device accounts", min_value=2, max_value=100, value=3))
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
    fingerprints = ("demo-v1",) if mode == "Synthetic demo" else (
        hashlib.sha256(history.getvalue()).hexdigest(), hashlib.sha256(incoming.getvalue()).hexdigest())
    analysis_key = (fingerprints, contamination, policy)
    # Cache only the current analysis in this session, never in a shared global cache.
    if st.session_state.get("analysis_key") != analysis_key:
        if mode == "Synthetic demo":
            baseline, events = make_demo()
        else:
            baseline = pd.read_csv(history, nrows=100001, dtype=str)
            events = pd.read_csv(incoming, nrows=100001, dtype=str)
        with st.spinner("Learning historical behaviour and scoring new events..."):
            detector = Detector(contamination).fit(baseline)
            result = enrich(detector.baseline, detector.score(events), policy)
        st.session_state["analysis_data"] = (baseline, events, detector, result)
        st.session_state["analysis_key"] = analysis_key
    baseline, events, detector, result = st.session_state["analysis_data"]
except (ValueError, OSError, UnicodeError) as exc:
    st.error(f"Unable to analyse the data: {exc}")
    st.stop()

alerts = result[result.combined_alert]
a, b, c, d = st.columns(4)
a.metric("Events analysed", f"{len(result):,}")
b.metric("Flagged events", f"{len(alerts):,}")
c.metric("High severity", int(result.combined_severity.eq("high").sum()))
d.metric("Affected accounts", alerts.user.nunique())

overview, investigation, casework, entities, methodology, exports = st.tabs(["Overview", "Investigate", "Incidents", "Entities", "Model & evaluation", "Exports"])
with overview:
    l, r = st.columns(2)
    with l:
        st.subheader("Authentication activity (UTC)")
        timeline = result.set_index("timestamp").assign(events=1, alerts=result.combined_alert.to_numpy().astype(int))
        span = result.timestamp.max() - result.timestamp.min()
        # Avoid allocating empty time buckets for very sparse or long histories.
        freq = "h" if span <= pd.Timedelta(days=31) else "D" if span <= pd.Timedelta(days=366) else "MS"
        timeline = timeline[["events", "alerts"]].groupby(pd.Grouper(freq=freq)).sum()
        if len(timeline) > 2000:
            timeline = timeline.tail(2000)
            st.caption("Showing the latest 2,000 time buckets.")
        st.line_chart(timeline)
    with r:
        st.subheader("Most flagged accounts")
        st.bar_chart(alerts.groupby("user").size().sort_values(ascending=False).head(10))
    st.subheader("UTC weekday and hour activity")
    hours = heatmap(result)
    hours.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    maximum = max(1, int(hours.to_numpy().max()))
    styled = hours.style.map(lambda v: f"background-color: rgba(31, 151, 171, {0.08 + 0.7*v/maximum:.3f})")
    st.dataframe(styled, use_container_width=True)
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
    view = result[result.combined_severity.isin(sev)]
    filters = st.columns(3)
    countries = filters[0].multiselect("Country", sorted(result.country.unique()))
    outcomes = filters[1].multiselect("Outcome", ["success", "failure"])
    indicators = filters[2].multiselect("Behavioural indicators", list(RULES), format_func=lambda k: RULES[k][0])
    if countries:
        view = view[view.country.isin(countries)]
    if outcomes:
        view = view[view.outcome.isin(outcomes)]
    if indicators:
        view = view[view[indicators].any(axis=1)]
    if query:
        view = view[view.user.str.contains(query, case=False, regex=False) | view.ip.str.contains(query, regex=False)]
    view = view.sort_values(["combined_alert", "anomaly_score"], ascending=False)
    cols = ["timestamp", "combined_severity", "user", "ip", "country", "device", "outcome",
            "anomaly_score", "baseline_percentile", "reasons", "advanced_reasons", "mitre_techniques", "ml_anomaly", "rule_alert", "advanced_alert"]
    st.dataframe(view[cols], hide_index=True, use_container_width=True)
    st.download_button("Export filtered events (CSV)", safe_csv(view), "identitywatch-alerts.csv", "text/csv")
    if not view.empty:
        chosen = st.selectbox("Inspect event", view.index.tolist(),
            format_func=lambda i: f"{result.loc[i, 'timestamp']} | {result.loc[i, 'user']} | {result.loc[i, 'combined_severity']}")
        event = result.loc[chosen]
        st.write("**Behavioural indicators:**", event.reasons)
        st.write("**Additional indicators:**", event.advanced_reasons or "None")
        st.write("**ATT&CK references:**", event.mitre_techniques or "None")
        st.write("**Event ID:**", event.event_id)
        st.write("**Account timeline (latest 200 events)**")
        st.dataframe(result[result.user.eq(event.user)][cols].tail(200), hide_index=True, use_container_width=True)
        st.write("**Suggested analyst action:** Verify the sign-in with the account owner, inspect MFA and device logs, and correlate adjacent activity. Escalate if corroborated. Do not disable an account based on the score alone.")
        st.json({k: int(event[k]) for k in ["new_ip", "new_country", "new_device", "user_failures_10m", "source_failures_10m", "source_users_10m"]})

with casework:
    st.subheader("Correlated incidents")
    gap = st.slider("Incident inactivity gap (minutes)", 5, 120, 30, 5)
    cases = incidents(result, gap)
    st.metric("Incident groups", len(cases))
    # State is session-local and keyed by the analysed event set and grouping policy.
    dataset = hashlib.sha256((repr(analysis_key) + f":{gap}").encode()).hexdigest()
    if st.session_state.get("case_dataset") != dataset:
        st.session_state["case_dataset"] = dataset
        st.session_state["case_notes"] = {}
    notes = st.session_state["case_notes"]
    if not cases.empty:
        selected = st.selectbox("Select incident", cases.incident_id.tolist())
        old = notes.get(selected, {"status": "New", "notes": ""})
        with st.form("incident_review"):
            status_options = ["New", "Investigating", "Confirmed", "False positive", "Closed"]
            status = st.selectbox("Disposition", status_options, index=status_options.index(old["status"]), key=f"status:{dataset}:{selected}")
            note = st.text_area("Analyst notes", value=old["notes"], max_chars=2000, key=f"notes:{dataset}:{selected}")
            if st.form_submit_button("Save review"):
                notes[selected] = {"status": status, "notes": note}
        cases["status"] = cases.incident_id.map(lambda i: notes.get(i, {}).get("status", "New"))
        cases["notes"] = cases.incident_id.map(lambda i: notes.get(i, {}).get("notes", ""))
        st.dataframe(cases.drop(columns="event_ids"), hide_index=True, use_container_width=True)
        st.download_button("Export incident register", safe_csv(cases), "incidents.csv", "text/csv")
    else:
        st.info("No correlated incidents for the current analysis.")
    st.caption("Reviews exist only in this browser session. Export the register before closing it.")

with entities:
    st.subheader("Entity investigation")
    key = st.selectbox("Group by", ["user", "ip", "country", "device", "network"])
    summary = entity_summary(result, key)
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.download_button("Export entity summary", safe_csv(summary), "entities.csv", "text/csv")
    st.subheader("Behavioural indicator frequency")
    st.bar_chart(result[list(RULES)].sum().rename("events"))

with methodology:
    st.subheader("How the detector works")
    st.markdown("Isolation Forest learns a historical feature distribution without attack labels. Features are computed chronologically, using only prior successful sign-ins for user profiles and a rolling ten-minute failure window. Evaluation profiles remain frozen. Separate deterministic rules supplement the ML score.")
    st.subheader("Behavioural distribution drift")
    st.dataframe(drift(detector.baseline, result), hide_index=True, use_container_width=True)
    st.caption("Total variation distance ranges from 0 to 1. Values of 0.2 or more warrant review, not automatic retraining.")
    st.write("Baseline events:", len(baseline))
    st.write("ML alerts:", int(result.ml_anomaly.sum()), "| Original rule alerts:", int(result.rule_alert.sum()), "| Advanced rule alerts:", int(result.advanced_alert.sum()))
    if mode == "Synthetic demo":
        st.subheader("Synthetic holdout evaluation")
        actual = events.sort_values("timestamp", kind="stable").scenario.ne("normal").to_numpy()
        for name, predicted in [("ML only", result.ml_anomaly), ("Rules only", result.rule_alert), ("Combined (all rules + ML)", result.combined_alert)]:
            st.write(name, {"precision": round(precision_score(actual, predicted, zero_division=0), 3),
                "recall": round(recall_score(actual, predicted, zero_division=0), 3),
                "F1": round(f1_score(actual, predicted, zero_division=0), 3)})
        st.dataframe(evaluate(actual, result), hide_index=True, use_container_width=True)
        st.write("ML average precision:", round(average_precision_score(actual, result.anomaly_score), 3))
        st.caption("Generated scenarios only. These metrics do not establish real-world effectiveness.")
        st.download_button("Download demo baseline", safe_csv(baseline), "baseline.csv", "text/csv")
        st.download_button("Download demo evaluation", safe_csv(events), "events.csv", "text/csv")
    st.warning("Travel uses approximate country centroids for nine countries, not precise locations. Shared devices and VPNs can produce false positives.")
    st.warning("This is an analyst aid, not an automated blocking system. VPNs, shared IPs, travel, sparse profiles and contaminated baselines can cause incorrect alerts.")

with exports:
    st.subheader("Evidence exports")
    st.download_button("Full analysis CSV", safe_csv(result), "identitywatch-results.csv", "text/csv")
    st.download_button("Full analysis JSON", json_export(result), "identitywatch-results.json", "application/json")
    if "export_secret" not in st.session_state:
        st.session_state["export_secret"] = secrets.token_bytes(32)
    st.download_button("Pseudonymised JSON", json_export(anonymise(result, st.session_state["export_secret"])),
                       "identitywatch-pseudonymised.json", "application/json")
    st.caption("Keyed identifiers are consistent during this session. Countries, timestamps and behavioural signals remain visible. Pseudonymisation does not make the export anonymous.")
