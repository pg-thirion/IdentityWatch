# IdentityWatch

Unsupervised authentication anomaly detection for security analysts. A Python and Streamlit portfolio project demonstrating identity threat detection and response (ITDR) without requiring labelled training data or external AI services.

## Features

- Isolation Forest learns historical sign-in behaviour and produces continuous anomaly scores.
- Causal features: new IP, country and device, circular UTC login-hour distance, rolling ten-minute failure counts and distinct targeted users, cold starts.
- Separate deterministic alerts for repeated failures, suspected password spraying and unfamiliar successful sign-ins.
- Interactive analyst dashboard with activity charts, alert queue, account/IP filters, event evidence and CSV export.
- Reproducible synthetic sign-ins and labelled holdout scenarios for evaluating ML, rules and combined alerts separately.
- Batch CLI, input validation and automated tests.

## Quick start

Python 3.11 or 3.12 recommended.

```bash
git clone https://github.com/pg-thirion/IdentityWatch.git
cd IdentityWatch
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

Open http://localhost:8501 and choose Synthetic demo. No accounts, credentials, or external API keys are needed.

For batch analysis:

```bash
python -m identitywatch --demo --output demo-results.csv
python -m identitywatch --baseline history.csv --events new.csv --output results.csv
python -m pytest -q
```

The CLI refuses to overwrite an existing output file.

## Input format

Both CSVs require the following headers:

```csv
timestamp,user,ip,country,device,outcome
2026-01-01T09:00:00Z,employee-01,192.0.2.1,ZA,laptop-01,success
```

Use stable pseudonymous account and device identifiers, ISO 8601 timestamps with offsets, valid IPv4/IPv6 addresses and success/failure outcomes. Country is supplied by the log source, not inferred from IP. Naive timestamps are interpreted as UTC. Include at least 100 baseline events, with all evaluation events strictly later than the baseline. Each file supports up to 100,000 events. Additional columns are ignored for modelling.

## Architecture

CSV ingestion -> validation -> chronological feature extraction -> Isolation Forest scoring -> separate rules -> investigation/export

Historical profiles learn only from successful events, and each event's novelty is measured before updating the profile. Evaluation replays the baseline and freezes user profiles while continuing rolling failure windows. This reduces look-ahead leakage and automatic trust of potentially compromised devices. The anomaly percentile is calculated against in-sample baseline scores; it is not calibrated to risk or a probability of compromise. Contamination controls the model threshold, not the expected operational alert rate. Tied timestamps are processed in input order.

The demo uses documentation IP ranges and fabricated accounts. It includes repeated failures, password spraying and unusual successful logins. Synthetic scenario labels are excluded from training and scoring. The dashboard computes precision, recall, F1 and average precision on the later synthetic holdout. Results are not a substitute for real-world validation.

## Security and privacy

Local-first, no external data calls, no model pickle loading, no automatic account blocking. CSV exports neutralise common spreadsheet formula prefixes. Do not commit real authentication logs. No authentication or multi-tenant access control is included, so do not expose the dashboard publicly with real logs.

For a production pilot, add authenticated ingestion from Entra ID or another identity provider, retention controls, encryption, analyst RBAC, audit logs, drift monitoring and independent validation. Establish a trusted historical baseline and review false positives caused by VPNs, NAT, shift patterns, travel and new employees. Rules and explanations are behavioural indicators, not model attributions. Severity is a triage heuristic; corroborate findings before remediation.

## Project layout

- `identitywatch/detector.py`: feature extraction and modelling
- `identitywatch/demo.py`: synthetic data
- `identitywatch/export.py`: spreadsheet-safe CSV export
- `identitywatch/__main__.py`: CLI
- `app.py`: Streamlit dashboard
- `tests/`: regression tests
