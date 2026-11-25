"""Batch analysis: python -m identitywatch --baseline history.csv --events new.csv."""
import argparse
from pathlib import Path
import pandas as pd
from .detector import Detector
from .demo import make_demo
from .export import safe_csv

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("identitywatch-results.csv"))
    parser.add_argument("--contamination", type=float, default=.03)
    args = parser.parse_args()
    if args.demo and (args.baseline or args.events):
        parser.error("--demo cannot be combined with input files.")
    if not args.demo and not (args.baseline and args.events):
        parser.error("Supply both CSV paths or --demo.")
    try:
        baseline, events = make_demo() if args.demo else (
            pd.read_csv(args.baseline, nrows=100001, dtype=str), pd.read_csv(args.events, nrows=100001, dtype=str))
        result = Detector(args.contamination).fit(baseline).score(events)
        with args.output.open("x", encoding="utf-8", newline="") as handle:
            handle.write(safe_csv(result))
        print(f"Scored {len(result)} events; {result.alert.sum()} alerts. Output: {args.output}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Analysis failed: {exc}\n")

if __name__ == "__main__":
    main()
