import json
import sys
import pandas as pd
import pytest
from identitywatch.__main__ import main
from identitywatch.detector import Detector
from identitywatch.demo import make_demo
from identitywatch.advanced import enrich, evaluate, incidents


@pytest.fixture(scope="module")
def analysis():
    baseline, events = make_demo()
    model = Detector().fit(baseline)
    return events, enrich(model.baseline, model.score(events))


def test_integrated_demo(analysis):
    events, result = analysis
    actual = events.scenario.ne("normal")
    assert len(result) == len(events)
    assert result.event_id.is_unique
    assert result.loc[actual, "combined_alert"].mean() >= .8
    assert result.loc[~actual, "combined_alert"].mean() < .3
    assert incidents(result).event_count.sum() == result.combined_alert.sum()
    assert evaluate(actual, result).shape[0] == 4


@pytest.mark.parametrize("format", ["csv", "json"])
def test_cli_demo(tmp_path, monkeypatch, format):
    path = tmp_path / f"results.{format}"
    monkeypatch.setattr(sys, "argv", ["identitywatch", "--demo", "--format", format, "--output", str(path)])
    main()
    result = pd.read_csv(path) if format == "csv" else pd.DataFrame(json.loads(path.read_text()))
    assert len(result) == 257
    assert result.combined_alert.any()
    assert result.event_id.is_unique


def test_cli_real_csv_input(tmp_path, monkeypatch):
    baseline, events = make_demo()
    history, incoming, output = [tmp_path / f"{name}.csv" for name in ["baseline", "incoming", "output"]]
    baseline.to_csv(history, index=False)
    events.to_csv(incoming, index=False)
    monkeypatch.setattr(sys, "argv", ["identitywatch", "--baseline", str(history), "--events", str(incoming), "--output", str(output)])
    main()
    result = pd.read_csv(output)
    assert len(result) == len(events)
    assert "scenario" not in result.columns


def test_cli_invalid_policy(tmp_path, monkeypatch):
    output = tmp_path / "invalid.csv"
    monkeypatch.setattr(sys, "argv", ["identitywatch", "--demo", "--burst", "0", "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2 and not output.exists()


def test_cli_no_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "existing.csv"
    output.write_text("keep this")
    monkeypatch.setattr(sys, "argv", ["identitywatch", "--demo", "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2 and output.read_text() == "keep this"


def test_cli_requires_input(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["identitywatch"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
