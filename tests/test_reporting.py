import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gridsentinel.reporting import render as render_mod


def _fake_kpis():
    return {
        "raw_events": 200_000,
        "incidents": 17_000,
        "compression": 11.8,
        "critical": 580,
        "noise_rate": 0.833,
        "capacity": 4000,
        "headroom": 0.527,
        "weeks_to_70": "0.7",
        "horizon_hours": 672,
        "avg_kw": 2068,
        "total_kwh": 694_792,
        "idle_kwh": 1316,
        "idle_dollars": 79,
        "perf_w": 0.9157,
    }


def _fake_clusters():
    return pd.DataFrame(
        [
            {
                "cluster_id": 0,
                "cluster_label": "thermal_throttle · gpu · info · ~15:00 (223×)",
                "count": 223,
                "critical_count": 0,
                "all_benign_ratio": 1.0,
            }
        ]
    )


def test_deterministic_narrative_runs_without_key():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
        narrative = render_mod._deterministic_narrative(_fake_kpis(), _fake_clusters())
        assert "12" in narrative or "11.8" in narrative or "10×" in narrative  # references compression
        assert "thermal_throttle" in narrative
        assert "$79" in narrative or "79" in narrative


def test_llm_narrative_returns_none_when_no_key():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
        assert render_mod._llm_narrative(_fake_kpis(), _fake_clusters()) is None


def test_llm_narrative_calls_sdk_when_key_set(monkeypatch):
    """Confirms dispatch — does NOT make a real API call."""
    import sys
    import types

    captured: dict = {}

    class FakeMessage:
        def __init__(self, text):
            self.text = text

    class FakeResponse:
        def __init__(self, text):
            self.content = [FakeMessage(text)]

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse("This is a fake LLM narrative for testing.")

    class FakeAnthropic:
        def __init__(self, api_key=None):
            captured["api_key_used"] = api_key
            self.messages = FakeMessages()

    fake_module = types.SimpleNamespace(Anthropic=FakeAnthropic)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    out = render_mod._llm_narrative(_fake_kpis(), _fake_clusters())
    assert out is not None
    assert "fake LLM narrative" in out
    assert captured["api_key_used"] == "test-key-not-real"
    assert "max_tokens" in captured
    assert "messages" in captured


def test_report_template_renders():
    """End-to-end smoke: render_weekly() writes a markdown file when no key set."""
    with patch.object(render_mod, "_build_kpis", return_value=(_fake_kpis(), _fake_clusters())):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            path = render_mod.render_weekly(week="2026-W21-test")
    assert Path(path).exists()
    text = Path(path).read_text()
    assert "12.0×" in text or "11.8×" in text
    assert "Deterministic mode" in text
    Path(path).unlink()
