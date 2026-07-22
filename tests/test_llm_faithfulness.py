"""
Tests for the LLM faithfulness framework (src/llm_faithfulness.py). Pure logic only — no API
calls, no model training: ground-truth attribution, JSON parsing, and scoring.
"""
import json
import numpy as np
import pandas as pd
import pytest

from llm_faithfulness import (
    ground_truth_attribution, parse_llm_json, score_faithfulness, _key_present,
    FACTOR_NAMES, build_prompt,
)


def _synthetic_error_df(n=600, seed=0):
    """abs_err driven strongly by high_demand (+) and night (-); weekend/peak have no effect."""
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    demand = rng.gamma(2, 20, n)
    p90 = np.percentile(demand, 90)
    is_weekend = rng.integers(0, 2, n)
    err = 5 + 3.0 * (demand >= p90) * 20 - 2.0 * np.isin(hour, [0, 1, 2, 3, 4, 5, 22, 23]) * 3
    err = np.clip(err + rng.normal(0, 1, n), 0.1, None)
    df = pd.DataFrame({"zone": rng.choice(["A", "B", "C"], n), "hour": hour,
                       "is_weekend": is_weekend, "demand": demand, "pred": demand,
                       "abs_err": err})
    zmean = df.groupby("zone")["demand"].transform("mean")
    df["_zone_lo"] = zmean <= zmean.quantile(1 / 3)
    df["_zone_hi"] = zmean >= zmean.quantile(2 / 3)
    df["_p90"] = p90
    from llm_faithfulness import FACTOR_FNS
    for f, fn in FACTOR_FNS.items():
        df[f] = fn(df).astype(int)
    return df


def test_ground_truth_recovers_known_drivers():
    truth = ground_truth_attribution(_synthetic_error_df())
    gt = truth["factors"]
    assert gt["high_demand"]["direction"] == "increases" and gt["high_demand"]["significant"]
    assert gt["night"]["direction"] == "decreases" and gt["night"]["significant"]
    assert not gt["weekend"]["significant"]           # no real effect
    assert truth["ranked_drivers"][0] == "high_demand"   # strongest driver ranked first


def test_parse_llm_json_tolerates_fences_and_prose():
    fenced = "```json\n{\"factors\": {\"high_demand\": \"Increases\"}, \"ranked_drivers\": [\"high_demand\"]}\n```"
    out = parse_llm_json(fenced)
    assert out["factors"]["high_demand"] == "increases"      # lowercased
    prose = "Here is my analysis.\n{\"factors\": {}, \"ranked_drivers\": []}\nThanks!"
    assert parse_llm_json(prose)["ranked_drivers"] == []
    with pytest.raises(Exception):
        parse_llm_json(None)


def test_score_faithfulness_perfect_and_wrong():
    truth = ground_truth_attribution(_synthetic_error_df())
    gt = truth["factors"]
    # perfect claims: exact directions + true ranked drivers
    perfect = {"factors": {f: gt[f]["direction"] if gt[f]["significant"] else "none"
                           for f in FACTOR_NAMES},
               "ranked_drivers": truth["ranked_drivers"][:3]}
    s = score_faithfulness(perfect, truth)
    assert s["directional_accuracy"] == 1.0 and s["hallucination_rate"] == 0.0
    assert s["faithfulness"] >= 0.9

    # adversarial: flip every direction + hallucinate on a non-significant factor
    wrong = {"factors": {f: ("decreases" if gt[f]["direction"] == "increases" else "increases")
                         for f in FACTOR_NAMES},
             "ranked_drivers": ["weekend", "peak_hour"]}
    sw = score_faithfulness(wrong, truth)
    assert sw["directional_accuracy"] == 0.0
    assert sw["faithfulness"] < s["faithfulness"]


def test_key_present_logic(monkeypatch):
    assert _key_present("mock") is True
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert not _key_present("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "x" * 40)
    assert _key_present("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "short")       # placeholder-length -> treated absent
    assert not _key_present("openai")


def test_build_prompt_lists_all_factors():
    df = _synthetic_error_df(n=100)
    p = build_prompt(df, n_sample=20)
    for f in FACTOR_NAMES:
        assert f in p
    assert "STRICT JSON" in p
