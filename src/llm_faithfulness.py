#!/usr/bin/env python3
"""
LLM explainability with a QUANTIFIED FAITHFULNESS evaluation (Phase 4).

Prior LLM-for-explanation work emits free text that is never checked against reality. We instead
measure whether an LLM's explanation of a forecaster's failures *agrees with the ground-truth error
attribution*:

  1. Ground truth (`ground_truth_attribution`): for the leakage-free held-out test set, quantify how
     much each interpretable factor (high-demand, peak-hour, night, weekend, zone-volume, ...) really
     drives the model's absolute error — signed effect + significance (Mann-Whitney).
  2. Elicitation (`build_prompt` + `call_llm`): show the LLM a SAMPLE of test cells (features +
     prediction + error) — NOT the answer — and ask it to infer, per candidate factor, whether the
     factor increases/decreases error and to rank the top drivers, as strict JSON.
  3. Scoring (`score_faithfulness`): directional accuracy on truly-significant factors, top-3 driver
     recall, hallucination rate (claimed-significant but actually not), and Spearman rank agreement
     -> a composite faithfulness score in [0,1].
  4. Provider ablation (`run`): OpenAI / Anthropic / HuggingFace, plus a deterministic 'mock' provider
     so the framework is fully testable with no API key or cost.

Keys are read from the environment (never logged). Usage:
    python src/llm_faithfulness.py --data data/processed/chicago_taxi_sides.csv \
        --providers mock --out results/faithfulness_chicago.json
"""
import argparse
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.insert(0, str(Path(__file__).parent))
from splits import temporal_split_indices  # noqa: E402
from baseline_models import BaselineModelTrainer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SEED = 42
RUSH = {7, 8, 9, 17, 18, 19}
NIGHT = {0, 1, 2, 3, 4, 5, 22, 23}

# Interpretable candidate factors the LLM must reason about (name -> per-row boolean builder).
# Kept deliberately generic/observable so the task is genuine inference, not a giveaway.
FACTOR_FNS = {
    "high_demand":      lambda d: d["demand"] >= d["_p90"],
    "peak_hour":        lambda d: d["hour"].isin(RUSH),
    "night":            lambda d: d["hour"].isin(NIGHT),
    "weekend":          lambda d: d["is_weekend"] == 1,
    "high_volume_zone": lambda d: d["_zone_hi"],
    "low_volume_zone":  lambda d: d["_zone_lo"],
}
FACTOR_NAMES = list(FACTOR_FNS)


# --------------------------------------------------------------------------------------
# 1. Build the per-cell error dataset (same RF + leakage-free split as robustness_ci)
# --------------------------------------------------------------------------------------
def build_error_dataset(data_path, target="trip_count", zone_col="pickup_borough",
                        dt_col="pickup_datetime"):
    df = pd.read_csv(data_path)
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col]).reset_index(drop=True)
    idx = temporal_split_indices(df, dt_col, ratios=(0.7, 0.15, 0.15))
    trainval = df.iloc[np.concatenate([idx["train"], idx["val"]])]
    test = df.iloc[idx["test"]].copy()

    trainer = BaselineModelTrainer(task_type="regression", random_state=SEED)
    Xtr, ytr = trainer.prepare_features(trainval, target)
    Xte, yte = trainer.prepare_features(test, target)
    rf = RandomForestRegressor(n_estimators=100, min_samples_split=5, random_state=SEED, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    yte = np.asarray(yte)

    out = pd.DataFrame({
        "zone": test[zone_col].to_numpy(), "hour": test["hour"].to_numpy(),
        "is_weekend": test["is_weekend"].to_numpy(), "demand": yte,
        "pred": pred, "abs_err": np.abs(yte - pred),
    })
    # zone volume terciles (from this test set) + demand p90 -> factor helper columns
    zmean = out.groupby("zone")["demand"].transform("mean")
    lo, hi = zmean.quantile(1 / 3), zmean.quantile(2 / 3)
    out["_zone_lo"], out["_zone_hi"] = zmean <= lo, zmean >= hi
    out["_p90"] = np.percentile(out["demand"], 90)
    for f, fn in FACTOR_FNS.items():
        out[f] = fn(out).astype(int)
    return out


# --------------------------------------------------------------------------------------
# 2. Ground-truth error attribution
# --------------------------------------------------------------------------------------
def ground_truth_attribution(df, alpha=0.05, min_ratio=0.15):
    """Per factor: signed effect on abs_err (ratio of group means), direction, significance."""
    attr = {}
    for f in FACTOR_NAMES:
        e1 = df.loc[df[f] == 1, "abs_err"].to_numpy()
        e0 = df.loc[df[f] == 0, "abs_err"].to_numpy()
        if len(e1) < 10 or len(e0) < 10:
            attr[f] = {"direction": "none", "significant": False, "ratio": 1.0, "effect": 0.0,
                       "n1": int(len(e1))}
            continue
        m1, m0 = float(e1.mean()), float(e0.mean())
        ratio = m1 / m0 if m0 > 0 else float("inf")
        try:
            p = float(mannwhitneyu(e1, e0, alternative="two-sided").pvalue)
        except ValueError:
            p = 1.0
        sig = (p < alpha) and (abs(ratio - 1.0) >= min_ratio)
        attr[f] = {"direction": ("increases" if m1 > m0 else "decreases") if sig else "none",
                   "significant": bool(sig), "ratio": round(ratio, 3),
                   "effect": round(abs(np.log(ratio)) if ratio > 0 else 0.0, 3),
                   "p_value": round(p, 5), "mean_err_factor": round(m1, 2),
                   "mean_err_rest": round(m0, 2), "n1": int(len(e1))}
    ranked = sorted((f for f in FACTOR_NAMES if attr[f]["significant"]),
                    key=lambda f: attr[f]["effect"], reverse=True)
    return {"factors": attr, "ranked_drivers": ranked}


# --------------------------------------------------------------------------------------
# 3. Prompt + LLM clients
# --------------------------------------------------------------------------------------
def build_prompt(df, n_sample=40, seed=SEED):
    sample = df.sample(min(n_sample, len(df)), random_state=seed)
    cols = ["zone", "hour", "is_weekend", "demand", "pred", "abs_err"]
    table = sample[cols].round(1).to_csv(index=False)
    return f"""You are auditing a machine-learning model that forecasts hourly taxi demand per city zone.
Below is a random sample of held-out test cells. Columns: zone, hour (0-23), is_weekend (0/1),
demand (actual trips), pred (model prediction), abs_err (|actual - pred|).

{table}
For EACH of these candidate factors, infer from the data whether it tends to INCREASE or DECREASE
the model's absolute error (or has no clear effect), and rank the strongest error drivers.
Candidate factors: {", ".join(FACTOR_NAMES)}
  - high_demand: the cell's actual demand is in the top ~10%
  - peak_hour: hour in 7-9 or 17-19
  - night: hour in 22-5
  - weekend: is_weekend == 1
  - high_volume_zone / low_volume_zone: the zone's average demand is high / low

Respond with STRICT JSON only, no prose:
{{"factors": {{"<factor>": "increases" | "decreases" | "none", ...for all 6 factors...}},
  "ranked_drivers": ["<most important>", "<next>", "<next>"]}}"""


def call_llm(provider, prompt, model=None, timeout=60):
    """Return raw text from a provider. Reads keys from env. 'mock' needs no key."""
    if provider == "mock":
        return _MOCK_RESPONSE
    if provider in ("openai", "groq"):
        # Groq is OpenAI-API-compatible: same SDK, different base_url + a free open model.
        from openai import OpenAI
        if provider == "groq":
            client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                            base_url="https://api.groq.com/openai/v1", timeout=timeout)
            model = model or "llama-3.3-70b-versatile"
        else:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout)
            model = model or "gpt-4o-mini"
        kw = dict(model=model, temperature=0, messages=[{"role": "user", "content": prompt}])
        try:                                            # JSON mode when the model supports it
            r = client.chat.completions.create(**kw, response_format={"type": "json_object"})
        except Exception:                               # noqa: BLE001 — fall back to plain + tolerant parse
            r = client.chat.completions.create(**kw)
        return r.choices[0].message.content
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=timeout)
        r = client.messages.create(model=model or "claude-sonnet-5", max_tokens=1024,
                                   messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    if provider == "huggingface":
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=os.environ["HUGGINGFACE_API_KEY"], timeout=timeout)
        r = client.chat_completion(messages=[{"role": "user", "content": prompt}],
                                   model=model or "mistralai/Mistral-7B-Instruct-v0.3",
                                   max_tokens=512, temperature=0.01)
        return r.choices[0].message.content
    raise ValueError(f"unknown provider {provider!r}")


# Deterministic mock: a plausible-but-imperfect analyst answer, for tests + no-key runs.
_MOCK_RESPONSE = json.dumps({
    "factors": {"high_demand": "increases", "peak_hour": "increases", "night": "decreases",
                "weekend": "none", "high_volume_zone": "increases", "low_volume_zone": "decreases"},
    "ranked_drivers": ["high_demand", "high_volume_zone", "peak_hour"]})


def parse_llm_json(text):
    """Extract the first JSON object from an LLM reply, tolerating code fences / prose."""
    if text is None:
        raise ValueError("empty LLM response")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    obj = json.loads(m.group(0) if m else text)
    factors = {k: str(v).lower().strip() for k, v in obj.get("factors", {}).items()}
    ranked = [str(x).strip() for x in obj.get("ranked_drivers", [])]
    return {"factors": factors, "ranked_drivers": ranked}


# --------------------------------------------------------------------------------------
# 4. Faithfulness scoring
# --------------------------------------------------------------------------------------
def score_faithfulness(claims, truth):
    gt = truth["factors"]
    true_sig = [f for f in FACTOR_NAMES if gt[f]["significant"]]
    claimed = claims["factors"]

    # directional accuracy on truly-significant factors
    correct = [f for f in true_sig if claimed.get(f) == gt[f]["direction"]]
    dir_acc = len(correct) / len(true_sig) if true_sig else float("nan")

    # top-3 driver recall
    true_top = truth["ranked_drivers"][:3]
    llm_top = claims["ranked_drivers"][:3]
    recall = len(set(true_top) & set(llm_top)) / len(true_top) if true_top else float("nan")

    # hallucination: claimed 'increases/decreases' on factors that are truly not significant
    claimed_sig = [f for f in FACTOR_NAMES if claimed.get(f) in ("increases", "decreases")]
    halluc = [f for f in claimed_sig if not gt[f]["significant"]]
    halluc_rate = len(halluc) / len(claimed_sig) if claimed_sig else 0.0

    # Spearman rank agreement over the shared significant factors (by |effect|)
    shared = [f for f in true_sig if f in llm_top or f in claimed_sig]
    rho = float("nan")
    if len(true_top) >= 2 and len(llm_top) >= 2:
        allf = list(dict.fromkeys(true_top + llm_top))
        tr = [true_top.index(f) if f in true_top else len(true_top) for f in allf]
        lr = [llm_top.index(f) if f in llm_top else len(llm_top) for f in allf]
        if len(set(tr)) > 1 and len(set(lr)) > 1:
            rho = float(spearmanr(tr, lr).correlation)

    parts = [x for x in (dir_acc, recall, 1 - halluc_rate) if not np.isnan(x)]
    composite = float(np.mean(parts)) if parts else float("nan")
    return {"directional_accuracy": _r(dir_acc), "top3_driver_recall": _r(recall),
            "hallucination_rate": _r(halluc_rate), "rank_spearman": _r(rho),
            "faithfulness": _r(composite), "n_true_significant": len(true_sig),
            "true_direction_hits": correct, "hallucinated": halluc}


def _r(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), 3)


# --------------------------------------------------------------------------------------
# 5. Run (provider ablation)
# --------------------------------------------------------------------------------------
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "groq": "llama-3.3-70b-versatile",
                  "anthropic": "claude-sonnet-5", "huggingface": "mistralai/Mistral-7B-Instruct-v0.3",
                  "mock": "mock"}


def _key_present(provider):
    envk = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY", "groq": "GROQ_API_KEY"}.get(provider)
    return provider == "mock" or (envk and len(os.environ.get(envk, "")) >= 30)


def run(data_path, providers=("mock",), n_sample=40, out=None, models=None, repeats=1):
    df = build_error_dataset(data_path)
    truth = ground_truth_attribution(df)
    logger.info(f"Ground-truth error drivers (ranked): {truth['ranked_drivers']}")
    for f in FACTOR_NAMES:
        a = truth["factors"][f]
        logger.info(f"  {f:18s} {a['direction']:10s} ratio={a['ratio']:.2f} "
                    f"sig={a['significant']} (n1={a['n1']})")
    prompt = build_prompt(df, n_sample=n_sample)

    results = {}
    for p in providers:
        model = (models or {}).get(p) or DEFAULT_MODELS.get(p)
        if not _key_present(p):
            logger.info(f"\n[{p}] skipped — no API key in env")
            results[p] = {"available": False}
            continue
        # LLM output varies run-to-run even at temperature 0, so average over `repeats`.
        reps = 1 if p == "mock" else repeats
        runs = []
        for _ in range(reps):
            try:
                claims = parse_llm_json(call_llm(p, prompt, model=model))
                runs.append(score_faithfulness(claims, truth) | {"claims": claims})
            except Exception as e:                    # noqa: BLE001
                logger.info(f"\n[{p}] error: {type(e).__name__}: {str(e)[:120]}")
                runs.append({"error": f"{type(e).__name__}: {str(e)[:200]}"})
        ok = [r for r in runs if "faithfulness" in r and r["faithfulness"] is not None]
        if not ok:
            results[p] = {"available": True, "model": model, "runs": runs}
            continue
        faiths = np.array([r["faithfulness"] for r in ok], float)
        agg = {"available": True, "model": model, "n_runs": len(ok),
               "faithfulness_mean": _r(faiths.mean()),
               "faithfulness_std": _r(faiths.std(ddof=1) if len(faiths) > 1 else 0.0),
               "directional_accuracy_mean": _r(np.nanmean([r["directional_accuracy"] for r in ok])),
               "top3_driver_recall_mean": _r(np.nanmean([r["top3_driver_recall"] for r in ok])),
               "hallucination_rate_mean": _r(np.nanmean([r["hallucination_rate"] for r in ok])),
               "runs": runs}
        results[p] = agg
        logger.info(f"\n[{p}] faithfulness={agg['faithfulness_mean']}±{agg['faithfulness_std']} "
                    f"over {len(ok)} runs (dir_acc={agg['directional_accuracy_mean']}, "
                    f"recall={agg['top3_driver_recall_mean']}, halluc={agg['hallucination_rate_mean']})")

    out_obj = {"data": str(data_path), "n_test": int(len(df)), "n_sample": n_sample,
               "repeats": repeats, "ground_truth": truth, "providers": results}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out_obj, open(out, "w"), indent=2, default=float)
        logger.info(f"\nWrote {out}")
    return out_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/chicago_taxi_sides.csv")
    ap.add_argument("--providers", default="mock",
                    help="comma list: mock,groq,openai,anthropic,huggingface "
                         "(groq = free, OpenAI-compatible, open models like llama-3.3-70b)")
    ap.add_argument("--n-sample", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat each real provider N times for faithfulness mean±std (LLMs are noisy)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args.data, providers=[p.strip() for p in args.providers.split(",") if p.strip()],
        n_sample=args.n_sample, out=args.out, repeats=args.repeats)


if __name__ == "__main__":
    main()
