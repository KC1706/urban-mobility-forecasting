# Roadmap

This project's core (data pipeline, baselines, robustness stress-testing, LLM
explainability) is functional. This roadmap tracks the work to raise it to
research-grade and resolve known limitations.

## ST-HAE: proper re-implementation (highest priority)

The current `src/st_hae_algorithm.py` is a **prototype** and is documented honestly as
such: it implements the spatial/temporal/hierarchical/ensemble *concept* in NumPy with
an identity-weight "GCN" and unlearned self-attention. On the held-out test set it
**underperforms** the RF/XGBoost baselines (R² ≈ 0.43 vs 0.98). It is included to show
the architecture and as a negative-result baseline, **not** as a validated contribution.

Planned real implementation:
- Replace the NumPy identity-matrix graph step with a trained **GCN** (`torch_geometric`
  `GCNConv`) over a zone-adjacency graph built from geographic proximity + demand correlation.
- Replace the hand-rolled softmax self-attention with a trained
  `torch.nn.MultiheadAttention` temporal block.
- Train the full model end-to-end (PyTorch), with early stopping and proper
  train/val/test temporal splits.
- Add an **honest ablation**: spatial-only / temporal-only / hierarchical-only / full,
  reporting MAE/RMSE/MAPE with confidence intervals.
- Compare against a real published baseline (e.g. **STGCN** or **Graph WaveNet**), not
  just RF/XGBoost.

## Evaluation rigor
- Extend the time window well beyond the current 32 days.
- Add uncertainty quantification (quantile regression / conformal prediction) to the
  robustness layer so tail-error claims have calibrated coverage.
- Report statistical significance on the per-zone / peak-hour robustness gaps.

## LLM explainability
- Move from free-text explanations toward a small evaluation: do generated explanations
  agree with ground-truth failure causes (per-zone / per-hour error attribution)?

## Reproducibility / engineering
- Pin dependency versions; add a `Makefile` or CLI entry points.
- Add unit tests for the data pipeline and metrics.
- Containerize (Dockerfile) and add a CI workflow.
