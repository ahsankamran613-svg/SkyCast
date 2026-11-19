"""EDA, multi-family ML training, evaluation, explainability, and model registry."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ChronologicalSplit",
    "compute_eda_summary",
    "compute_explanations_for_metric",
    "load_champion_bundle",
    "promote_champion_if_better",
    "save_eda_summary_json",
    "score_multi_horizon",
    "split_and_scale",
]


def __getattr__(name: str) -> Any:
    if name in {"compute_eda_summary", "save_eda_summary_json"}:
        from skycast.ml import eda

        return getattr(eda, name)
    if name in {"ChronologicalSplit", "score_multi_horizon", "split_and_scale"}:
        from skycast.ml import evaluate

        return getattr(evaluate, name)
    if name == "compute_explanations_for_metric":
        from skycast.ml.explain import compute_explanations_for_metric

        return compute_explanations_for_metric
    if name in {"load_champion_bundle", "promote_champion_if_better"}:
        from skycast.ml import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
