"""Immutable public contracts for shared experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol


def _required_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} cannot be empty")
    return text


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    stage: str
    family: str
    feature_view: str
    adapter: str
    model: str
    augmentation: str
    normalization: str
    loss: str
    selection_metric: str = "oof_macro_f1"
    requires_gpu: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "family",
            "feature_view",
            "model",
            "augmentation",
            "normalization",
            "loss",
            "selection_metric",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), name=name),
            )
        stage = _required_text(self.stage, name="stage").lower()
        adapter = _required_text(self.adapter, name="adapter").lower()
        if stage not in {"stage1", "stage2"}:
            raise ValueError("stage must be 'stage1' or 'stage2'")
        if adapter not in {"classical", "neural"}:
            raise ValueError("adapter must be 'classical' or 'neural'")
        if not self.selection_metric.lower().startswith("oof_"):
            raise ValueError("selection_metric must be a grouped OOF metric")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "parameters", dict(self.parameters))

    @property
    def experiment_id(self) -> str:
        """Compatibility name used by the earlier definition scripts."""

        return self.candidate_id

    @property
    def feature_set(self) -> str:
        """Compatibility name used by the earlier registry payload."""

        return self.feature_view


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str
    wave: str
    seeds: tuple[int, ...]
    selection_metric: str
    candidates: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    candidate_source: str = "explicit"
    continue_on_candidate_failure: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _required_text(self.schema_version, name="schema_version"),
        )
        object.__setattr__(self, "wave", _required_text(self.wave, name="wave"))
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
            raise ValueError("seeds must be non-empty, unique, non-negative integers")
        metric = _required_text(self.selection_metric, name="selection_metric").lower()
        if not metric.startswith("oof_"):
            raise ValueError("selection_metric must be a grouped OOF metric")
        source = _required_text(self.candidate_source, name="candidate_source").lower()
        if source not in {"explicit", "parent_rank_1", "parent_top_2"}:
            raise ValueError(
                "candidate_source must be explicit, parent_rank_1, or parent_top_2"
            )
        candidates = tuple(str(value).strip() for value in self.candidates)
        if any(not value for value in candidates) or len(candidates) != len(set(candidates)):
            raise ValueError("candidates must contain unique non-empty IDs")
        if source == "explicit" and not candidates:
            raise ValueError("explicit experiment configs require candidates")
        if source != "explicit" and candidates:
            raise ValueError("parent-derived experiment configs resolve candidates later")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "selection_metric", metric)
        object.__setattr__(self, "candidate_source", source)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "parameters", dict(self.parameters))


@dataclass(frozen=True)
class FoldRequest:
    experiment_run_id: str
    pipeline_run_id: str
    candidate: CandidateSpec
    seed: int
    fold: int
    train_records: tuple[Any, ...]
    validation_records: tuple[Any, ...]
    label_order: tuple[str, ...]
    output_dir: Path
    runtime: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FoldResult:
    candidate_id: str
    seed: int
    fold: int
    validation_record_ids: tuple[str, ...]
    true_labels: tuple[str, ...]
    probabilities: Any
    model_path: Path
    manifest_path: Path


class CandidateAdapter(Protocol):
    def fit_predict_fold(self, request: FoldRequest) -> FoldResult:
        """Fit on one training fold and predict original validation records."""

        ...
