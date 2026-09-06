"""Lifecycle CLI for leakage-safe shared-fold experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryinsight.experiments.contracts import CandidateSpec, ExperimentConfig  # noqa: E402
from cryinsight.experiments.registry import (  # noqa: E402
    ExperimentProtocolError,
    derive_candidate,
    experiment_registry,
    load_experiment_config,
    registry_payload,
)
from cryinsight.experiments.runner import (  # noqa: E402
    ExperimentPreparation,
    prepare_experiment,
    resume_experiment,
    summarize_experiment,
    train_experiment,
    verify_experiment,
    ExperimentRunStore,
)
from cryinsight.training.artefacts import sha256_file


DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"
DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "data_set_dbl_split" / "train"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shared Experiment Engine (grouped OOF selection only)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--pipeline-run-id")
    parser.add_argument("--parent-experiment-run-id")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--experiment-run-id")
    parser.add_argument("--stage1-data-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--stage2-data-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default=None)
    parser.add_argument("--require-gpu", action="store_true", default=None)
    parser.add_argument("--mixed-precision", action="store_true", default=None)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument(
        "--continue-on-candidate-failure",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def _required(parser: argparse.ArgumentParser, args: Any, *names: str) -> None:
    missing = [name for name in names if getattr(args, name) in {None, ""}]
    if missing:
        parser.error(
            "this mode requires "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )


def _run_dir(runs_dir: Path, experiment_run_id: str) -> Path:
    identifier = str(experiment_run_id)
    if Path(identifier).name != identifier:
        raise ExperimentProtocolError(
            "experiment run ID must be a directory name, not a path"
        )
    return runs_dir.resolve() / identifier


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentProtocolError(f"Could not read parent evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExperimentProtocolError(f"Parent evidence must be an object: {path}")
    return payload


def _parent_candidates(parent_dir: Path) -> tuple[list[dict[str, Any]], dict[str, CandidateSpec]]:
    verify_experiment(parent_dir)
    verification = _read_json(parent_dir / "verification.json")
    if verification.get("status") != "complete":
        raise ExperimentProtocolError("Parent experiment run must be verified complete")
    leaderboard = _read_json(parent_dir / "leaderboard.json")
    rows = leaderboard.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ExperimentProtocolError("Parent leaderboard has no verified rows")
    if any(row.get("verification_status") != "complete" for row in rows):
        raise ExperimentProtocolError("Parent leaderboard contains incomplete candidates")
    resolved = _read_json(parent_dir / "resolved_config.json")
    raw_specs = resolved.get("candidate_specs")
    if not isinstance(raw_specs, list):
        raise ExperimentProtocolError("Parent candidate snapshot is missing")
    specs: dict[str, CandidateSpec] = {}
    for raw in raw_specs:
        try:
            spec = CandidateSpec(**raw)
        except (TypeError, ValueError) as exc:
            raise ExperimentProtocolError(f"Invalid parent candidate snapshot: {exc}") from exc
        specs[spec.candidate_id] = spec
    ordered = sorted(rows, key=lambda row: (int(row.get("rank", 10**9)), str(row.get("candidate_id", ""))))
    return ordered, specs


def _validate_parent_wave(wave: str, pipeline_run_id: str, parent: Mapping[str, Any]) -> None:
    predecessor = {'B_features': 'A', 'B_augmentation': 'B_features', 'B_loss': 'B_augmentation', 'C': 'B_loss'}
    if parent.get('pipeline_run_id') != pipeline_run_id:
        raise ExperimentProtocolError('Parent experiment belongs to a different pipeline run')
    if parent.get('config', {}).get('wave') != predecessor.get(wave):
        raise ExperimentProtocolError(f'Wave {wave} requires completed predecessor {predecessor.get(wave)}')


def _select_parent_candidates(wave: str, rows: list[dict[str, Any]], specs: Mapping[str, CandidateSpec]) -> list[str]:
    selected = []
    definitions: set[str] = set()
    for row in rows:
        spec = specs.get(str(row.get('candidate_id', '')))
        if spec is None:
            raise ExperimentProtocolError('Ranked parent candidate is absent from its frozen snapshot')
        eligible = spec.stage == 'stage2' and spec.adapter == 'neural'
        if wave == 'B_features':
            eligible = eligible and spec.feature_view in {'all_blocks', 'multi_branch_blocks'}
        definition = _candidate_definition_key(spec)
        if eligible and definition not in definitions:
            selected.append(spec.candidate_id)
            definitions.add(definition)
        if len(selected) == (2 if wave == 'C' else 1):
            break
    if not selected:
        raise ExperimentProtocolError(f'No compatible Stage 2 neural architecture for Wave {wave}')
    return selected


def _candidate_definition_key(spec: CandidateSpec) -> str:
    """Ignore naming/catalog metadata, retain the concrete execution definition."""
    payload = asdict(spec)
    for name in ('candidate_id', 'family'):
        payload.pop(name)
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def _derive_unique_variants(anchor: CandidateSpec, variants: Mapping[str, Any]) -> dict[str, CandidateSpec]:
    if 'anchor' not in variants or not isinstance(variants['anchor'], Mapping):
        raise ExperimentProtocolError('Wave B variants require an unchanged anchor')
    derive_candidate(anchor, 'anchor', variants['anchor'])
    resolved = {anchor.candidate_id: anchor}
    definitions = {_candidate_definition_key(anchor)}
    for variant_id, overrides in variants.items():
        if not isinstance(overrides, Mapping):
            raise ExperimentProtocolError(f'Variant {variant_id} overrides must be an object')
        candidate = derive_candidate(anchor, str(variant_id), overrides)
        definition = _candidate_definition_key(candidate)
        if definition in definitions:
            continue
        if candidate.candidate_id in resolved:
            raise ExperimentProtocolError(f'Duplicate resolved candidate ID: {candidate.candidate_id}')
        resolved[candidate.candidate_id] = candidate
        definitions.add(definition)
    return resolved


def _resolved_config(
    config: ExperimentConfig,
    *,
    runs_dir: Path,
    parent_experiment_run_id: str | None,
    pipeline_run_id: str | None = None,
) -> tuple[ExperimentConfig, dict[str, CandidateSpec]]:
    if config.candidate_source == "explicit":
        registry = experiment_registry()
        return config, {candidate_id: registry[candidate_id] for candidate_id in config.candidates}
    if not parent_experiment_run_id:
        raise ExperimentProtocolError(
            f"Wave {config.wave} requires --parent-experiment-run-id"
        )
    parent_dir = _run_dir(runs_dir, parent_experiment_run_id)
    rows, parent_specs = _parent_candidates(parent_dir)
    parent_protocol = _read_json(parent_dir / 'protocol.json')
    _validate_parent_wave(config.wave, pipeline_run_id or str(parent_protocol.get('pipeline_run_id')), parent_protocol)
    selected_ids = _select_parent_candidates(config.wave, rows, parent_specs)
    resolved_specs: dict[str, CandidateSpec] = {}
    if config.candidate_source == "parent_rank_1":
        anchor = parent_specs[selected_ids[0]]
        variants = config.parameters.get("variants")
        if not isinstance(variants, Mapping) or "anchor" not in variants:
            raise ExperimentProtocolError("Wave B config requires concrete variants including anchor")
        resolved_specs = _derive_unique_variants(anchor, variants)
    else:
        resolved_specs = {candidate_id: parent_specs[candidate_id] for candidate_id in selected_ids}
    runtime_parameters = {
        key: value for key, value in config.parameters.items() if key != "variants"
    }
    return (
        ExperimentConfig(
            schema_version=config.schema_version,
            wave=config.wave,
            seeds=config.seeds,
            selection_metric=config.selection_metric,
            candidates=tuple(resolved_specs),
            parameters=runtime_parameters,
            candidate_source="explicit",
            continue_on_candidate_failure=config.continue_on_candidate_failure,
        ),
        resolved_specs,
    )


def _runtime_config(config: ExperimentConfig, args: Any) -> ExperimentConfig:
    parameters = dict(config.parameters)
    parameters.update(
        {
            "device": args.device or parameters.get('device', 'auto'),
            "require_gpu": bool(parameters.get('require_gpu', False) if args.require_gpu is None else args.require_gpu),
            "mixed_precision": bool(parameters.get('mixed_precision', False) if args.mixed_precision is None else args.mixed_precision),
        }
    )
    if args.feature_cache_dir is not None:
        parameters["feature_cache_dir"] = str(args.feature_cache_dir.resolve())
    continue_on_failure = (
        config.continue_on_candidate_failure
        if args.continue_on_candidate_failure is None
        else bool(args.continue_on_candidate_failure)
    )
    return replace(
        config,
        parameters=parameters,
        continue_on_candidate_failure=continue_on_failure,
    )


def _check_runtime_request(args: Any, frozen: Mapping[str, Any]) -> None:
    parameters = frozen.get('parameters', {})
    for name in ('device', 'require_gpu', 'mixed_precision', 'feature_cache_dir'):
        requested = getattr(args, name)
        if requested is None:
            continue
        if name == 'feature_cache_dir':
            requested = str(requested.resolve())
        default = 'auto' if name == 'device' else False if name in {'require_gpu', 'mixed_precision'} else None
        if requested != parameters.get(name, default):
            raise ExperimentProtocolError(f'Runtime setting {name} differs from prepared contract; prepare a new run')
    if args.continue_on_candidate_failure is not None and args.continue_on_candidate_failure != frozen.get('continue_on_candidate_failure', True):
        raise ExperimentProtocolError('Failure policy differs from prepared contract')


def _audit_payload(config: ExperimentConfig) -> dict[str, Any]:
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("numpy", "sklearn", "librosa", "tensorflow")
    }
    selected = (
        registry_payload(config.candidates)
        if config.candidate_source == "explicit"
        else {
            "schema_version": "2.0",
            "selection_scope": "grouped_oof_only",
            "experiments": [],
        }
    )
    return {
        **selected,
        "config": asdict(config),
        "candidate_source": config.candidate_source,
        "dependencies_available": dependencies,
        "training_started": False,
        "heldout_test_available_for_ranking": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.audit_only:
        _required(parser, args, "config")
        config = load_experiment_config(args.config)
        print(json.dumps(_audit_payload(config), indent=2, sort_keys=True))
        return 0
    if args.prepare_only:
        _required(parser, args, "pipeline_run_id", "config")
        source_config = load_experiment_config(args.config)
        config, specs = _resolved_config(
            source_config,
            runs_dir=args.runs_dir,
            parent_experiment_run_id=args.parent_experiment_run_id,
            pipeline_run_id=args.pipeline_run_id,
        )
        config = _runtime_config(config, args)
        provenance = {}
        if args.parent_experiment_run_id:
            parent = _run_dir(args.runs_dir, args.parent_experiment_run_id)
            rows, parent_specs = _parent_candidates(parent)
            selected = _select_parent_candidates(config.wave, rows, parent_specs)
            provenance = {'parent_experiment_run_id': parent.name,
                          'pipeline_run_id': args.pipeline_run_id,
                          'parent_wave': _read_json(parent / 'protocol.json')['config']['wave'],
                          'selected_candidate_ids': selected,
                          'selected_parent_ranks': {row['candidate_id']: row['rank'] for row in rows if row['candidate_id'] in selected},
                          'selection_policy': 'best_compatible_stage2_neural_architecture',
                          'duplicate_definition_policy': 'retain_anchor_then_first_ranked_unique_definition',
                          'resolved_unique_candidate_ids': list(specs),
                          'parent_artefact_sha256': {name: sha256_file(parent / name) for name in ('verification.json', 'leaderboard.json', 'resolved_config.json')}}
        run_dir = prepare_experiment(
            ExperimentPreparation(
                project_root=PROJECT_ROOT,
                pipeline_run_id=args.pipeline_run_id,
                config=config,
                stage_data_roots={
                    "stage1": args.stage1_data_root,
                    "stage2": args.stage2_data_root,
                },
                runs_root=args.runs_dir,
                run_id=args.experiment_run_id,
                candidate_specs=specs,
                source_config=source_config,
                parent_provenance=provenance,
            )
        )
        print(run_dir)
        return 0
    if args.train:
        _required(parser, args, "pipeline_run_id", "config", "experiment_run_id")
        requested = load_experiment_config(args.config)
        run_dir = _run_dir(args.runs_dir, args.experiment_run_id)
        protocol = _read_json(run_dir / "protocol.json")
        ExperimentRunStore.open(run_dir).verify_integrity()
        if protocol.get("pipeline_run_id") != args.pipeline_run_id:
            raise ExperimentProtocolError("Prepared run belongs to a different pipeline run")
        frozen_config = protocol.get("config", {})
        if _read_json(run_dir / 'source_config.json') != json.loads(json.dumps(asdict(requested))):
            raise ExperimentProtocolError("Prepared run full source config does not match --config")
        _check_runtime_request(args, frozen_config)
        train_experiment(run_dir)
        print(run_dir)
        return 0
    if args.resume:
        _required(parser, args, "experiment_run_id")
        run_dir = _run_dir(args.runs_dir, args.experiment_run_id)
        _check_runtime_request(args, _read_json(run_dir / 'protocol.json')['config'])
        resume_experiment(run_dir)
        print(run_dir)
        return 0
    _required(parser, args, "experiment_run_id")
    run_dir = _run_dir(args.runs_dir, args.experiment_run_id)
    summarize_experiment(run_dir)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
