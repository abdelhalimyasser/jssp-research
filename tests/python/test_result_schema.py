"""Tests for immutable RunManifest, confirmatory integrity checks, and RunResultRecord."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from jssp_research.benchmarks import get_catalog_entry, parse_jssp_file
from jssp_research.experiments import (
    CANONICAL_SERIALIZATION_VERSION,
    BudgetSpec,
    ConfirmatoryIntegrityError,
    ExperimentPhase,
    ManifestIntegrityError,
    PairingMode,
    ProtocolStage,
    RunManifest,
    build_seed_context,
    canonical_json_bytes,
    canonical_json_sha256,
    create_run_manifest,
    load_and_validate_manifest,
    write_manifest,
)
from jssp_research.results import (
    ResultValidationError,
    RunStatus,
    create_run_result_record,
    load_and_validate_result,
    write_result_record,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sample_dev_manifest(tmp_dir: Path, *, eval_budget: int = 1000) -> RunManifest:
    ft06_entry = get_catalog_entry("ft06", repo_root=REPO_ROOT)
    parsed = parse_jssp_file(
        REPO_ROOT / "tests" / "fixtures" / "medium_jssp.txt", instance_id="ft06"
    )
    seed_ctx = build_seed_context(
        replicate_index=0,
        instance_id="ft06",
        pairing_mode=PairingMode.PAIRED_INIT,
        pairing_group="phase_b_shared",
        configuration_id="B01",
    )
    return create_run_manifest(
        run_id="run-ft06-B01-r00",
        experiment_id="m3-test-exp",
        phase=ExperimentPhase.PHASE_B,
        protocol_stage=ProtocolStage.DEVELOPMENT,
        configuration_id="B01",
        algorithm_id="steepest_descent",
        algorithm_version="0.1.0",
        neighborhood_id="N5",
        configuration_snapshot={
            "config_id": "B01",
            "neighborhood": "N5",
            "eval_budget": eval_budget,
        },
        catalog_entry=ft06_entry,
        parsed_instance=parsed,
        seed_context=seed_ctx,
        budget=BudgetSpec(objective_evaluation_budget=eval_budget, population_size=64),
        repo_root=REPO_ROOT,
    )


def test_manifest_schema_immutability_and_hash_validation(tmp_path: Path) -> None:
    """Verify valid dev manifest persistence, frozen immutability, and tamper detection."""
    manifest = _sample_dev_manifest(tmp_path)
    out_file = write_manifest(manifest, tmp_path / "manifest.json")

    with pytest.raises(FileExistsError):
        write_manifest(manifest, out_file)

    reloaded = load_and_validate_manifest(out_file)
    assert reloaded == manifest
    assert reloaded.source_code.git.dirty is True
    assert reloaded.source_code.git.diff_sha256 is not None

    with pytest.raises(ValidationError):
        manifest.identity.run_id = "mutated"  # type: ignore[misc]

    # Tamper with manifest_sha256
    raw = json.loads(out_file.read_text("utf-8"))
    tampered_hash = dict(raw)
    tampered_hash["manifest_sha256"] = "0" * 64
    with pytest.raises(ManifestIntegrityError, match="manifest_sha256 mismatch"):
        RunManifest.model_validate(tampered_hash)

    # Missing required field
    missing = dict(raw)
    del missing["budget"]
    with pytest.raises(ValidationError):
        RunManifest.model_validate(missing)

    # Incompatible schema version
    bad_ver = dict(raw)
    bad_ver["identity"] = dict(raw["identity"], schema_version="jssp-manifest-v99")
    with pytest.raises(ManifestIntegrityError, match="Unsupported manifest schema version"):
        RunManifest.model_validate(bad_ver)

    # Invalid budget
    with pytest.raises(ValueError, match="strictly positive"):
        BudgetSpec(objective_evaluation_budget=0)

    # Invalid seed count/stream mismatch
    bad_streams = dict(raw)
    bad_streams["randomness"] = dict(raw["randomness"], streams=raw["randomness"]["streams"][:8])
    with pytest.raises(ManifestIntegrityError, match="Expected exactly 9"):
        RunManifest.model_validate(bad_streams)


def test_confirmatory_manifest_rejects_dirty_tree_and_bad_seed(tmp_path: Path) -> None:
    """Verify strict confirmatory integrity blocks dirty trees and unapproved seeds."""
    dev_manifest = _sample_dev_manifest(tmp_path)
    raw = dev_manifest.model_dump(mode="json")

    # Promoting a dirty manifest to confirmatory must raise ConfirmatoryIntegrityError
    dirty_conf = dict(raw)
    dirty_conf["identity"] = dict(raw["identity"], protocol_stage="confirmatory")
    body = {k: v for k, v in dirty_conf.items() if k != "manifest_sha256"}
    from jssp_research.experiments.manifests import canonical_json_sha256

    dirty_conf["manifest_sha256"] = canonical_json_sha256(body)
    with pytest.raises(ConfirmatoryIntegrityError, match="forbid a dirty Git working tree"):
        RunManifest.model_validate(dirty_conf)

    # Even with a simulated clean tree, non-frozen master seed must be rejected
    bad_seed_conf = dict(dirty_conf)
    bad_seed_conf["source_code"] = dict(
        raw["source_code"],
        git={
            "commit": "36564e2aacd312218b56c7413181c55a10ff14a8",
            "branch": "main",
            "dirty": False,
            "diff_sha256": None,
        },
    )
    bad_seed_conf["randomness"] = dict(raw["randomness"], master_seed=999999)
    body2 = {k: v for k, v in bad_seed_conf.items() if k != "manifest_sha256"}
    bad_seed_conf["manifest_sha256"] = canonical_json_sha256(body2)
    with pytest.raises(ConfirmatoryIntegrityError, match="does not match frozen schedule value"):
        RunManifest.model_validate(bad_seed_conf)

    # Clean tree + frozen seed + valid locks/hashes succeeds
    clean_conf = dict(bad_seed_conf)
    clean_conf["randomness"] = dict(raw["randomness"])
    body3 = {k: v for k, v in clean_conf.items() if k != "manifest_sha256"}
    clean_conf["manifest_sha256"] = canonical_json_sha256(body3)
    validated = RunManifest.model_validate(clean_conf)
    assert validated.identity.protocol_stage == ProtocolStage.CONFIRMATORY


def test_separate_result_records_and_manifest_cross_validation(tmp_path: Path) -> None:
    """Verify separate result records, retry attempts, and manifest cross-referencing."""
    manifest = _sample_dev_manifest(tmp_path, eval_budget=100)
    manifest_file = write_manifest(manifest, tmp_path / "manifests" / "run.manifest.json")

    attempt1 = create_run_result_record(
        manifest=manifest,
        attempt_index=1,
        status=RunStatus.FAILED,
        elapsed_seconds=0.05,
        objective_evaluations_used=5,
        error_info={
            "error_type": "SimulatedAbort",
            "message": "transient abort",
            "stage": "test",
        },
    )
    res1_path = write_result_record(attempt1, tmp_path / "results")
    assert load_and_validate_result(res1_path, manifest=manifest) == attempt1

    attempt2 = create_run_result_record(
        manifest=manifest,
        attempt_index=2,
        status=RunStatus.COMPLETED,
        elapsed_seconds=0.12,
        objective_evaluations_used=42,
        makespan=55,
        feasible=True,
        terminated_reason="local_optimum",
    )
    res2_path = write_result_record(attempt2, tmp_path / "results")
    assert res1_path != res2_path
    assert load_and_validate_result(res2_path, manifest=manifest) == attempt2

    with pytest.raises(FileExistsError):
        write_result_record(attempt2, res2_path)

    # Original manifest file remains untouched on disk
    assert load_and_validate_manifest(manifest_file) == manifest

    # Mismatched manifest link raises ResultValidationError
    other_manifest = _sample_dev_manifest(tmp_path, eval_budget=20)
    with pytest.raises(ResultValidationError, match="mismatch"):
        load_and_validate_result(res2_path, manifest=other_manifest)


def test_e02_deep_manifest_immutability_prevents_nested_mutation(tmp_path: Path) -> None:
    """Verify nested snapshots, lists, QMC metadata, budgets, and streams are unmodifiable."""
    ft06_entry = get_catalog_entry("ft06", repo_root=REPO_ROOT)
    parsed = parse_jssp_file(
        REPO_ROOT / "tests" / "fixtures" / "medium_jssp.txt", instance_id="ft06"
    )
    seed_ctx = build_seed_context(
        replicate_index=0,
        instance_id="ft06",
        pairing_mode=PairingMode.PAIRED_INIT,
        pairing_group="phase_a_shared",
        configuration_id="A09",
    )

    caller_snapshot: dict[str, object] = {
        "config_id": "A09",
        "nested_dict": {"alpha": 1.0, "deep_list": [10, {"sub": "ok"}]},
    }
    caller_qmc: dict[str, object] = {"scramble": "digit_perm", "indices": [1, 2, 3]}
    caller_budgets = {"local_search": 50}

    manifest = create_run_manifest(
        run_id="run-deep-freeze-01",
        experiment_id="m3.1-e02",
        phase=ExperimentPhase.PHASE_A,
        protocol_stage=ProtocolStage.DEVELOPMENT,
        configuration_id="A09",
        algorithm_id="scrambled_halton_mwr",
        algorithm_version="0.1.0",
        neighborhood_id=None,
        configuration_snapshot=caller_snapshot,  # type: ignore[arg-type]
        catalog_entry=ft06_entry,
        parsed_instance=parsed,
        seed_context=seed_ctx,
        budget=BudgetSpec(
            objective_evaluation_budget=64,
            algorithm_specific_budgets=caller_budgets,
        ),
        repo_root=REPO_ROOT,
        qmc_scramble_metadata=caller_qmc,  # type: ignore[arg-type]
    )

    # Mutating original caller objects must not leak into manifest
    caller_snapshot["config_id"] = "TAMPERED"
    nested = caller_snapshot["nested_dict"]
    assert isinstance(nested, dict)
    nested["alpha"] = 999.0
    deep_list = nested["deep_list"]
    assert isinstance(deep_list, list)
    deep_list.append(999)
    caller_qmc["scramble"] = "MUTATED"
    caller_budgets["local_search"] = 9999

    assert manifest.algorithm.configuration_snapshot["config_id"] == "A09"
    assert manifest.algorithm.configuration_snapshot["nested_dict"]["alpha"] == 1.0
    assert manifest.randomness.qmc_scramble_metadata is not None
    assert manifest.randomness.qmc_scramble_metadata["scramble"] == "digit_perm"
    assert manifest.budget.algorithm_specific_budgets["local_search"] == 50

    # Attempting in-place mutation on nested containers must raise TypeError or ValidationError.
    with pytest.raises(TypeError):
        manifest.algorithm.configuration_snapshot["injected"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        manifest.algorithm.configuration_snapshot["nested_dict"]["alpha"] = 2.0  # type: ignore[index]
    assert isinstance(manifest.algorithm.configuration_snapshot["nested_dict"]["deep_list"], tuple)
    with pytest.raises(TypeError):
        manifest.algorithm.configuration_snapshot["nested_dict"]["deep_list"][1]["sub"] = "bad"  # type: ignore[index]
    with pytest.raises(TypeError):
        manifest.randomness.qmc_scramble_metadata["scramble"] = "bad"  # type: ignore[index]
    with pytest.raises(TypeError):
        manifest.budget.algorithm_specific_budgets["local_search"] = 1  # type: ignore[index]
    with pytest.raises(ValidationError):
        manifest.randomness.streams[0].derived_seed_sha256 = "0" * 64  # type: ignore[misc]
    with pytest.raises(ValidationError):
        manifest.environment.cpu_model = "fake_cpu"  # type: ignore[misc]

    manifest.verify_integrity()


def test_e03_canonical_serialization_contract_and_compat(tmp_path: Path) -> None:
    """Verify canonical JSON reference vector, edge-case rules, and legacy compatibility."""
    vec = {
        "z": [{"b": 2, "a": 1}, -0.0, 0.0],
        "a": {
            "u64_max": (1 << 64) - 1,
            "above_js_safe": (1 << 53) + 7,
            "neg_i64": -(1 << 63),
        },
        "text": 'line1\ncafé"q"',
        "flag": True,
    }
    expected_bytes = (
        b'{"a":{"above_js_safe":9007199254740999,"neg_i64":-9223372036854775808,'
        b'"u64_max":18446744073709551615},"flag":true,'
        b'"text":"line1\\ncaf\\u00e9\\"q\\"","z":[{"a":1,"b":2},0.0,0.0]}'
    )
    assert canonical_json_bytes(vec) == expected_bytes
    assert (
        canonical_json_sha256(vec)
        == "63d986f4a1bdf93f903c8346c9f0cf87a12002f4c3c71c1c60e9fac4daa0380d"
    )
    assert canonical_json_sha256({"x": -0.0}) == canonical_json_sha256({"x": 0.0})

    for bad_val in (float("nan"), float("inf"), float("-inf"), "\ud800", 1 << 64):
        with pytest.raises(ManifestIntegrityError):
            canonical_json_bytes({"bad": bad_val})

    with pytest.raises(ManifestIntegrityError, match="Unsupported canonical serialization version"):
        canonical_json_bytes({"ok": 1}, serialization_version="jssp-canonical-json-v99")

    # Verify M3.1 manifests carry canonical_serialization_version and legacy pre-M3.1 compatibility
    manifest = _sample_dev_manifest(tmp_path)
    assert manifest.canonical_serialization_version == CANONICAL_SERIALIZATION_VERSION

    # Simulate legacy pre-M3.1 payload omitting top-level canonical_serialization_version
    legacy_payload = manifest.model_dump(
        mode="json", exclude={"canonical_serialization_version", "manifest_sha256"}
    )
    legacy_payload["manifest_sha256"] = canonical_json_sha256(legacy_payload)
    reloaded_legacy = RunManifest.model_validate(legacy_payload)
    assert reloaded_legacy.canonical_serialization_version is None
    assert reloaded_legacy.manifest_sha256 == legacy_payload["manifest_sha256"]

    # Unknown top-level canonical_serialization_version must fail cleanly
    bad_ser = manifest.model_dump(mode="json")
    bad_ser["canonical_serialization_version"] = "jssp-canonical-json-v99"
    with pytest.raises(ManifestIntegrityError, match="Unsupported canonical serialization version"):
        RunManifest.model_validate(bad_ser)
