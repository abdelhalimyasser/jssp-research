"""End-to-end M3 integration tests linking parsers, Rust validation/RNG, manifests, and CLI."""

from __future__ import annotations

import json
from pathlib import Path

from jssp_research.benchmarks import get_catalog_entry, parse_jssp_file
from jssp_research.cli import main as cli_main
from jssp_research.experiments import (
    BudgetSpec,
    ExperimentPhase,
    PairingMode,
    ProtocolStage,
    build_seed_context,
    create_run_manifest,
    load_and_validate_manifest,
    resolve_named_streams,
    write_manifest,
)
from jssp_research.native import decode_obr, rng_sample_range_and_shuffle, validate_schedule
from jssp_research.results import (
    RunStatus,
    create_run_result_record,
    load_and_validate_result,
    write_result_record,
)
from jssp_research.utils.provenance import collect_git_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_end_to_end_reproducible_seed_parse_manifest_and_result(tmp_path: Path) -> None:
    """Link benchmark parser, Rust instance/decoder/RNG, immutable manifest, and result record."""
    ft06_entry = get_catalog_entry("ft06", repo_root=REPO_ROOT)
    assert ft06_entry.local_relative_path is not None
    parsed = parse_jssp_file(REPO_ROOT / ft06_entry.local_relative_path, instance_id="ft06")
    assert parsed.canonical_sha256 == ft06_entry.canonical_sha256

    seed_ctx = build_seed_context(
        replicate_index=1,
        instance_id="ft06",
        pairing_mode=PairingMode.PAIRED_INIT,
        pairing_group="m3_integration",
        configuration_id="C01",
    )
    streams = {s.role: s for s in resolve_named_streams(seed_ctx)}

    # Build a deterministic shuffled OBR chromosome using Rust's ChaCha20 `init` stream
    canonical_genes = [j for j in range(parsed.num_jobs) for _ in range(parsed.num_machines)]
    _, perm = rng_sample_range_and_shuffle(
        streams["init"].seed_hex, parsed.num_jobs, 0, len(canonical_genes)
    )
    shuffled_obr = [canonical_genes[idx] for idx in perm]

    schedule = decode_obr(parsed.jobs, shuffled_obr)
    assert validate_schedule(parsed.jobs, schedule) is True

    # Repeat from the exact same seed context and confirm identical decoded schedule
    streams_repeat = {s.role: s for s in resolve_named_streams(seed_ctx)}
    _, perm_repeat = rng_sample_range_and_shuffle(
        streams_repeat["init"].seed_hex, parsed.num_jobs, 0, len(canonical_genes)
    )
    assert perm == perm_repeat
    assert decode_obr(parsed.jobs, [canonical_genes[i] for i in perm_repeat]) == schedule

    manifest = create_run_manifest(
        run_id="e2e-ft06-r01",
        experiment_id="m3-e2e",
        phase=ExperimentPhase.PILOT,
        protocol_stage=ProtocolStage.DEVELOPMENT,
        configuration_id="C01",
        algorithm_id="m3_obr_smoke",
        algorithm_version="0.1.0",
        neighborhood_id=None,
        configuration_snapshot={"seed_role": "init", "decode": "serial_sgs_earliest_gap"},
        catalog_entry=ft06_entry,
        parsed_instance=parsed,
        seed_context=seed_ctx,
        budget=BudgetSpec(objective_evaluation_budget=10),
        repo_root=REPO_ROOT,
    )
    manifest_path = write_manifest(manifest, tmp_path / "e2e.manifest.json")

    result = create_run_result_record(
        manifest=manifest,
        attempt_index=1,
        status=RunStatus.COMPLETED,
        elapsed_seconds=0.01,
        objective_evaluations_used=1,
        makespan=schedule.makespan,
        feasible=True,
        terminated_reason="single_decode_smoke",
    )
    result_path = write_result_record(result, tmp_path)

    reloaded_manifest = load_and_validate_manifest(manifest_path)
    reloaded_result = load_and_validate_result(result_path, manifest=reloaded_manifest)
    assert reloaded_result.objective_results.makespan == schedule.makespan
    assert reloaded_manifest.instance.canonical_parsed_sha256 == parsed.canonical_sha256


def test_cli_workflows_end_to_end(tmp_path: Path, capsys: object) -> None:
    """Exercise all M3 CLI subcommands against real repository fixtures and manifests."""
    assert cli_main(["--repo-root", str(REPO_ROOT), "benchmarks", "families"]) == 0
    assert cli_main(["--repo-root", str(REPO_ROOT), "benchmarks", "list", "--family", "FT"]) == 0
    assert cli_main(["--repo-root", str(REPO_ROOT), "benchmarks", "inspect", "ft06"]) == 0
    assert cli_main(["--repo-root", str(REPO_ROOT), "benchmarks", "validate"]) == 0
    assert (
        cli_main(
            [
                "--repo-root",
                str(REPO_ROOT),
                "benchmarks",
                "checksum",
                str(REPO_ROOT / "tests" / "fixtures" / "tiny_jssp.txt"),
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "--repo-root",
                str(REPO_ROOT),
                "rng",
                "inspect",
                "--replicate",
                "0",
                "--instance",
                "ft06",
            ]
        )
        == 0
    )

    manifest_out = tmp_path / "cli_run.manifest.json"
    assert (
        cli_main(
            [
                "--repo-root",
                str(REPO_ROOT),
                "manifest",
                "create",
                "--run-id",
                "cli-run-01",
                "--experiment-id",
                "cli-test",
                "--phase",
                "pilot",
                "--stage",
                "development",
                "--config-id",
                "A01",
                "--algorithm-id",
                "uniform_random",
                "--instance-id",
                "ft06",
                "--replicate",
                "0",
                "--eval-budget",
                "64",
                "--output",
                str(manifest_out),
            ]
        )
        == 0
    )
    assert cli_main(["--repo-root", str(REPO_ROOT), "manifest", "validate", str(manifest_out)]) == 0

    # Confirmatory manifests are rejected only when the working tree is not clean.
    git_provenance = collect_git_provenance(REPO_ROOT)
    clean_working_tree = git_provenance.dirty is False and git_provenance.diff_sha256 is None
    bad_conf = tmp_path / "bad_conf.manifest.json"
    rc = cli_main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "manifest",
            "create",
            "--run-id",
            "cli-conf-bad",
            "--experiment-id",
            "cli-test",
            "--phase",
            "phase_a",
            "--stage",
            "confirmatory",
            "--config-id",
            "A01",
            "--algorithm-id",
            "uniform_random",
            "--instance-id",
            "ft06",
            "--replicate",
            "0",
            "--eval-budget",
            "64",
            "--output",
            str(bad_conf),
        ]
    )
    assert rc == (0 if clean_working_tree else 1)
    assert bad_conf.exists() is clean_working_tree

    # Check output validity of at least one JSON payload (last valid command was manifest validate)
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "valid" in captured.out
    assert json.loads(manifest_out.read_text("utf-8"))["identity"]["run_id"] == "cli-run-01"
