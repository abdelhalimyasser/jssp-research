"""Focused M3 CLI for benchmarks, checksums, RNG inspection, and run manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from jssp_research.benchmarks import (
    DEFAULT_REPO_ROOT,
    BenchmarkTrack,
    SplitAssignment,
    family_summary,
    get_catalog_entry,
    list_catalog_entries,
    parse_jssp_file,
    sha256_file,
    validate_catalog,
    verify_sha256,
)
from jssp_research.experiments import (
    BudgetSpec,
    ExperimentPhase,
    PairingMode,
    ProtocolStage,
    build_seed_context,
    create_run_manifest,
    load_and_validate_manifest,
    resolve_named_streams,
    rust_rng_metadata,
    write_manifest,
)
from jssp_research.results import load_and_validate_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jssp",
        description="Classical JSSP Research Framework M3 Reproducibility & Catalog CLI.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root directory for catalog/provenance resolution.",
    )
    sub = parser.add_subparsers(dest="group", required=True)

    # Benchmarks group
    bench = sub.add_parser("benchmarks", help="Inspect and validate benchmark catalog and files.")
    bench_sub = bench.add_subparsers(dest="command", required=True)

    bench_sub.add_parser("families", help="List cataloged benchmark families by track.")

    b_list = bench_sub.add_parser("list", help="List catalog instances matching optional filters.")
    b_list.add_argument("--track", choices=[t.value for t in BenchmarkTrack])
    b_list.add_argument("--family", type=str)
    b_list.add_argument("--split", choices=[s.value for s in SplitAssignment])

    b_inspect = bench_sub.add_parser("inspect", help="Inspect metadata for one catalog instance.")
    b_inspect.add_argument("instance_id", type=str)

    bench_sub.add_parser("validate", help="Validate all catalog entries and local fixtures.")

    b_csum = bench_sub.add_parser(
        "checksum", help="Compute raw SHA-256 and canonical parsed SHA-256 for an instance file."
    )
    b_csum.add_argument("path", type=Path)
    b_csum.add_argument("--format", dest="format_id", default="standard_jssp_0based_v1")
    b_csum.add_argument("--instance-id", type=str, default=None)
    b_csum.add_argument("--verify", type=str, default=None, help="Expected raw SHA-256 hex digest.")

    # Manifests group
    man = sub.add_parser("manifest", help="Create or validate immutable run manifests.")
    man_sub = man.add_subparsers(dest="command", required=True)

    m_create = man_sub.add_parser("create", help="Create and persist an immutable run manifest.")
    m_create.add_argument("--run-id", required=True)
    m_create.add_argument("--experiment-id", required=True)
    m_create.add_argument("--phase", required=True, choices=[p.value for p in ExperimentPhase])
    m_create.add_argument("--stage", required=True, choices=[s.value for s in ProtocolStage])
    m_create.add_argument("--config-id", required=True)
    m_create.add_argument("--algorithm-id", required=True)
    m_create.add_argument("--algorithm-version", default="0.1.0")
    m_create.add_argument("--neighborhood-id", default=None)
    m_create.add_argument("--instance-id", required=True)
    m_create.add_argument("--instance-file", type=Path, default=None)
    m_create.add_argument("--format", dest="format_id", default=None)
    m_create.add_argument("--replicate", required=True, type=int)
    m_create.add_argument("--master-seed", type=int, default=None)
    m_create.add_argument(
        "--pairing-mode",
        default=PairingMode.PAIRED_INIT.value,
        choices=[m.value for m in PairingMode] + ["crn"],
    )
    m_create.add_argument("--pairing-group", default="default_group")
    m_create.add_argument("--eval-budget", required=True, type=int)
    m_create.add_argument("--output", required=True, type=Path)

    m_val = man_sub.add_parser("validate", help="Validate an existing manifest (and result).")
    m_val.add_argument("manifest_path", type=Path)
    m_val.add_argument("--result", type=Path, default=None)

    # RNG group
    rng = sub.add_parser("rng", help="Inspect deterministic named RNG streams.")
    rng_sub = rng.add_subparsers(dest="command", required=True)

    r_insp = rng_sub.add_parser("inspect", help="Derive and display D07-A stream seed hashes.")
    r_insp.add_argument("--replicate", required=True, type=int)
    r_insp.add_argument("--instance", required=True)
    r_insp.add_argument("--master-seed", type=int, default=None)
    r_insp.add_argument(
        "--pairing-mode",
        default=PairingMode.PAIRED_INIT.value,
        choices=[m.value for m in PairingMode] + ["crn"],
    )
    r_insp.add_argument("--pairing-group", default="default_group")
    r_insp.add_argument("--config-id", default="default_config")
    r_insp.add_argument("--preview-draws", type=int, default=2)

    return parser


def _handle_benchmarks(args: argparse.Namespace) -> int:
    repo_root: Path = args.repo_root
    if args.command == "families":
        rows = [
            {
                "track": r.track.value,
                "family": r.family,
                "total": r.total,
                "available_locally": r.available_locally,
                "development": r.development,
                "confirmatory_or_held_out": r.confirmatory_or_held_out,
                "deferred": r.deferred,
            }
            for r in family_summary(repo_root=repo_root)
        ]
        print(json.dumps(rows, indent=2))
        return 0

    if args.command == "list":
        track = BenchmarkTrack(args.track) if args.track else None
        split = SplitAssignment(args.split) if args.split else None
        entries = list_catalog_entries(
            track=track, family=args.family, split=split, repo_root=repo_root
        )
        print(json.dumps([e.model_dump(mode="json") for e in entries], indent=2))
        return 0

    if args.command == "inspect":
        entry = get_catalog_entry(args.instance_id, repo_root=repo_root)
        print(json.dumps(entry.model_dump(mode="json"), indent=2))
        return 0

    if args.command == "validate":
        report = validate_catalog(repo_root=repo_root)
        payload = {
            "valid": report.is_valid,
            "total_entries": report.total_entries,
            "metadata_entries_validated": report.metadata_entries_validated,
            "track_counts": report.track_counts,
            "local_files_parsed_and_verified": report.local_files_parsed_and_verified,
            "pending_file_acquisition_entries": report.pending_file_acquisition_entries,
            "development_entries": report.development_entries,
            "confirmatory_entries": report.confirmatory_entries,
            "deferred_entries": report.deferred_entries,
            "errors": list(report.errors),
        }
        print(json.dumps(payload, indent=2))
        return 0 if report.is_valid else 1

    if args.command == "checksum":
        file_path: Path = args.path
        parsed = parse_jssp_file(
            file_path,
            instance_id=args.instance_id,
            format_id=args.format_id,
        )
        raw_digest = sha256_file(file_path)
        if args.verify is not None:
            verify_sha256(raw_digest, args.verify, context=str(file_path))
        print(
            json.dumps(
                {
                    "path": str(file_path),
                    "instance_id": parsed.instance_id,
                    "format_id": parsed.format_id,
                    "parser_version": parsed.parser_version,
                    "dimensions": [parsed.num_jobs, parsed.num_machines],
                    "raw_sha256": raw_digest,
                    "canonical_sha256": parsed.canonical_sha256,
                },
                indent=2,
            )
        )
        return 0

    raise ValueError(f"Unknown benchmarks subcommand: {args.command}")


def _handle_manifest(args: argparse.Namespace) -> int:
    repo_root: Path = args.repo_root
    if args.command == "create":
        entry = get_catalog_entry(args.instance_id, repo_root=repo_root)
        if args.instance_file is not None:
            inst_path = args.instance_file
        elif entry.local_relative_path is not None:
            inst_path = repo_root / entry.local_relative_path
        else:
            raise FileNotFoundError(
                f"Instance {entry.instance_id!r} has no local file; supply --instance-file"
            )

        fmt = args.format_id or entry.format_id or "standard_jssp_0based_v1"
        parsed = parse_jssp_file(inst_path, instance_id=entry.instance_id, format_id=fmt)
        seed_ctx = build_seed_context(
            replicate_index=args.replicate,
            instance_id=entry.instance_id,
            pairing_mode=args.pairing_mode,
            pairing_group=args.pairing_group,
            configuration_id=args.config_id,
            explicit_master_seed=args.master_seed,
        )
        manifest = create_run_manifest(
            run_id=args.run_id,
            experiment_id=args.experiment_id,
            phase=args.phase,
            protocol_stage=args.stage,
            configuration_id=args.config_id,
            algorithm_id=args.algorithm_id,
            algorithm_version=args.algorithm_version,
            neighborhood_id=args.neighborhood_id,
            configuration_snapshot={"config_id": args.config_id},
            catalog_entry=entry,
            parsed_instance=parsed,
            seed_context=seed_ctx,
            budget=BudgetSpec(objective_evaluation_budget=args.eval_budget),
            repo_root=repo_root,
        )
        out_path = write_manifest(manifest, args.output)
        print(
            json.dumps(
                {
                    "manifest_path": str(out_path),
                    "manifest_sha256": manifest.manifest_sha256,
                    "protocol_stage": manifest.identity.protocol_stage.value,
                    "git_dirty": manifest.source_code.git.dirty,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "validate":
        manifest = load_and_validate_manifest(args.manifest_path)
        result_sha: str | None = None
        if args.result is not None:
            result = load_and_validate_result(args.result, manifest=manifest)
            result_sha = result.result_sha256
        print(
            json.dumps(
                {
                    "valid": True,
                    "run_id": manifest.identity.run_id,
                    "manifest_sha256": manifest.manifest_sha256,
                    "result_sha256": result_sha,
                },
                indent=2,
            )
        )
        return 0

    raise ValueError(f"Unknown manifest subcommand: {args.command}")


def _handle_rng(args: argparse.Namespace) -> int:
    if args.command != "inspect":
        raise ValueError(f"Unknown rng subcommand: {args.command}")
    ctx = build_seed_context(
        replicate_index=args.replicate,
        instance_id=args.instance,
        pairing_mode=args.pairing_mode,
        pairing_group=args.pairing_group,
        configuration_id=args.config_id,
        explicit_master_seed=args.master_seed,
    )
    meta = rust_rng_metadata()
    streams = resolve_named_streams(ctx, preview_count=args.preview_draws)
    payload = {
        "algorithm": meta.algorithm,
        "rng_crate_version": meta.rng_crate_version,
        "hash_crate_version": meta.hash_crate_version,
        "chacha_rounds": meta.chacha_rounds,
        "seed_derivation_schema": meta.seed_derivation_schema,
        "master_seed_schedule_id": meta.master_seed_schedule_id,
        "context": ctx.model_dump(mode="json"),
        "streams": [
            {
                "role": s.role,
                "coupling_key": s.coupling_key,
                "seed_hex": s.seed_hex,
                "preview_draws_u64": [hex(w) for w in s.preview_draws_u64],
            }
            for s in streams
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for the `jssp` command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.group == "benchmarks":
            return _handle_benchmarks(args)
        if args.group == "manifest":
            return _handle_manifest(args)
        if args.group == "rng":
            return _handle_rng(args)
        parser.error(f"Unknown command group: {args.group}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
