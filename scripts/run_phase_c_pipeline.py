"""Execute Phase C (C01-C11), freeze winner C*, and generate analysis/reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from jssp_research.analysis import (
    lower_bound_gap,
    paired_comparisons_table,
    relative_percentage_deviation,
    run_friedman_analysis,
    summarize_numeric,
)
from jssp_research.benchmarks import parse_jssp_file
from jssp_research.config import load_phase_c_matrix
from jssp_research.experiments.manifests import canonical_json_sha256
from jssp_research.native import (
    PhaseCConfigSpec,
    default_phase_c_budget,
    get_frozen_master_seeds,
    get_rng_metadata,
    list_phase_c_configs,
    run_phase_c_batch,
)
from jssp_research.utils.provenance import (
    collect_environment_provenance,
    collect_git_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_JSON = REPO_ROOT / "data/metadata/selected_benchmark_suite.json"
FROZEN_A_JSON = REPO_ROOT / "results/phase_a/winner/frozen_winner_a.json"
FROZEN_B_JSON = REPO_ROOT / "results/phase_b/winner/frozen_winner_b.json"
PHASE_C_ROOT = REPO_ROOT / "results/phase_c"
REPORTS_ROOT = REPO_ROOT / "reports"
NUM_REPLICATES = 20


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _phase_c_record_to_row(
    inst: dict[str, Any],
    spec: PhaseCConfigSpec,
    frozen_a_id: str,
    frozen_b_id: str,
    replicate: int,
    seed: int,
    budget: int,
    rec: Any,
) -> dict[str, Any]:
    bks = int(inst["best_known_value"])
    lb = inst["lower_bound"]
    return {
        "instance_id": inst["instance_id"],
        "family": inst["benchmark_family"],
        "split": inst["split"],
        "jobs": inst["jobs"],
        "machines": inst["machines"],
        "operation_count": inst["operation_count"],
        "bks": bks,
        "lb": lb,
        "initializer_id": frozen_a_id,
        "frozen_b_id": frozen_b_id,
        "config_id": spec.code,
        "config_name": spec.name,
        "is_memetic": spec.is_memetic,
        "is_dual_view_paired": spec.is_dual_view_paired,
        "ls_neighborhood": rec.ls_neighborhood,
        "ls_strategy": rec.ls_strategy,
        "genotype_capacity": rec.genotype_capacity,
        "replicate_index": replicate,
        "master_seed": seed,
        "evaluation_budget": budget,
        "evals_used": rec.evals_used,
        "decode_evals": rec.decode_evals,
        "ls_evals": rec.ls_evals,
        "generations_executed": rec.generations_executed,
        "best_makespan": rec.best_makespan,
        "best_rpd_bks": relative_percentage_deviation(rec.best_makespan, bks),
        "best_gap_lb": lower_bound_gap(rec.best_makespan, lb),
        "hit_bks": rec.best_makespan <= bks,
        "best_found_eval": rec.best_found_eval,
        "best_found_generation": rec.best_found_generation,
        "feasible_final": rec.feasible_final,
        "anytime_cmax_auc": rec.anytime_cmax_auc,
        "anytime_rpd_auc": relative_percentage_deviation(rec.anytime_cmax_auc, bks),
        "final_obr_diversity": rec.final_obr_diversity,
        "final_lam_diversity": rec.final_lam_diversity,
        "final_bald_diversity": rec.final_bald_diversity,
        "final_gate_p": rec.final_gate_p,
        "lam_best_makespan": rec.lam_best_makespan,
        "bald_best_makespan": rec.bald_best_makespan,
        "crossover_count": rec.crossover_count,
        "mutation_count": rec.mutation_count,
        "writeback_count": rec.writeback_count,
        "migrations_applied": rec.migrations_applied,
        "total_elapsed_ms": rec.total_elapsed_ns / 1e6,
    }


def _summarize_split(
    rows: list[dict[str, Any]],
    specs: tuple[PhaseCConfigSpec, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    instance_ids = sorted({r["instance_id"] for r in rows})
    by_cfg_inst: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        by_cfg_inst.setdefault((r["config_id"], r["instance_id"]), []).append(r)

    cfg_ids = [s.code for s in specs]
    spec_by_id = {s.code: s for s in specs}
    med_rpd_mat = np.zeros((len(instance_ids), len(cfg_ids)), dtype=float)
    scores_by_cfg: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []

    for c_idx, cid in enumerate(cfg_ids):
        spec = spec_by_id[cid]
        inst_med_rpds: list[float] = []
        inst_mean_rpds: list[float] = []
        inst_best_rpds: list[float] = []
        inst_med_cmax: list[float] = []
        inst_best_cmax: list[float] = []
        inst_std_cmax: list[float] = []
        inst_iqr_cmax: list[float] = []
        inst_hit_rate: list[float] = []
        inst_auc_rpd: list[float] = []
        inst_div: list[float] = []
        inst_lam_div: list[float] = []
        inst_bald_div: list[float] = []
        inst_gate_p: list[float] = []
        inst_gens: list[float] = []
        inst_ms: list[float] = []

        for i_idx, iid in enumerate(instance_ids):
            grp = by_cfg_inst[(cid, iid)]
            rpds = np.array([g["best_rpd_bks"] for g in grp], dtype=float)
            cmax = np.array([g["best_makespan"] for g in grp], dtype=float)
            sc = summarize_numeric(cmax)
            med_rpd = float(np.median(rpds))
            med_rpd_mat[i_idx, c_idx] = med_rpd

            inst_med_rpds.append(med_rpd)
            inst_mean_rpds.append(float(np.mean(rpds)))
            inst_best_rpds.append(float(np.min(rpds)))
            inst_med_cmax.append(sc["median"])
            inst_best_cmax.append(sc["min"])
            inst_std_cmax.append(sc["std"])
            inst_iqr_cmax.append(sc["iqr"])
            inst_hit_rate.append(float(np.mean([1.0 if g["hit_bks"] else 0.0 for g in grp])))
            inst_auc_rpd.append(float(np.median([g["anytime_rpd_auc"] for g in grp])))
            inst_div.append(float(np.mean([g["final_obr_diversity"] for g in grp])))
            inst_gens.append(float(np.mean([g["generations_executed"] for g in grp])))
            inst_ms.append(float(np.mean([g["total_elapsed_ms"] for g in grp])))

            lam_d = [g["final_lam_diversity"] for g in grp if g["final_lam_diversity"] is not None]
            bald_d = [
                g["final_bald_diversity"] for g in grp if g["final_bald_diversity"] is not None
            ]
            gp = [g["final_gate_p"] for g in grp if g["final_gate_p"] is not None]
            if lam_d:
                inst_lam_div.append(float(np.mean(lam_d)))
            if bald_d:
                inst_bald_div.append(float(np.mean(bald_d)))
            if gp:
                inst_gate_p.append(float(np.mean(gp)))

        scores_by_cfg[cid] = med_rpd_mat[:, c_idx].copy()
        summary_rows.append(
            {
                "config_id": cid,
                "config_name": spec.name,
                "is_memetic": spec.is_memetic,
                "is_dual_view_paired": spec.is_dual_view_paired,
                "n_instances": len(instance_ids),
                "n_runs": sum(len(by_cfg_inst[(cid, iid)]) for iid in instance_ids),
                "feasible_rate": 1.0,
                "best_rpd_mean": float(np.mean(inst_best_rpds)),
                "mean_rpd_mean": float(np.mean(inst_mean_rpds)),
                "median_rpd_mean": float(np.mean(inst_med_rpds)),
                "median_of_instance_medians_rpd": float(np.median(inst_med_rpds)),
                "best_cmax_mean": float(np.mean(inst_best_cmax)),
                "median_cmax_mean": float(np.mean(inst_med_cmax)),
                "within_instance_std_mean": float(np.mean(inst_std_cmax)),
                "within_instance_iqr_mean": float(np.mean(inst_iqr_cmax)),
                "anytime_rpd_auc_mean": float(np.mean(inst_auc_rpd)),
                "hit_bks_rate": float(np.mean(inst_hit_rate)),
                "final_obr_div_mean": float(np.mean(inst_div)),
                "final_lam_div_mean": float(np.mean(inst_lam_div)) if inst_lam_div else None,
                "final_bald_div_mean": float(np.mean(inst_bald_div)) if inst_bald_div else None,
                "final_gate_p_mean": float(np.mean(inst_gate_p)) if inst_gate_p else None,
                "mean_generations": float(np.mean(inst_gens)),
                "mean_elapsed_ms": float(np.mean(inst_ms)),
            }
        )

    friedman = run_friedman_analysis(cfg_ids, med_rpd_mat)
    for row in summary_rows:
        row["friedman_rank"] = friedman.mean_ranks[row["config_id"]]

    # Note: C11 is an exploratory ablation and excluded from primary base winner selection;
    # still rank all 11 in tables, and select C* strictly among primary candidates C01..C10!
    summary_rows.sort(
        key=lambda r: (
            r["median_rpd_mean"],
            r["friedman_rank"],
            r["anytime_rpd_auc_mean"],
            r["mean_elapsed_ms"],
        )
    )
    primary_rows = [r for r in summary_rows if r["config_id"] != "C11"]
    winner_id = primary_rows[0]["config_id"]

    # Research contrasts defined in docs/experiments/phase_c.md §4:
    pairs = [
        ("memetic_vs_classical", "C02", "C01"),
        ("memetic_vs_classical", "C03", "C01"),
        ("memetic_vs_classical", "C04", "C01"),
        ("memetic_vs_classical", "C05", "C01"),
        ("memetic_vs_classical", "C06", "C01"),
        ("dual_view_vs_single_memetic", "C05", "C02"),
        ("dual_view_vs_single_memetic", "C05", "C03"),
        ("dual_view_vs_single_memetic", "C05", "C04"),
        ("dual_view_vs_single_memetic", "C06", "C02"),
        ("dual_view_vs_single_memetic", "C06", "C03"),
        ("dual_view_vs_single_memetic", "C06", "C04"),
        ("adaptive_vs_fixed_gate", "C06", "C05"),
        ("adaptive_vs_fixed_gate", "C06", "C07"),
        ("adaptive_vs_fixed_gate", "C06", "C08"),
        ("diversity_ablation", "C06", "C09"),
        ("paired_vs_unpaired_control", "C05", "C10"),
        ("paired_vs_unpaired_control", "C06", "C10"),
        ("exploratory_migration", "C11", "C06"),
    ]
    wilcoxon_rows = paired_comparisons_table(scores_by_cfg, pairs, lower_is_better=True)

    stats_bundle = {
        "n_instances": len(instance_ids),
        "friedman_med_rpd": {
            "statistic": friedman.statistic,
            "p_value": friedman.p_value,
            "kendall_w": friedman.kendall_w,
            "mean_ranks": friedman.mean_ranks,
        },
        "paired_wilcoxon_holm": wilcoxon_rows,
        "selected_top_primary_id": winner_id,
    }
    return summary_rows, stats_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase C benchmark pipeline.")
    parser.add_argument("--max-ls-evals", type=int, default=50)
    parser.add_argument("--replicates", type=int, default=NUM_REPLICATES)
    args = parser.parse_args()

    started_utc = datetime.now(tz=UTC).isoformat()
    frozen_a = json.loads(FROZEN_A_JSON.read_text("utf-8"))
    frozen_b = json.loads(FROZEN_B_JSON.read_text("utf-8"))
    frozen_a_id = str(frozen_a["winner_config_id"])
    frozen_b_id = str(frozen_b["winner_config_id"])
    frozen_b_nb = str(frozen_b["winner_neighborhood"])
    frozen_b_strat = str(frozen_b["winner_strategy"])

    suite_doc = json.loads(SUITE_JSON.read_text("utf-8"))
    common_instances = [r for r in suite_doc["instances"] if r["in_common_pipeline_suite"]]

    specs = list_phase_c_configs()
    matrix_cfg = load_phase_c_matrix()
    master_seeds = tuple(int(s) for s in get_frozen_master_seeds())
    rng_meta = get_rng_metadata()

    for sub in ("raw", "manifests", "analysis", "winner"):
        (PHASE_C_ROOT / sub).mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting Phase C across {len(common_instances)} Common Suite instances x "
        f"{len(specs)} configs x {args.replicates} frozen seeds "
        f"(A*={frozen_a_id}, B*={frozen_b_id} [{frozen_b_strat} + {frozen_b_nb}], "
        f"b_LS={args.max_ls_evals})...",
        flush=True,
    )

    tasks: list[tuple[str, int, int | None, str, str, str]] = []
    task_meta: list[tuple[PhaseCConfigSpec, int]] = []
    for spec in specs:
        for rep in range(args.replicates):
            tasks.append((spec.code, rep, None, "paired_init", "phase_c_shared_init", spec.code))
            task_meta.append((spec, rep))

    raw_path = PHASE_C_ROOT / "raw" / "phase_c_common_32.jsonl"
    all_rows: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as fh:
        for idx, inst in enumerate(common_instances, start=1):
            t0 = time.perf_counter()
            parsed = parse_jssp_file(
                REPO_ROOT / inst["raw_relative_path"], instance_id=inst["instance_id"]
            )
            budget = default_phase_c_budget(parsed.num_jobs, parsed.num_machines)
            res_list = run_phase_c_batch(
                parsed.jobs,
                inst["instance_id"],
                tasks,
                initializer_id=frozen_a_id,
                ls_neighborhood=frozen_b_nb,
                ls_strategy=frozen_b_strat,
                max_ls_evals=args.max_ls_evals,
                evaluation_budget=budget,
                num_threads=0,
            )
            dt = time.perf_counter() - t0
            for (spec, rep), rec in zip(task_meta, res_list, strict=True):
                if not rec.feasible_final:
                    raise RuntimeError(
                        f"Infeasible Phase C result: {inst['instance_id']} {spec.code}"
                    )
                row = _phase_c_record_to_row(
                    inst, spec, frozen_a_id, frozen_b_id, rep, master_seeds[rep], budget, rec
                )
                all_rows.append(row)
                raw_row = dict(row)
                raw_row["gate_p_trajectory"] = [[g, p] for g, p in rec.gate_p_trajectory]
                fh.write(json.dumps(raw_row) + "\n")

            best_row = min(all_rows[-len(tasks) :], key=lambda r: r["best_makespan"])
            print(
                f"  [Phase C {idx:2d}/{len(common_instances)}] {inst['instance_id']:6s} "
                f"({inst['split']:8s}, {inst['jobs']:2d}x{inst['machines']:2d}, B={budget:5d}): "
                f"{len(tasks)} runs in {dt:6.2f}s | best Cmax={best_row['best_makespan']} "
                f"(BKS={inst['best_known_value']}, {best_row['config_id']})",
                flush=True,
            )

    dev_rows = [r for r in all_rows if r["split"] == "development"]
    ho_rows = [r for r in all_rows if r["split"] == "held_out"]

    dev_summary, dev_stats = _summarize_split(dev_rows, specs)
    ho_summary, ho_stats = _summarize_split(ho_rows, specs)

    _write_csv(PHASE_C_ROOT / "analysis" / "phase_c_dev_rankings.csv", dev_summary)
    _write_json(PHASE_C_ROOT / "analysis" / "phase_c_dev_rankings.json", dev_summary)
    _write_csv(PHASE_C_ROOT / "analysis" / "phase_c_held_out_evaluation.csv", ho_summary)
    _write_json(PHASE_C_ROOT / "analysis" / "phase_c_held_out_evaluation.json", ho_summary)

    stats_payload = {
        "schema_version": "jssp-phase-c-stats-v1",
        "development_16": dev_stats,
        "held_out_16": ho_stats,
    }
    _write_json(PHASE_C_ROOT / "analysis" / "phase_c_statistical_tests.json", stats_payload)
    _write_csv(
        PHASE_C_ROOT / "analysis" / "phase_c_statistical_tests.csv",
        [dict(r, split="development") for r in dev_stats["paired_wilcoxon_holm"]]
        + [dict(r, split="held_out") for r in ho_stats["paired_wilcoxon_holm"]],
    )

    # Pre-registered Winner Selection strictly on Development Split across primary C01..C10
    winner_id = str(dev_stats["selected_top_primary_id"])
    dev_winner_row = next(r for r in dev_summary if r["config_id"] == winner_id)
    ho_winner_row = next(r for r in ho_summary if r["config_id"] == winner_id)
    winner_matrix_cfg = next(c for c in matrix_cfg.configs if c.config_id == winner_id)

    winner_body: dict[str, Any] = {
        "schema_version": "jssp-frozen-winner-c-v1",
        "frozen_at_utc": datetime.now(tz=UTC).isoformat(),
        "transferred_frozen_winner_a": {
            "config_id": frozen_a_id,
            "winner_lock_sha256": frozen_a["winner_lock_sha256"],
        },
        "transferred_frozen_winner_b": {
            "config_id": frozen_b_id,
            "neighborhood": frozen_b_nb,
            "strategy": frozen_b_strat,
            "winner_lock_sha256": frozen_b["winner_lock_sha256"],
        },
        "selection_protocol": (
            "Selected strictly on the 16-instance Development Split across primary C01..C10 "
            "(excluding exploratory C11 from base claims) under equalized B_GA budget, lowest "
            "mean instance-level median RPD_BKS -> Friedman rank -> anytime RPD AUC -> runtime."
        ),
        "winner_config_id": winner_id,
        "winner_config_name": dev_winner_row["config_name"],
        "winner_parameters": winner_matrix_cfg.model_dump(mode="json"),
        "development_metrics": dev_winner_row,
        "held_out_validation_metrics": ho_winner_row,
    }
    winner_body["winner_lock_sha256"] = canonical_json_sha256(winner_body)
    _write_json(PHASE_C_ROOT / "winner" / "frozen_winner_c.json", winner_body)

    git_prov = collect_git_provenance(REPO_ROOT)
    env_prov = collect_environment_provenance(REPO_ROOT)
    manifest_doc = {
        "schema_version": "jssp-phase-c-manifest-v1",
        "phase": "phase_c",
        "started_utc": started_utc,
        "completed_utc": datetime.now(tz=UTC).isoformat(),
        "git": git_prov.model_dump(mode="json"),
        "environment": env_prov.model_dump(mode="json"),
        "rng": {
            "algorithm": rng_meta.algorithm,
            "seed_schedule_id": rng_meta.master_seed_schedule_id,
            "master_seeds_v1": list(master_seeds),
            "pairing_mode": "paired_init",
            "pairing_group": "phase_c_shared_init",
        },
        "frozen_winner_a_id": frozen_a_id,
        "frozen_winner_b_id": frozen_b_id,
        "frozen_b_neighborhood": frozen_b_nb,
        "frozen_b_strategy": frozen_b_strat,
        "max_ls_evals": args.max_ls_evals,
        "total_runs": len(all_rows),
        "selected_benchmark_suite_sha256": _sha256_file(SUITE_JSON),
        "raw_jsonl_sha256": _sha256_file(raw_path),
        "frozen_winner_c_sha256": winner_body["winner_lock_sha256"],
    }
    _write_json(PHASE_C_ROOT / "manifests" / "phase_c_manifest.json", manifest_doc)

    dev_chi2 = dev_stats["friedman_med_rpd"]["statistic"]
    dev_p = dev_stats["friedman_med_rpd"]["p_value"]
    ho_chi2 = ho_stats["friedman_med_rpd"]["statistic"]
    ho_p = ho_stats["friedman_med_rpd"]["p_value"]
    lines = [
        "# Phase C Summary — Memetic & Dual-View Genetic Algorithms (`C01`–`C11`)",
        "",
        (
            f"- **Transferred Frozen Components:** $A^* =$ `{frozen_a_id}`, "
            f"$B^* =$ `{frozen_b_id}` (`{frozen_b_strat} + {frozen_b_nb}`, "
            f"$b_{{\\text{{LS}}}} = {args.max_ls_evals}$)"
        ),
        (
            f"- **Selected & Frozen Primary Winner ($C^*$):** **`{winner_id}` — "
            f"{dev_winner_row['config_name']}**"
        ),
        (
            "- **Development Split (16 instances):** Mean instance-level median `RPD_BKS` = "
            f"`{dev_winner_row['median_rpd_mean']:.3f}%`, Mean Best `RPD_BKS` = "
            f"`{dev_winner_row['best_rpd_mean']:.3f}%`, Friedman rank = "
            f"`{dev_winner_row['friedman_rank']:.2f}` / 11."
        ),
        (
            "- **Held-Out Generalization (16 held-out instances):** Median `RPD_BKS` = "
            f"`{ho_winner_row['median_rpd_mean']:.3f}%`, Best `RPD_BKS` = "
            f"`{ho_winner_row['best_rpd_mean']:.3f}%`, Hit BKS = "
            f"`{ho_winner_row['hit_bks_rate']:.3f}`."
        ),
        (
            f"- **Experimental Volume & Friedman:** `{len(all_rows):,}` runs (`32` instances "
            f"$\\times$ `11` configs $\\times$ `{args.replicates}` seeds); Dev $\\chi^2_F = "
            f"{dev_chi2:.2f}$ ($p = {dev_p:.3e}$), Held-Out $\\chi^2_F = {ho_chi2:.2f}$ "
            f"($p = {ho_p:.3e}$)."
        ),
        "",
        "## All 11 Phase C Configurations — Development & Held-Out Summary",
        "",
        (
            "| Rank (Dev) | ID | Method | Dev Med RPD | Dev Best RPD | Dev Rank | "
            "Held-Out Med RPD | Held-Out Best RPD | Mean Gate p | Final Div | Mean ms |"
        ),
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ho_by_id = {r["config_id"]: r for r in ho_summary}
    for rank, drow in enumerate(dev_summary, start=1):
        cid = drow["config_id"]
        hrow = ho_by_id[cid]
        gp_str = (
            f"{drow['final_gate_p_mean']:.3f}" if drow["final_gate_p_mean"] is not None else "—"
        )
        lines.append(
            f"| {rank} | `{cid}` | {drow['config_name']} | "
            f"{drow['median_rpd_mean']:.3f}% | {drow['best_rpd_mean']:.3f}% | "
            f"{drow['friedman_rank']:.2f} | {hrow['median_rpd_mean']:.3f}% | "
            f"{hrow['best_rpd_mean']:.3f}% | {gp_str} | "
            f"{drow['final_obr_div_mean']:.3f} | {drow['mean_elapsed_ms']:.1f} |"
        )

    win_uri = (PHASE_C_ROOT / "winner/frozen_winner_c.json").as_posix()
    st_uri = (PHASE_C_ROOT / "analysis/phase_c_statistical_tests.json").as_posix()
    man_uri = (PHASE_C_ROOT / "manifests/phase_c_manifest.json").as_posix()
    lines.extend(
        [
            "",
            "## Key Artifacts",
            f"- Frozen Winner $C^*$: [`results/phase_c/winner/frozen_winner_c.json`](file://{win_uri})",
            f"- Statistical Tests: [`results/phase_c/analysis/phase_c_statistical_tests.json`](file://{st_uri})",
            f"- Manifest: [`results/phase_c/manifests/phase_c_manifest.json`](file://{man_uri})",
        ]
    )
    (REPORTS_ROOT / "phase_c_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nPhase C COMPLETE. Frozen Winner C*: {winner_id} ({dev_winner_row['config_name']})")


if __name__ == "__main__":
    main()
