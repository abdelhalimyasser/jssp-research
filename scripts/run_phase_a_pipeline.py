"""Execute Phase A (A01-A24), compute statistics, select and freeze winner A*, and report."""

from __future__ import annotations

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
from jssp_research.config import load_classic_ga_config, load_phase_a_matrix
from jssp_research.experiments.manifests import canonical_json_sha256
from jssp_research.native import (
    PhaseAInitializerSpec,
    default_phase_a_ga_budget,
    get_frozen_master_seeds,
    get_rng_metadata,
    list_phase_a_initializers,
    run_phase_a_batch,
)
from jssp_research.utils.provenance import (
    collect_environment_provenance,
    collect_git_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_JSON = REPO_ROOT / "data/metadata/selected_benchmark_suite.json"
PHASE_A_ROOT = REPO_ROOT / "results/phase_a"
REPORTS_ROOT = REPO_ROOT / "reports"
NUM_REPLICATES = 20


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _init_record_to_row(
    inst: dict[str, Any],
    spec: PhaseAInitializerSpec,
    replicate: int,
    seed: int,
    budget: int,
    rec: Any,
) -> dict[str, Any]:
    bks = int(inst["best_known_value"])
    lb = inst["lower_bound"]
    init_diag = rec.initial
    fin_diag = rec.final_pop
    init_best_rpd = relative_percentage_deviation(init_diag.best_makespan, bks)
    init_mean_rpd = relative_percentage_deviation(init_diag.mean_makespan, bks)
    init_med_rpd = relative_percentage_deviation(init_diag.median_makespan, bks)
    final_best_rpd = relative_percentage_deviation(rec.best_makespan, bks)
    anytime_rpd_auc = relative_percentage_deviation(rec.anytime_cmax_auc, bks)

    return {
        "instance_id": inst["instance_id"],
        "family": inst["benchmark_family"],
        "split": inst["split"],
        "jobs": inst["jobs"],
        "machines": inst["machines"],
        "operation_count": inst["operation_count"],
        "bks": bks,
        "lb": lb,
        "config_id": spec.code,
        "config_name": spec.name,
        "alpha": spec.alpha,
        "is_deterministic_init": spec.is_deterministic,
        "uses_sobol": spec.uses_sobol,
        "dyadic_sobol_net_balance": spec.has_dyadic_sobol_net_balance,
        "replicate_index": replicate,
        "master_seed": seed,
        "evaluation_budget": budget,
        "evals_used": rec.evals_used,
        "generations_executed": rec.generations_executed,
        # Layer A initial population metrics
        "init_feasible_rate": init_diag.feasible_count / init_diag.population_size,
        "init_best_makespan": init_diag.best_makespan,
        "init_mean_makespan": init_diag.mean_makespan,
        "init_median_makespan": init_diag.median_makespan,
        "init_worst_makespan": init_diag.worst_makespan,
        "init_std_makespan": init_diag.std_makespan,
        "init_best_rpd": init_best_rpd,
        "init_mean_rpd": init_mean_rpd,
        "init_median_rpd": init_med_rpd,
        "init_unique_chromosomes": init_diag.unique_chromosomes,
        "init_unique_schedules": init_diag.unique_schedules,
        "init_obr_duplicate_rate": init_diag.obr_duplicate_rate,
        "init_schedule_duplicate_rate": init_diag.schedule_duplicate_rate,
        "init_obr_hamming_diversity": init_diag.obr_hamming_diversity,
        "init_schedule_diversity": init_diag.schedule_diversity,
        "init_qmc_1d_ks_discrepancy": init_diag.qmc_1d_ks_discrepancy,
        "init_elapsed_ms": (init_diag.generation_elapsed_ns + init_diag.decode_elapsed_ns) / 1e6,
        # Layer B downstream Classical GA (C01) metrics
        "ga_best_makespan": rec.best_makespan,
        "ga_best_rpd": final_best_rpd,
        "ga_best_gap_lb": lower_bound_gap(rec.best_makespan, lb),
        "ga_hit_bks": rec.best_makespan <= bks,
        "ga_final_mean_makespan": fin_diag.mean_makespan,
        "ga_final_median_makespan": fin_diag.median_makespan,
        "ga_final_unique_chromosomes": fin_diag.unique_chromosomes,
        "ga_final_unique_schedules": fin_diag.unique_schedules,
        "ga_final_obr_hamming_diversity": fin_diag.obr_hamming_diversity,
        "ga_final_schedule_diversity": fin_diag.schedule_diversity,
        "ga_best_found_eval": rec.best_found_eval,
        "ga_best_found_generation": rec.best_found_generation,
        "ga_anytime_cmax_auc": rec.anytime_cmax_auc,
        "ga_anytime_rpd_auc": anytime_rpd_auc,
        "ga_crossover_count": rec.crossover_count,
        "ga_mutation_count": rec.mutation_count,
        "ga_total_elapsed_ms": rec.total_elapsed_ns / 1e6,
        "best_schedule_hash": rec.best_schedule_hash,
    }


def _run_tier1_layer_a_242(
    instances: list[dict[str, Any]],
    specs: tuple[PhaseAInitializerSpec, ...],
    master_seeds: tuple[int, ...],
    raw_path: Path,
) -> list[dict[str, Any]]:
    """Evaluate standalone Layer A initialization across ALL 242 verified Track 1 instances."""
    records: list[dict[str, Any]] = []
    # Deterministic initializers take 1 evaluation replicate; stochastic take all 20 frozen seeds.
    base_tasks: list[tuple[str, int, int | None, str, str, str]] = []
    task_meta: list[tuple[PhaseAInitializerSpec, int]] = []
    for spec in specs:
        reps = [0] if spec.is_deterministic else list(range(NUM_REPLICATES))
        for r in reps:
            base_tasks.append((spec.code, r, None, "paired_init", "phase_a_shared_init", spec.code))
            task_meta.append((spec, r))

    with raw_path.open("w", encoding="utf-8") as fh:
        for idx, inst in enumerate(instances, start=1):
            parsed = parse_jssp_file(
                REPO_ROOT / inst["raw_relative_path"], instance_id=inst["instance_id"]
            )
            res_list = run_phase_a_batch(
                parsed.jobs,
                inst["instance_id"],
                base_tasks,
                pop_size=64,
                evaluation_budget=64,
                num_threads=0,
                trajectory_mode="none",
            )
            for (spec, rep), rec in zip(task_meta, res_list, strict=True):
                row = _init_record_to_row(inst, spec, rep, master_seeds[rep], 64, rec)
                records.append(row)
                fh.write(json.dumps(row) + "\n")
            if idx % 50 == 0 or idx == len(instances):
                print(f"  [Layer A 242] processed {idx}/{len(instances)} instances", flush=True)
    return records


def _run_tier2_common_suite(
    instances: list[dict[str, Any]],
    specs: tuple[PhaseAInitializerSpec, ...],
    master_seeds: tuple[int, ...],
    raw_path: Path,
) -> list[dict[str, Any]]:
    """Run all 24 initializers + Classical GA (C01) x 20 frozen seeds on Common Suite (32 inst)."""
    records: list[dict[str, Any]] = []
    tasks: list[tuple[str, int, int | None, str, str, str]] = []
    task_meta: list[tuple[PhaseAInitializerSpec, int]] = []
    for spec in specs:
        for rep in range(NUM_REPLICATES):
            tasks.append((spec.code, rep, None, "paired_init", "phase_a_shared_init", spec.code))
            task_meta.append((spec, rep))

    with raw_path.open("w", encoding="utf-8") as fh:
        for idx, inst in enumerate(instances, start=1):
            t0 = time.perf_counter()
            parsed = parse_jssp_file(
                REPO_ROOT / inst["raw_relative_path"], instance_id=inst["instance_id"]
            )
            budget = default_phase_a_ga_budget(parsed.num_jobs, parsed.num_machines)
            res_list = run_phase_a_batch(
                parsed.jobs,
                inst["instance_id"],
                tasks,
                pop_size=64,
                tournament_size=2,
                crossover_rate=0.85,
                mutation_rate=0.15,
                elitism_count=2,
                evaluation_budget=budget,
                num_threads=0,
                trajectory_mode="sampled",
            )
            dt = time.perf_counter() - t0
            for (spec, rep), rec in zip(task_meta, res_list, strict=True):
                row = _init_record_to_row(inst, spec, rep, master_seeds[rep], budget, rec)
                row["sampled_trajectory"] = [
                    {
                        "generation": s.generation,
                        "evals_used": s.evals_used,
                        "best_makespan": s.best_makespan,
                        "mean_makespan": s.mean_makespan,
                        "obr_hamming_diversity": s.obr_hamming_diversity,
                        "unique_chromosomes": s.unique_chromosomes,
                    }
                    for s in rec.trajectory
                ]
                fh.write(json.dumps(row) + "\n")
                # Strip trajectory in memory table to keep aggregation lean
                slim = dict(row)
                slim.pop("sampled_trajectory", None)
                records.append(slim)
            best_row = min(records[-len(tasks) :], key=lambda r: r["ga_best_makespan"])
            print(
                f"  [Phase A Tier 2 {idx:2d}/{len(instances)}] {inst['instance_id']:6s} "
                f"({inst['split']:8s}, {inst['jobs']:2d}x{inst['machines']:2d}, B={budget:5d}): "
                f"480 runs in {dt:6.2f}s | best Cmax={best_row['ga_best_makespan']} "
                f"(BKS={inst['best_known_value']}, {best_row['config_id']})",
                flush=True,
            )
    return records


def _summarize_tier2_split(
    rows: list[dict[str, Any]],
    specs: tuple[PhaseAInitializerSpec, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate instance-level medians across seeds and run Friedman + Holm-Wilcoxon tests."""
    instance_ids = sorted({r["instance_id"] for r in rows})
    by_cfg_inst: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        by_cfg_inst.setdefault((r["config_id"], r["instance_id"]), []).append(r)

    cfg_ids = [s.code for s in specs]
    spec_by_id = {s.code: s for s in specs}

    # Build instance-blocked matrices (shape: n_inst x 24)
    ga_med_rpd_mat = np.zeros((len(instance_ids), len(cfg_ids)), dtype=float)
    init_best_rpd_mat = np.zeros((len(instance_ids), len(cfg_ids)), dtype=float)
    scores_ga_rpd: dict[str, np.ndarray] = {}

    summary_rows: list[dict[str, Any]] = []
    for c_idx, cid in enumerate(cfg_ids):
        spec = spec_by_id[cid]
        inst_ga_med_rpds = []
        inst_ga_mean_rpds = []
        inst_ga_best_rpds = []
        inst_ga_med_cmax = []
        inst_ga_best_cmax = []
        inst_ga_std_cmax = []
        inst_ga_iqr_cmax = []
        inst_ga_hit_rate = []
        inst_ga_auc_rpd = []
        inst_init_best_rpd = []
        inst_init_mean_rpd = []
        inst_init_med_rpd = []
        inst_init_ham_div = []
        inst_init_sched_div = []
        inst_init_dup_obr = []
        inst_init_dup_sched = []
        inst_init_ks = []
        inst_init_ms = []
        inst_ga_ms = []
        inst_feasible = []

        for i_idx, iid in enumerate(instance_ids):
            grp = by_cfg_inst[(cid, iid)]
            ga_rpds = np.array([g["ga_best_rpd"] for g in grp], dtype=float)
            ga_cmax = np.array([g["ga_best_makespan"] for g in grp], dtype=float)
            s_cmax = summarize_numeric(ga_cmax)

            # For deterministic initializers, all 20 Layer A copies are identical.
            init_best = float(np.median([g["init_best_rpd"] for g in grp]))
            init_mean = float(np.median([g["init_mean_rpd"] for g in grp]))
            init_med = float(np.median([g["init_median_rpd"] for g in grp]))

            ga_med_rpd = float(np.median(ga_rpds))
            ga_med_rpd_mat[i_idx, c_idx] = ga_med_rpd
            init_best_rpd_mat[i_idx, c_idx] = init_best

            inst_ga_med_rpds.append(ga_med_rpd)
            inst_ga_mean_rpds.append(float(np.mean(ga_rpds)))
            inst_ga_best_rpds.append(float(np.min(ga_rpds)))
            inst_ga_med_cmax.append(s_cmax["median"])
            inst_ga_best_cmax.append(s_cmax["min"])
            inst_ga_std_cmax.append(s_cmax["std"])
            inst_ga_iqr_cmax.append(s_cmax["iqr"])
            inst_ga_hit_rate.append(float(np.mean([1.0 if g["ga_hit_bks"] else 0.0 for g in grp])))
            inst_ga_auc_rpd.append(float(np.median([g["ga_anytime_rpd_auc"] for g in grp])))

            inst_init_best_rpd.append(init_best)
            inst_init_mean_rpd.append(init_mean)
            inst_init_med_rpd.append(init_med)
            inst_init_ham_div.append(float(np.mean([g["init_obr_hamming_diversity"] for g in grp])))
            inst_init_sched_div.append(float(np.mean([g["init_schedule_diversity"] for g in grp])))
            inst_init_dup_obr.append(float(np.mean([g["init_obr_duplicate_rate"] for g in grp])))
            inst_init_dup_sched.append(
                float(np.mean([g["init_schedule_duplicate_rate"] for g in grp]))
            )
            ks_vals = [
                g["init_qmc_1d_ks_discrepancy"]
                for g in grp
                if g["init_qmc_1d_ks_discrepancy"] is not None
            ]
            if ks_vals:
                inst_init_ks.append(float(np.mean(ks_vals)))
            inst_init_ms.append(float(np.mean([g["init_elapsed_ms"] for g in grp])))
            inst_ga_ms.append(float(np.mean([g["ga_total_elapsed_ms"] for g in grp])))
            inst_feasible.append(float(np.mean([g["init_feasible_rate"] for g in grp])))

        scores_ga_rpd[cid] = ga_med_rpd_mat[:, c_idx].copy()
        actual_runs = sum(len(by_cfg_inst[(cid, iid)]) for iid in instance_ids)
        summary_rows.append(
            {
                "config_id": cid,
                "config_name": spec.name,
                "alpha": spec.alpha if spec.alpha is not None else "inf(MWR)",
                "deterministic_init": spec.is_deterministic,
                "uses_sobol": spec.uses_sobol,
                "dyadic_sobol_net": spec.has_dyadic_sobol_net_balance,
                "n_instances": len(instance_ids),
                "n_runs": actual_runs,
                "init_feasible_rate": float(np.mean(inst_feasible)),
                "init_best_rpd_mean": float(np.mean(inst_init_best_rpd)),
                "init_mean_rpd_mean": float(np.mean(inst_init_mean_rpd)),
                "init_median_rpd_mean": float(np.mean(inst_init_med_rpd)),
                "init_obr_hamming_div_mean": float(np.mean(inst_init_ham_div)),
                "init_schedule_div_mean": float(np.mean(inst_init_sched_div)),
                "init_obr_dup_rate_mean": float(np.mean(inst_init_dup_obr)),
                "init_sched_dup_rate_mean": float(np.mean(inst_init_dup_sched)),
                "init_1d_ks_discrepancy_mean": float(np.mean(inst_init_ks))
                if inst_init_ks
                else None,
                "init_elapsed_ms_mean": float(np.mean(inst_init_ms)),
                "ga_best_rpd_mean": float(np.mean(inst_ga_best_rpds)),
                "ga_mean_rpd_mean": float(np.mean(inst_ga_mean_rpds)),
                "ga_median_rpd_mean": float(np.mean(inst_ga_med_rpds)),
                "ga_median_of_instance_medians_rpd": float(np.median(inst_ga_med_rpds)),
                "ga_best_cmax_mean": float(np.mean(inst_ga_best_cmax)),
                "ga_median_cmax_mean": float(np.mean(inst_ga_med_cmax)),
                "ga_within_instance_std_mean": float(np.mean(inst_ga_std_cmax)),
                "ga_within_instance_iqr_mean": float(np.mean(inst_ga_iqr_cmax)),
                "ga_anytime_rpd_auc_mean": float(np.mean(inst_ga_auc_rpd)),
                "ga_hit_bks_rate": float(np.mean(inst_ga_hit_rate)),
                "ga_total_elapsed_ms_mean": float(np.mean(inst_ga_ms)),
            }
        )

    friedman_ga = run_friedman_analysis(cfg_ids, ga_med_rpd_mat)
    friedman_init = run_friedman_analysis(cfg_ids, init_best_rpd_mat)

    for row in summary_rows:
        cid = row["config_id"]
        row["friedman_ga_rank"] = friedman_ga.mean_ranks[cid]
        row["friedman_init_rank"] = friedman_init.mean_ranks[cid]

    # Sort deterministically by Winner Selection rule:
    # 1) lowest mean instance-level median GA RPD; 2) lowest Friedman rank;
    # 3) highest initial OBR Hamming diversity; 4) lowest runtime.
    summary_rows.sort(
        key=lambda r: (
            r["ga_median_rpd_mean"],
            r["friedman_ga_rank"],
            -r["init_obr_hamming_div_mean"],
            r["ga_total_elapsed_ms_mean"],
        )
    )
    winner_id = summary_rows[0]["config_id"]

    # Planned comparison families
    pairs: list[tuple[str, str, str]] = [
        # Baseline controls vs A01 Uniform Random
        ("qmc_vs_random_a01", "A03", "A01"),
        ("qmc_vs_random_a01", "A04", "A01"),
        ("qmc_vs_random_a01", "A05", "A01"),
        ("qmc_vs_random_a01", "A06", "A01"),
        ("qmc_vs_random_a01", "A07", "A01"),
        ("qmc_vs_random_a01", "A09", "A01"),
        ("qmc_vs_random_a01", "A11", "A01"),
        ("qmc_vs_random_a01", "A12", "A01"),
        # MWR effect within generator
        ("mwr_effect", "A03", "A01"),
        ("mwr_effect", "A08", "A04"),
        ("mwr_effect", "A09", "A05"),
        ("mwr_effect", "A10", "A06"),
        ("mwr_effect", "A11", "A07"),
        # Scrambling effect
        ("scrambling_effect", "A05", "A04"),
        ("scrambling_effect", "A07", "A06"),
        ("scrambling_effect", "A09", "A08"),
        ("scrambling_effect", "A11", "A10"),
        # Alpha sensitivity vs alpha=1.0
        ("alpha_sensitivity", "A15", "A03"),
        ("alpha_sensitivity", "A16", "A03"),
        ("alpha_sensitivity", "A19", "A09"),
        ("alpha_sensitivity", "A20", "A09"),
        ("alpha_sensitivity", "A23", "A11"),
        ("alpha_sensitivity", "A24", "A11"),
        # Mixed Halton/Sobol vs pure components
        ("mixed_qmc_vs_pure", "A12", "A05"),
        ("mixed_qmc_vs_pure", "A12", "A07"),
        ("mixed_qmc_vs_pure", "A13", "A07"),
        ("mixed_qmc_vs_pure", "A14", "A05"),
    ]
    for baseline in ["A01", "A02", "A03", "A05", "A07", "A09", "A11", "A12"]:
        if winner_id != baseline:
            pairs.append(("winner_vs_key_baselines", winner_id, baseline))

    wilcoxon_rows = paired_comparisons_table(scores_ga_rpd, pairs, lower_is_better=True)

    stats_bundle = {
        "n_instances": len(instance_ids),
        "friedman_downstream_ga_med_rpd": {
            "statistic": friedman_ga.statistic,
            "p_value": friedman_ga.p_value,
            "kendall_w": friedman_ga.kendall_w,
            "mean_ranks": friedman_ga.mean_ranks,
        },
        "friedman_initial_best_rpd": {
            "statistic": friedman_init.statistic,
            "p_value": friedman_init.p_value,
            "kendall_w": friedman_init.kendall_w,
            "mean_ranks": friedman_init.mean_ranks,
        },
        "paired_wilcoxon_holm": wilcoxon_rows,
        "selected_top_id": winner_id,
    }
    return summary_rows, stats_bundle


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started_utc = datetime.now(tz=UTC).isoformat()
    suite_doc = json.loads(SUITE_JSON.read_text("utf-8"))
    all_instances = suite_doc["instances"]
    common_instances = [r for r in all_instances if r["in_common_pipeline_suite"]]

    specs = list_phase_a_initializers()
    master_seeds = tuple(int(s) for s in get_frozen_master_seeds())
    rng_meta = get_rng_metadata()

    for sub in ("raw", "manifests", "analysis", "winner"):
        (PHASE_A_ROOT / sub).mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Starting Phase A Layer A across ALL {len(all_instances)} Track 1 instances...")
    layer_a_raw_path = PHASE_A_ROOT / "raw" / "phase_a_layer_a_242.jsonl"
    layer_a_rows = _run_tier1_layer_a_242(all_instances, specs, master_seeds, layer_a_raw_path)

    print(
        f"Starting Phase A Tier 2 GA Evaluation across {len(common_instances)} "
        "Common Suite instances (15,360 runs)..."
    )
    tier2_raw_path = PHASE_A_ROOT / "raw" / "phase_a_common32_runs.jsonl"
    tier2_rows = _run_tier2_common_suite(common_instances, specs, master_seeds, tier2_raw_path)

    # Summarize Layer A across all 242 instances
    layer_a_summary, _ = _summarize_tier2_split(layer_a_rows, specs)
    _write_csv(PHASE_A_ROOT / "analysis" / "phase_a_layer_a_242_summary.csv", layer_a_summary)

    # Tier 2 strictly split into Development (16) and Held-Out (16)
    dev_rows = [r for r in tier2_rows if r["split"] == "development"]
    ho_rows = [r for r in tier2_rows if r["split"] == "held_out"]

    dev_summary, dev_stats = _summarize_tier2_split(dev_rows, specs)
    ho_summary, ho_stats = _summarize_tier2_split(ho_rows, specs)

    _write_csv(PHASE_A_ROOT / "analysis" / "phase_a_dev_summary.csv", dev_summary)
    _write_csv(PHASE_A_ROOT / "analysis" / "phase_a_held_out_summary.csv", ho_summary)

    # Family breakdown across common 32 instances
    fam_rows: list[dict[str, Any]] = []
    for fam in sorted({r["family"] for r in tier2_rows}):
        sub = [r for r in tier2_rows if r["family"] == fam]
        for s in specs:
            cfg_sub = [r for r in sub if r["config_id"] == s.code]
            fam_rows.append(
                {
                    "family": fam,
                    "config_id": s.code,
                    "n_runs": len(cfg_sub),
                    "init_best_rpd_mean": float(np.mean([r["init_best_rpd"] for r in cfg_sub])),
                    "init_hamming_div_mean": float(
                        np.mean([r["init_obr_hamming_diversity"] for r in cfg_sub])
                    ),
                    "ga_best_rpd_mean": float(np.mean([r["ga_best_rpd"] for r in cfg_sub])),
                    "ga_median_cmax_mean": float(np.mean([r["ga_best_makespan"] for r in cfg_sub])),
                }
            )
    _write_csv(PHASE_A_ROOT / "analysis" / "phase_a_family_breakdown.csv", fam_rows)

    stats_out = {
        "schema_version": "jssp-phase-a-stats-v1",
        "development_16": dev_stats,
        "held_out_16": ho_stats,
    }
    (PHASE_A_ROOT / "analysis" / "phase_a_friedman_and_wilcoxon.json").write_text(
        json.dumps(stats_out, indent=2) + "\n", encoding="utf-8"
    )

    # Pre-registered Winner Selection strictly on Development Split (16 instances)
    dev_winner_row = dev_summary[0]
    winner_id = str(dev_winner_row["config_id"])
    winner_matrix_cfg = next(c for c in load_phase_a_matrix().configs if c.config_id == winner_id)
    winner_ho_row = next(r for r in ho_summary if r["config_id"] == winner_id)

    winner_body = {
        "schema_version": "jssp-frozen-winner-a-v1",
        "frozen_at_utc": datetime.now(tz=UTC).isoformat(),
        "selection_protocol": (
            "Selected strictly on the 16-instance Development Split by lowest mean instance-level "
            "median downstream Classical GA (C01) RPD_BKS, breaking ties by dev Friedman rank, "
            "initial OBR Hamming diversity (>0.50), and runtime."
        ),
        "winner_config_id": winner_id,
        "winner_config_name": dev_winner_row["config_name"],
        "winner_parameters": winner_matrix_cfg.model_dump(mode="json"),
        "development_metrics": dev_winner_row,
        "held_out_validation_metrics": winner_ho_row,
    }
    winner_body["winner_lock_sha256"] = canonical_json_sha256(winner_body)
    (PHASE_A_ROOT / "winner" / "frozen_winner_a.json").write_text(
        json.dumps(winner_body, indent=2) + "\n", encoding="utf-8"
    )

    # Manifest index
    git_prov = collect_git_provenance(REPO_ROOT)
    env_prov = collect_environment_provenance(REPO_ROOT)
    ga_cfg = load_classic_ga_config().model_dump(mode="json")
    manifest_doc = {
        "schema_version": "jssp-phase-a-manifest-index-v1",
        "phase": "phase_a",
        "started_utc": started_utc,
        "completed_utc": datetime.now(tz=UTC).isoformat(),
        "git": git_prov.model_dump(mode="json"),
        "environment": env_prov.model_dump(mode="json"),
        "rng": {
            "algorithm": rng_meta.algorithm,
            "seed_schedule_id": rng_meta.master_seed_schedule_id,
            "master_seeds_v1": list(master_seeds),
            "pairing_mode": "paired_init",
            "pairing_group": "phase_a_shared_init",
        },
        "classic_ga_baseline_c01": ga_cfg,
        "layer_a_242_runs": len(layer_a_rows),
        "tier2_common32_runs": len(tier2_rows),
        "selected_benchmark_suite_sha256": _sha256_file(SUITE_JSON),
        "layer_a_raw_sha256": _sha256_file(layer_a_raw_path),
        "tier2_raw_sha256": _sha256_file(tier2_raw_path),
        "frozen_winner_a_sha256": winner_body["winner_lock_sha256"],
    }
    (PHASE_A_ROOT / "manifests" / "phase_a_manifest_index.json").write_text(
        json.dumps(manifest_doc, indent=2) + "\n", encoding="utf-8"
    )

    # Concise summary report
    dev_chi2 = dev_stats["friedman_downstream_ga_med_rpd"]["statistic"]
    dev_p = dev_stats["friedman_downstream_ga_med_rpd"]["p_value"]
    dev_w = dev_stats["friedman_downstream_ga_med_rpd"]["kendall_w"]
    ho_chi2 = ho_stats["friedman_downstream_ga_med_rpd"]["statistic"]
    ho_p = ho_stats["friedman_downstream_ga_med_rpd"]["p_value"]
    w_name = dev_winner_row["config_name"]
    lines = [
        "# Phase A Summary — Initial Population Generation (`A01`–`A24`)",
        "",
        f"- **Selected & Frozen Winner ($A^*$):** **`{winner_id}` — {w_name}**",
        (
            "- **Winner Selection Rule (Dev Only, 16 instances):** Lowest mean instance-level "
            f"median `C01` GA `RPD_BKS` = `{dev_winner_row['ga_median_rpd_mean']:.3f}%` "
            f"(Friedman rank `{dev_winner_row['friedman_ga_rank']:.2f}` / 24)."
        ),
        (
            "- **Held-Out Generalization ($A^*$, 16 held-out instances):** Median GA `RPD_BKS` = "
            f"`{winner_ho_row['ga_median_rpd_mean']:.3f}%`, Mean Best `RPD_BKS` = "
            f"`{winner_ho_row['ga_best_rpd_mean']:.3f}%`, Initial OBR Diversity = "
            f"`{winner_ho_row['init_obr_hamming_div_mean']:.4f}`."
        ),
        (
            f"- **Experimental Volume:** `{len(layer_a_rows):,}` standalone Layer A runs "
            f"across all `242` verified Track 1 instances + `{len(tier2_rows):,}` full `C01` "
            f"Classical GA runs across the `{len(common_instances)}`-instance Common Suite "
            "(`16` dev + `16` held-out $\\times$ `24` configs $\\times$ `20` frozen seeds)."
        ),
        (
            f"- **Blocked Friedman Test (Downstream GA Median RPD):** Dev $\\chi^2_F = "
            f"{dev_chi2:.2f}$ ($p = {dev_p:.3e}$, Kendall $W = {dev_w:.4f}$); "
            f"Held-Out $\\chi^2_F = {ho_chi2:.2f}$ ($p = {ho_p:.3e}$)."
        ),
        "",
        "## All 24 Configurations — Development (Winner Selection) & Held-Out Results",
        "",
        (
            "| Rank (Dev) | ID | Method | Init Best RPD (Dev) | Init Div (Dev) | "
            "GA Med RPD (Dev) | Dev Rank | GA Med RPD (Held-Out) | "
            "GA Best RPD (Held-Out) | Mean Time (ms) |"
        ),
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ho_by_id = {r["config_id"]: r for r in ho_summary}
    for rank, drow in enumerate(dev_summary, start=1):
        cid = drow["config_id"]
        hrow = ho_by_id[cid]
        lines.append(
            f"| {rank} | `{cid}` | {drow['config_name']} | "
            f"{drow['init_best_rpd_mean']:.2f}% | {drow['init_obr_hamming_div_mean']:.3f} | "
            f"{drow['ga_median_rpd_mean']:.3f}% | {drow['friedman_ga_rank']:.2f} | "
            f"{hrow['ga_median_rpd_mean']:.3f}% | {hrow['ga_best_rpd_mean']:.3f}% | "
            f"{drow['ga_total_elapsed_ms_mean']:.1f} |"
        )
    win_json_uri = (PHASE_A_ROOT / "winner/frozen_winner_a.json").as_posix()
    stats_uri = (PHASE_A_ROOT / "analysis/phase_a_friedman_and_wilcoxon.json").as_posix()
    lay_uri = (PHASE_A_ROOT / "analysis/phase_a_layer_a_242_summary.csv").as_posix()
    man_uri = (PHASE_A_ROOT / "manifests/phase_a_manifest_index.json").as_posix()
    lines.extend(
        [
            "",
            "## Key Artifacts",
            f"- Frozen Winner Lock (`{winner_body['winner_lock_sha256'][:16]}...`): "
            f"[`results/phase_a/winner/frozen_winner_a.json`](file://{win_json_uri})",
            "- Statistical Tests (Friedman + Holm-Wilcoxon + A12 + Cliff's d): "
            f"[`results/phase_a/analysis/phase_a_friedman_and_wilcoxon.json`](file://{stats_uri})",
            "- Full 242-Instance Layer A Summary: "
            f"[`results/phase_a/analysis/phase_a_layer_a_242_summary.csv`](file://{lay_uri})",
            "- Manifest Index: "
            f"[`results/phase_a/manifests/phase_a_manifest_index.json`](file://{man_uri})",
        ]
    )
    (REPORTS_ROOT / "phase_a_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nPhase A COMPLETE. Frozen Winner A*: {winner_id} ({dev_winner_row['config_name']})")


if __name__ == "__main__":
    main()
