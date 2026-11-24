"""Execute Phase B (B01-B24, B29-B32), freeze winner B*, and generate analysis/reports."""

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
from jssp_research.config import load_phase_b_matrix
from jssp_research.experiments.manifests import canonical_json_sha256
from jssp_research.native import (
    PhaseBConfigSpec,
    default_phase_b_budget,
    get_frozen_master_seeds,
    get_rng_metadata,
    list_phase_b_configs,
    run_phase_b_batch,
)
from jssp_research.utils.provenance import (
    collect_environment_provenance,
    collect_git_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_JSON = REPO_ROOT / "data/metadata/selected_benchmark_suite.json"
FROZEN_A_JSON = REPO_ROOT / "results/phase_a/winner/frozen_winner_a.json"
PHASE_B_ROOT = REPO_ROOT / "results/phase_b"
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


def _phase_b_record_to_row(
    inst: dict[str, Any],
    spec: PhaseBConfigSpec,
    initializer_id: str,
    replicate: int,
    seed: int,
    budget: int,
    rec: Any,
) -> dict[str, Any]:
    bks = int(inst["best_known_value"])
    lb = inst["lower_bound"]
    c = rec.counters
    init_rpd = relative_percentage_deviation(rec.initial_makespan, bks)
    best_rpd = relative_percentage_deviation(rec.best_makespan, bks)
    final_rpd = relative_percentage_deviation(rec.final_makespan, bks)
    anytime_rpd_auc = relative_percentage_deviation(rec.anytime_cmax_auc, bks)
    improvement_delta = float(rec.initial_makespan - rec.best_makespan)
    improvement_pct = (
        100.0 * improvement_delta / float(rec.initial_makespan) if rec.initial_makespan > 0 else 0.0
    )

    return {
        "instance_id": inst["instance_id"],
        "family": inst["benchmark_family"],
        "split": inst["split"],
        "jobs": inst["jobs"],
        "machines": inst["machines"],
        "operation_count": inst["operation_count"],
        "bks": bks,
        "lb": lb,
        "initializer_id": initializer_id,
        "config_id": spec.code,
        "strategy_name": spec.strategy_name,
        "neighborhood_id": spec.neighborhood_code,
        "replicate_index": replicate,
        "master_seed": seed,
        "evaluation_budget": budget,
        "evals_used": rec.evals_used,
        "initial_makespan": rec.initial_makespan,
        "final_makespan": rec.final_makespan,
        "best_makespan": rec.best_makespan,
        "init_rpd_bks": init_rpd,
        "best_rpd_bks": best_rpd,
        "final_rpd_bks": final_rpd,
        "best_gap_lb": lower_bound_gap(rec.best_makespan, lb),
        "hit_bks": rec.best_makespan <= bks,
        "improvement_delta": improvement_delta,
        "improvement_pct": improvement_pct,
        "best_found_eval": rec.best_found_eval,
        "feasible_final": rec.feasible_final,
        "reached_local_optimum": rec.reached_local_optimum,
        "terminated_reason": rec.terminated_reason,
        "anytime_cmax_auc": rec.anytime_cmax_auc,
        "anytime_rpd_auc": anytime_rpd_auc,
        "iterations": c.iterations,
        "generated_moves": c.generated_moves,
        "duplicate_pruned_moves": c.duplicate_pruned_moves,
        "infeasible_precond_moves": c.infeasible_precond_moves,
        "clipped_moves": c.clipped_moves,
        "evaluated_moves": c.evaluated_moves,
        "feasible_moves": c.feasible_moves,
        "rejected_cyclic_moves": c.rejected_cyclic_moves,
        "improving_moves": c.improving_moves,
        "accepted_moves": c.accepted_moves,
        "adapter_left_shift_count": c.adapter_left_shift_count,
        "perturbations_applied": c.perturbations_applied,
        "perturbations_accepted": c.perturbations_accepted,
        "restarts_triggered": c.restarts_triggered,
        "total_elapsed_ms": rec.total_elapsed_ns / 1e6,
    }


def _run_phase_b_sweep(
    instances: list[dict[str, Any]],
    specs: tuple[PhaseBConfigSpec, ...],
    initializer_id: str,
    master_seeds: tuple[int, ...],
    budget_multiplier: int,
    replicates: int,
    raw_path: Path,
) -> list[dict[str, Any]]:
    """Run all active Phase B configurations with matched frozen A* starts across instances."""
    records: list[dict[str, Any]] = []
    tasks: list[tuple[str, int, int | None, str, str, str]] = []
    task_meta: list[tuple[PhaseBConfigSpec, int]] = []
    for spec in specs:
        for rep in range(replicates):
            tasks.append((spec.code, rep, None, "paired_init", "phase_b_matched_start", spec.code))
            task_meta.append((spec, rep))

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as fh:
        for idx, inst in enumerate(instances, start=1):
            t0 = time.perf_counter()
            parsed = parse_jssp_file(
                REPO_ROOT / inst["raw_relative_path"], instance_id=inst["instance_id"]
            )
            budget = default_phase_b_budget(
                parsed.num_jobs, parsed.num_machines, multiplier=budget_multiplier
            )
            res_list = run_phase_b_batch(
                parsed.jobs,
                inst["instance_id"],
                tasks,
                initializer_id=initializer_id,
                init_pop_size=64,
                evaluation_budget=budget,
                budget_multiplier=budget_multiplier,
                num_threads=0,
            )
            dt = time.perf_counter() - t0
            for (spec, rep), rec in zip(task_meta, res_list, strict=True):
                if not rec.feasible_final:
                    raise RuntimeError(
                        f"Infeasible Phase B schedule: {inst['instance_id']} {spec.code} rep={rep}"
                    )
                if np.isnan(rec.anytime_cmax_auc):
                    raise RuntimeError(
                        f"NaN AUC returned: {inst['instance_id']} {spec.code} rep={rep}"
                    )
                row = _phase_b_record_to_row(
                    inst, spec, initializer_id, rep, master_seeds[rep], budget, rec
                )
                records.append(row)
                raw_row = dict(row)
                raw_row["best_milestones"] = [[e, cmax] for e, cmax in rec.best_milestones]
                fh.write(json.dumps(raw_row) + "\n")

            best_row = min(records[-len(tasks) :], key=lambda r: r["best_makespan"])
            print(
                f"  [Phase B {idx:2d}/{len(instances)}] {inst['instance_id']:6s} "
                f"({inst['split']:8s}, {inst['jobs']:2d}x{inst['machines']:2d}, B={budget:5d}): "
                f"{len(tasks)} runs in {dt:6.2f}s | best Cmax={best_row['best_makespan']} "
                f"(BKS={inst['best_known_value']}, {best_row['config_id']})",
                flush=True,
            )
    return records


def _summarize_split(
    rows: list[dict[str, Any]],
    specs: tuple[PhaseBConfigSpec, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate instance-level medians on a split and compute Friedman + Holm-Wilcoxon tests."""
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
        inst_lo_rate: list[float] = []
        inst_auc_rpd: list[float] = []
        inst_imp_pct: list[float] = []
        inst_ms: list[float] = []
        inst_evals: list[float] = []
        inst_cycles: list[float] = []
        inst_left_shifts: list[float] = []

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
            inst_lo_rate.append(
                float(np.mean([1.0 if g["reached_local_optimum"] else 0.0 for g in grp]))
            )
            inst_auc_rpd.append(float(np.median([g["anytime_rpd_auc"] for g in grp])))
            inst_imp_pct.append(float(np.mean([g["improvement_pct"] for g in grp])))
            inst_ms.append(float(np.mean([g["total_elapsed_ms"] for g in grp])))
            inst_evals.append(float(np.mean([g["evals_used"] for g in grp])))
            inst_cycles.append(float(np.mean([g["rejected_cyclic_moves"] for g in grp])))
            inst_left_shifts.append(float(np.mean([g["adapter_left_shift_count"] for g in grp])))

        scores_by_cfg[cid] = med_rpd_mat[:, c_idx].copy()
        actual_runs = sum(len(by_cfg_inst[(cid, iid)]) for iid in instance_ids)
        summary_rows.append(
            {
                "config_id": cid,
                "strategy_name": spec.strategy_name,
                "neighborhood_id": spec.neighborhood_code,
                "n_instances": len(instance_ids),
                "n_runs": actual_runs,
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
                "improvement_pct_mean": float(np.mean(inst_imp_pct)),
                "hit_bks_rate": float(np.mean(inst_hit_rate)),
                "local_optimum_rate": float(np.mean(inst_lo_rate)),
                "mean_evals_used": float(np.mean(inst_evals)),
                "mean_rejected_cyclic_moves": float(np.mean(inst_cycles)),
                "mean_adapter_left_shifts": float(np.mean(inst_left_shifts)),
                "mean_elapsed_ms": float(np.mean(inst_ms)),
            }
        )

    friedman = run_friedman_analysis(cfg_ids, med_rpd_mat)
    for row in summary_rows:
        row["friedman_rank"] = friedman.mean_ranks[row["config_id"]]

    # Selection order:
    # 1) lowest mean instance-level median RPD_BKS
    # 2) lowest Friedman rank
    # 3) lowest anytime RPD AUC
    # 4) lowest mean runtime
    summary_rows.sort(
        key=lambda r: (
            r["median_rpd_mean"],
            r["friedman_rank"],
            r["anytime_rpd_auc_mean"],
            r["mean_elapsed_ms"],
        )
    )
    winner_id = summary_rows[0]["config_id"]

    # Planned comparison families:
    # 1) Neighborhood N8 vs N5/N6/N7 within key strategies
    # 2) Winner vs representative baselines
    pairs: list[tuple[str, str, str]] = [
        ("n_vs_n5_steepest", "B02", "B01"),
        ("n_vs_n5_steepest", "B03", "B01"),
        ("n_vs_n5_steepest", "B04", "B01"),
        ("n_vs_n5_tabu", "B14", "B13"),
        ("n_vs_n5_tabu", "B15", "B13"),
        ("n_vs_n5_tabu", "B16", "B13"),
        ("n_vs_n5_ils", "B18", "B17"),
        ("n_vs_n5_ils", "B19", "B17"),
        ("n_vs_n5_ils", "B20", "B17"),
        ("n_vs_n5_ils_mh", "B30", "B29"),
        ("n_vs_n5_ils_mh", "B31", "B29"),
        ("n_vs_n5_ils_mh", "B32", "B29"),
        ("ils_mh_vs_ils", "B29", "B17"),
        ("ils_mh_vs_ils", "B30", "B18"),
        ("ils_mh_vs_ils", "B31", "B19"),
        ("ils_mh_vs_ils", "B32", "B20"),
    ]
    for baseline in ["B01", "B04", "B05", "B08", "B12", "B16", "B20", "B24", "B32"]:
        if winner_id != baseline:
            pairs.append(("winner_vs_baselines", winner_id, baseline))

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
        "selected_top_id": winner_id,
    }
    return summary_rows, stats_bundle


def _summarize_marginal(
    rows: list[dict[str, Any]],
    group_key: str,
) -> list[dict[str, Any]]:
    """Compute marginal performance across neighborhood (`N5`–`N8`) or strategy rows."""
    groups = sorted({str(r[group_key]) for r in rows})
    out: list[dict[str, Any]] = []
    for g in groups:
        sub = [r for r in rows if str(r[group_key]) == g]
        dev_sub = [r for r in sub if r["split"] == "development"]
        ho_sub = [r for r in sub if r["split"] == "held_out"]
        out.append(
            {
                group_key: g,
                "n_runs_total": len(sub),
                "dev_mean_best_rpd": float(np.mean([r["best_rpd_bks"] for r in dev_sub])),
                "dev_mean_anytime_rpd_auc": float(np.mean([r["anytime_rpd_auc"] for r in dev_sub])),
                "ho_mean_best_rpd": float(np.mean([r["best_rpd_bks"] for r in ho_sub])),
                "ho_mean_anytime_rpd_auc": float(np.mean([r["anytime_rpd_auc"] for r in ho_sub])),
                "mean_improvement_pct": float(np.mean([r["improvement_pct"] for r in sub])),
                "mean_hit_bks_rate": float(np.mean([1.0 if r["hit_bks"] else 0.0 for r in sub])),
                "mean_elapsed_ms": float(np.mean([r["total_elapsed_ms"] for r in sub])),
                "mean_rejected_cyclic_moves": float(
                    np.mean([r["rejected_cyclic_moves"] for r in sub])
                ),
            }
        )
    out.sort(key=lambda r: r["dev_mean_best_rpd"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase B benchmark pipeline.")
    parser.add_argument("--budget-multiplier", type=int, default=100)
    parser.add_argument("--replicates", type=int, default=NUM_REPLICATES)
    args = parser.parse_args()

    started_utc = datetime.now(tz=UTC).isoformat()
    if not FROZEN_A_JSON.exists():
        raise FileNotFoundError(
            f"Frozen Phase A winner missing at {FROZEN_A_JSON}. Run Phase A before Phase B."
        )
    frozen_a = json.loads(FROZEN_A_JSON.read_text("utf-8"))
    frozen_a_id = str(frozen_a["winner_config_id"])
    frozen_a_sha = str(frozen_a["winner_lock_sha256"])

    suite_doc = json.loads(SUITE_JSON.read_text("utf-8"))
    common_instances = [r for r in suite_doc["instances"] if r["in_common_pipeline_suite"]]

    specs = list_phase_b_configs()
    matrix_cfg = load_phase_b_matrix()
    master_seeds = tuple(int(s) for s in get_frozen_master_seeds())
    rng_meta = get_rng_metadata()

    for sub in ("raw", "manifests", "analysis", "winner"):
        (PHASE_B_ROOT / sub).mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting Phase B across {len(common_instances)} Common Suite instances x "
        f"{len(specs)} active configs x {args.replicates} frozen seeds "
        f"(matched start A*={frozen_a_id}, budget={args.budget_multiplier}*L)...",
        flush=True,
    )
    raw_path = PHASE_B_ROOT / "raw" / "phase_b_common_32.jsonl"
    all_rows = _run_phase_b_sweep(
        common_instances,
        specs,
        frozen_a_id,
        master_seeds,
        args.budget_multiplier,
        args.replicates,
        raw_path,
    )

    dev_rows = [r for r in all_rows if r["split"] == "development"]
    ho_rows = [r for r in all_rows if r["split"] == "held_out"]

    dev_summary, dev_stats = _summarize_split(dev_rows, specs)
    ho_summary, ho_stats = _summarize_split(ho_rows, specs)
    nb_marginal = _summarize_marginal(all_rows, "neighborhood_id")
    st_marginal = _summarize_marginal(all_rows, "strategy_name")

    _write_csv(PHASE_B_ROOT / "analysis" / "phase_b_dev_rankings.csv", dev_summary)
    _write_json(PHASE_B_ROOT / "analysis" / "phase_b_dev_rankings.json", dev_summary)
    _write_csv(PHASE_B_ROOT / "analysis" / "phase_b_held_out_evaluation.csv", ho_summary)
    _write_json(PHASE_B_ROOT / "analysis" / "phase_b_held_out_evaluation.json", ho_summary)
    _write_csv(PHASE_B_ROOT / "analysis" / "phase_b_neighborhood_marginal.csv", nb_marginal)
    _write_json(PHASE_B_ROOT / "analysis" / "phase_b_neighborhood_marginal.json", nb_marginal)
    _write_csv(PHASE_B_ROOT / "analysis" / "phase_b_strategy_marginal.csv", st_marginal)
    _write_json(PHASE_B_ROOT / "analysis" / "phase_b_strategy_marginal.json", st_marginal)

    stats_payload = {
        "schema_version": "jssp-phase-b-stats-v1",
        "excluded_configs_d04_a": list(matrix_cfg.excluded_config_ids),
        "development_16": dev_stats,
        "held_out_16": ho_stats,
    }
    _write_json(PHASE_B_ROOT / "analysis" / "phase_b_statistical_tests.json", stats_payload)
    _write_csv(
        PHASE_B_ROOT / "analysis" / "phase_b_statistical_tests.csv",
        [dict(r, split="development") for r in dev_stats["paired_wilcoxon_holm"]]
        + [dict(r, split="held_out") for r in ho_stats["paired_wilcoxon_holm"]],
    )

    # Pre-registered Winner Selection strictly on Development Split (16 instances)
    dev_winner_row = dev_summary[0]
    winner_id = str(dev_winner_row["config_id"])
    winner_matrix_cfg = next(c for c in matrix_cfg.configs if c.config_id == winner_id)
    winner_ho_row = next(r for r in ho_summary if r["config_id"] == winner_id)

    winner_body: dict[str, Any] = {
        "schema_version": "jssp-frozen-winner-b-v1",
        "frozen_at_utc": datetime.now(tz=UTC).isoformat(),
        "transferred_frozen_winner_a": {
            "config_id": frozen_a_id,
            "winner_lock_sha256": frozen_a_sha,
        },
        "selection_protocol": (
            "Selected strictly on the 16-instance Development Split by lowest mean instance-level "
            "median RPD_BKS within equal B=100*L candidate evaluation budget, breaking ties by "
            "dev Friedman rank, anytime RPD AUC, and runtime."
        ),
        "winner_config_id": winner_id,
        "winner_strategy": dev_winner_row["strategy_name"],
        "winner_neighborhood": dev_winner_row["neighborhood_id"],
        "winner_parameters": winner_matrix_cfg.model_dump(mode="json"),
        "development_metrics": dev_winner_row,
        "held_out_validation_metrics": winner_ho_row,
    }
    winner_body["winner_lock_sha256"] = canonical_json_sha256(winner_body)
    _write_json(PHASE_B_ROOT / "winner" / "frozen_winner_b.json", winner_body)

    git_prov = collect_git_provenance(REPO_ROOT)
    env_prov = collect_environment_provenance(REPO_ROOT)
    manifest_doc = {
        "schema_version": "jssp-phase-b-manifest-v1",
        "phase": "phase_b",
        "started_utc": started_utc,
        "completed_utc": datetime.now(tz=UTC).isoformat(),
        "git": git_prov.model_dump(mode="json"),
        "environment": env_prov.model_dump(mode="json"),
        "rng": {
            "algorithm": rng_meta.algorithm,
            "seed_schedule_id": rng_meta.master_seed_schedule_id,
            "master_seeds_v1": list(master_seeds),
            "pairing_mode": "paired_init",
            "pairing_group": "phase_b_matched_start",
        },
        "frozen_winner_a_id": frozen_a_id,
        "frozen_winner_a_sha256": frozen_a_sha,
        "active_configs": len(specs),
        "excluded_configs_d04_a": list(matrix_cfg.excluded_config_ids),
        "budget_multiplier_L": args.budget_multiplier,
        "total_runs": len(all_rows),
        "selected_benchmark_suite_sha256": _sha256_file(SUITE_JSON),
        "raw_jsonl_sha256": _sha256_file(raw_path),
        "frozen_winner_b_sha256": winner_body["winner_lock_sha256"],
    }
    _write_json(PHASE_B_ROOT / "manifests" / "phase_b_manifest.json", manifest_doc)

    # Concise Markdown Report
    dev_chi2 = dev_stats["friedman_med_rpd"]["statistic"]
    dev_p = dev_stats["friedman_med_rpd"]["p_value"]
    dev_w = dev_stats["friedman_med_rpd"]["kendall_w"]
    ho_chi2 = ho_stats["friedman_med_rpd"]["statistic"]
    ho_p = ho_stats["friedman_med_rpd"]["p_value"]
    lines = [
        "# Phase B Summary — Local Search & Metaheuristics (`B01`–`B24`, `B29`–`B32`)",
        "",
        (
            f"- **Frozen Starting Initializer ($A^*$):** `{frozen_a_id}` "
            f"(matched starting schedules per instance $\\times$ frozen seed)."
        ),
        (
            f"- **Selected & Frozen Winner ($B^*$):** **`{winner_id}` — "
            f"{dev_winner_row['strategy_name']} + {dev_winner_row['neighborhood_id']}**"
        ),
        (
            "- **Development Split Winner Selection (16 instances):** Mean instance-level "
            f"median `RPD_BKS` = `{dev_winner_row['median_rpd_mean']:.3f}%`, Mean Best `RPD_BKS` = "
            f"`{dev_winner_row['best_rpd_mean']:.3f}%`, Friedman rank = "
            f"`{dev_winner_row['friedman_rank']:.2f}` / 28."
        ),
        (
            "- **Held-Out Generalization ($B^*$, 16 held-out instances):** Mean instance-level "
            f"median `RPD_BKS` = `{winner_ho_row['median_rpd_mean']:.3f}%`, Mean Best `RPD_BKS` = "
            f"`{winner_ho_row['best_rpd_mean']:.3f}%`, Mean Improvement = "
            f"`{winner_ho_row['improvement_pct_mean']:.2f}%`."
        ),
        (
            f"- **Experimental Volume & Protocol:** `{len(all_rows):,}` runs (`32` Common Suite "
            f"instances $\\times$ `28` active configs $\\times$ `{args.replicates}` frozen seeds; "
            f"budget `{args.budget_multiplier} * L`; `B25`–`B28` excluded per `D04-A`)."
        ),
        (
            f"- **Blocked Friedman Test:** Dev $\\chi^2_F = {dev_chi2:.2f}$ ($p = {dev_p:.3e}$, "
            f"Kendall $W = {dev_w:.4f}$); Held-Out $\\chi^2_F = {ho_chi2:.2f}$ ($p = {ho_p:.3e}$)."
        ),
        "",
        "## All 28 Active Phase B Configurations — Development & Held-Out Summary",
        "",
        (
            "| Rank (Dev) | ID | Strategy | Nbh | Dev Med RPD | Dev Best RPD | Dev Rank | "
            "Held-Out Med RPD | Held-Out Best RPD | Anytime RPD AUC (Dev) | Mean ms |"
        ),
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    ho_by_id = {r["config_id"]: r for r in ho_summary}
    for rank, drow in enumerate(dev_summary, start=1):
        cid = drow["config_id"]
        hrow = ho_by_id[cid]
        lines.append(
            f"| {rank} | `{cid}` | {drow['strategy_name']} | `{drow['neighborhood_id']}` | "
            f"{drow['median_rpd_mean']:.3f}% | {drow['best_rpd_mean']:.3f}% | "
            f"{drow['friedman_rank']:.2f} | {hrow['median_rpd_mean']:.3f}% | "
            f"{hrow['best_rpd_mean']:.3f}% | {drow['anytime_rpd_auc_mean']:.3f}% | "
            f"{drow['mean_elapsed_ms']:.1f} |"
        )
    lines.extend(["", "## Neighborhood Marginal Comparison (`N5`–`N8`)", ""])
    lines.extend(
        [
            "| Neighborhood | Dev Best RPD | Held-Out Best RPD | Mean Improvement | "
            "Cycle Rejects/Run | Mean ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in nb_marginal:
        lines.append(
            f"| `{row['neighborhood_id']}` | {row['dev_mean_best_rpd']:.3f}% | "
            f"{row['ho_mean_best_rpd']:.3f}% | {row['mean_improvement_pct']:.2f}% | "
            f"{row['mean_rejected_cyclic_moves']:.1f} | {row['mean_elapsed_ms']:.1f} |"
        )

    win_uri = (PHASE_B_ROOT / "winner/frozen_winner_b.json").as_posix()
    man_uri = (PHASE_B_ROOT / "manifests/phase_b_manifest.json").as_posix()
    st_uri = (PHASE_B_ROOT / "analysis/phase_b_statistical_tests.json").as_posix()
    lines.extend(
        [
            "",
            "## Key Artifacts",
            f"- Frozen Winner $B^*$: [`results/phase_b/winner/frozen_winner_b.json`](file://{win_uri})",
            f"- Manifest: [`results/phase_b/manifests/phase_b_manifest.json`](file://{man_uri})",
            "- Statistical Tests: "
            f"[`results/phase_b/analysis/phase_b_statistical_tests.json`](file://{st_uri})",
        ]
    )
    (REPORTS_ROOT / "phase_b_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"\nPhase B COMPLETE. Frozen Winner B*: {winner_id} "
        f"({dev_winner_row['strategy_name']} + {dev_winner_row['neighborhood_id']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
