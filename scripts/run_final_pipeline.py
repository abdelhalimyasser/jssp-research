"""Execute Final Integrated Optimization Pipeline (P1-P5) and generate master report."""

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
from jssp_research.experiments.manifests import canonical_json_sha256
from jssp_research.native import (
    decode_obr,
    default_phase_c_budget,
    get_frozen_master_seeds,
    get_rng_metadata,
    run_integrated_acb_batch,
    run_phase_a_batch,
    run_phase_b_batch,
    run_phase_c_batch,
    validate_schedule,
)
from jssp_research.utils.provenance import (
    collect_environment_provenance,
    collect_git_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_JSON = REPO_ROOT / "data/metadata/selected_benchmark_suite.json"
FROZEN_A_JSON = REPO_ROOT / "results/phase_a/winner/frozen_winner_a.json"
FROZEN_B_JSON = REPO_ROOT / "results/phase_b/winner/frozen_winner_b.json"
FROZEN_C_JSON = REPO_ROOT / "results/phase_c/winner/frozen_winner_c.json"
FINAL_ROOT = REPO_ROOT / "results/final_pipeline"
REPORTS_ROOT = REPO_ROOT / "reports"
NUM_REPLICATES = 20

PIPELINE_IDS = (
    "P1_Baseline_A01_C01",
    "P2_FrozenA_C01",
    "P3_FrozenA_FrozenB",
    "P4_FrozenA_FrozenC",
    "P5_Integrated_A_C_B",
)


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


def _make_row(
    inst: dict[str, Any],
    pid: str,
    rep: int,
    seed: int,
    budget: int,
    evals_used: int,
    best_cmax: int,
    best_found_eval: int,
    anytime_cmax_auc: float,
    elapsed_ms: float,
    stage_c_best: int | None = None,
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
        "pipeline_id": pid,
        "replicate_index": rep,
        "master_seed": seed,
        "evaluation_budget": budget,
        "evals_used": evals_used,
        "stage_c_best_makespan": stage_c_best,
        "best_makespan": best_cmax,
        "best_rpd_bks": relative_percentage_deviation(best_cmax, bks),
        "best_gap_lb": lower_bound_gap(best_cmax, lb),
        "hit_bks": best_cmax <= bks,
        "best_found_eval": best_found_eval,
        "feasible_final": True,
        "anytime_cmax_auc": anytime_cmax_auc,
        "anytime_rpd_auc": relative_percentage_deviation(anytime_cmax_auc, bks),
        "total_elapsed_ms": elapsed_ms,
    }


def _summarize_split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    instance_ids = sorted({r["instance_id"] for r in rows})
    by_pid_inst: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        by_pid_inst.setdefault((r["pipeline_id"], r["instance_id"]), []).append(r)

    pids = list(PIPELINE_IDS)
    med_rpd_mat = np.zeros((len(instance_ids), len(pids)), dtype=float)
    scores: dict[str, np.ndarray] = {}
    summary: list[dict[str, Any]] = []

    for c_idx, pid in enumerate(pids):
        inst_med_rpds, inst_mean_rpds, inst_best_rpds = [], [], []
        inst_med_cmax, inst_best_cmax, inst_std_cmax = [], [], []
        inst_hit, inst_auc, inst_ms = [], [], []
        for i_idx, iid in enumerate(instance_ids):
            grp = by_pid_inst[(pid, iid)]
            rpds = np.array([g["best_rpd_bks"] for g in grp], dtype=float)
            cmax = np.array([g["best_makespan"] for g in grp], dtype=float)
            sc = summarize_numeric(cmax)
            med_r = float(np.median(rpds))
            med_rpd_mat[i_idx, c_idx] = med_r
            inst_med_rpds.append(med_r)
            inst_mean_rpds.append(float(np.mean(rpds)))
            inst_best_rpds.append(float(np.min(rpds)))
            inst_med_cmax.append(sc["median"])
            inst_best_cmax.append(sc["min"])
            inst_std_cmax.append(sc["std"])
            inst_hit.append(float(np.mean([1.0 if g["hit_bks"] else 0.0 for g in grp])))
            inst_auc.append(float(np.median([g["anytime_rpd_auc"] for g in grp])))
            inst_ms.append(float(np.mean([g["total_elapsed_ms"] for g in grp])))

        scores[pid] = med_rpd_mat[:, c_idx].copy()
        summary.append(
            {
                "pipeline_id": pid,
                "n_instances": len(instance_ids),
                "n_runs": sum(len(by_pid_inst[(pid, iid)]) for iid in instance_ids),
                "feasible_rate": 1.0,
                "best_rpd_mean": float(np.mean(inst_best_rpds)),
                "mean_rpd_mean": float(np.mean(inst_mean_rpds)),
                "median_rpd_mean": float(np.mean(inst_med_rpds)),
                "best_cmax_mean": float(np.mean(inst_best_cmax)),
                "median_cmax_mean": float(np.mean(inst_med_cmax)),
                "within_instance_std_mean": float(np.mean(inst_std_cmax)),
                "anytime_rpd_auc_mean": float(np.mean(inst_auc)),
                "hit_bks_rate": float(np.mean(inst_hit)),
                "mean_elapsed_ms": float(np.mean(inst_ms)),
            }
        )

    friedman = run_friedman_analysis(pids, med_rpd_mat)
    for row in summary:
        row["friedman_rank"] = friedman.mean_ranks[row["pipeline_id"]]

    pairs = [
        ("progressive_pipeline", "P2_FrozenA_C01", "P1_Baseline_A01_C01"),
        ("progressive_pipeline", "P3_FrozenA_FrozenB", "P2_FrozenA_C01"),
        ("progressive_pipeline", "P4_FrozenA_FrozenC", "P2_FrozenA_C01"),
        ("progressive_pipeline", "P5_Integrated_A_C_B", "P1_Baseline_A01_C01"),
        ("progressive_pipeline", "P5_Integrated_A_C_B", "P2_FrozenA_C01"),
        ("progressive_pipeline", "P5_Integrated_A_C_B", "P3_FrozenA_FrozenB"),
        ("progressive_pipeline", "P5_Integrated_A_C_B", "P4_FrozenA_FrozenC"),
    ]
    wilcoxon = paired_comparisons_table(scores, pairs, lower_is_better=True)
    return summary, {
        "n_instances": len(instance_ids),
        "friedman_med_rpd": {
            "statistic": friedman.statistic,
            "p_value": friedman.p_value,
            "kendall_w": friedman.kendall_w,
            "mean_ranks": friedman.mean_ranks,
        },
        "paired_wilcoxon_holm": wilcoxon,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Final Integrated Pipeline P1-P5.")
    parser.add_argument("--replicates", type=int, default=NUM_REPLICATES)
    args = parser.parse_args()

    started_utc = datetime.now(tz=UTC).isoformat()
    frozen_a = json.loads(FROZEN_A_JSON.read_text("utf-8"))
    frozen_b = json.loads(FROZEN_B_JSON.read_text("utf-8"))
    frozen_c = json.loads(FROZEN_C_JSON.read_text("utf-8"))

    a_id = str(frozen_a["winner_config_id"])
    b_id = str(frozen_b["winner_config_id"])
    b_nb = str(frozen_b["winner_neighborhood"])
    c_id = str(frozen_c["winner_config_id"])
    c_embedded_ls_strategy = "steepest_descent"

    suite_doc = json.loads(SUITE_JSON.read_text("utf-8"))
    common_instances = [r for r in suite_doc["instances"] if r["in_common_pipeline_suite"]]
    master_seeds = tuple(int(s) for s in get_frozen_master_seeds())
    rng_meta = get_rng_metadata()

    for sub in ("raw", "manifests", "analysis", "winner"):
        (FINAL_ROOT / sub).mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting Final Integrated Pipeline (P1-P5) across {len(common_instances)} instances x "
        f"{len(PIPELINE_IDS)} pipelines x {args.replicates} seeds "
        f"(A*={a_id}, B*={b_id}, C*={c_id})...",
        flush=True,
    )

    reps = list(range(args.replicates))
    raw_path = FINAL_ROOT / "raw" / "final_pipeline_common_32.jsonl"
    all_rows: list[dict[str, Any]] = []

    with raw_path.open("w", encoding="utf-8") as fh:
        for idx, inst in enumerate(common_instances, start=1):
            t0 = time.perf_counter()
            parsed = parse_jssp_file(
                REPO_ROOT / inst["raw_relative_path"], instance_id=inst["instance_id"]
            )
            budget = default_phase_c_budget(parsed.num_jobs, parsed.num_machines)

            p1_tasks = [
                ("A01", r, None, "paired_init", "final_pipeline_shared_init", "P1_Baseline_A01_C01")
                for r in reps
            ]
            p2_tasks = [
                (a_id, r, None, "paired_init", "final_pipeline_shared_init", "P2_FrozenA_C01")
                for r in reps
            ]
            p3_tasks = [
                (b_id, r, None, "paired_init", "final_pipeline_shared_init", "P3_FrozenA_FrozenB")
                for r in reps
            ]
            p4_tasks = [
                (c_id, r, None, "paired_init", "final_pipeline_shared_init", "P4_FrozenA_FrozenC")
                for r in reps
            ]

            r1 = run_phase_a_batch(
                parsed.jobs,
                inst["instance_id"],
                p1_tasks,
                evaluation_budget=budget,
                trajectory_mode="none",
            )
            r2 = run_phase_a_batch(
                parsed.jobs,
                inst["instance_id"],
                p2_tasks,
                evaluation_budget=budget,
                trajectory_mode="none",
            )
            r3 = run_phase_b_batch(
                parsed.jobs,
                inst["instance_id"],
                p3_tasks,
                initializer_id=a_id,
                evaluation_budget=budget,
            )
            r4 = run_phase_c_batch(
                parsed.jobs,
                inst["instance_id"],
                p4_tasks,
                initializer_id=a_id,
                ls_neighborhood=b_nb,
                ls_strategy=c_embedded_ls_strategy,
                max_ls_evals=50,
                evaluation_budget=budget,
            )
            r5 = run_integrated_acb_batch(
                parsed.jobs,
                inst["instance_id"],
                reps,
                frozen_a_id=a_id,
                frozen_b_id=b_id,
                frozen_c_id=c_id,
                max_ls_evals=50,
                evaluation_budget=budget,
                stage_c_fraction=0.85,
            )

            inst_rows: list[dict[str, Any]] = []
            for rep, x in zip(reps, r1, strict=True):
                # Independent M2 check on replicate 0
                if rep == 0:
                    s = decode_obr(parsed.jobs, x.best_chromosome)
                    validate_schedule(parsed.jobs, s)
                inst_rows.append(
                    _make_row(
                        inst,
                        "P1_Baseline_A01_C01",
                        rep,
                        master_seeds[rep],
                        budget,
                        x.evals_used,
                        x.best_makespan,
                        x.best_found_eval,
                        x.anytime_cmax_auc,
                        x.total_elapsed_ns / 1e6,
                    )
                )
            for rep, x in zip(reps, r2, strict=True):
                inst_rows.append(
                    _make_row(
                        inst,
                        "P2_FrozenA_C01",
                        rep,
                        master_seeds[rep],
                        budget,
                        x.evals_used,
                        x.best_makespan,
                        x.best_found_eval,
                        x.anytime_cmax_auc,
                        x.total_elapsed_ns / 1e6,
                    )
                )
            for rep, x in zip(reps, r3, strict=True):
                inst_rows.append(
                    _make_row(
                        inst,
                        "P3_FrozenA_FrozenB",
                        rep,
                        master_seeds[rep],
                        budget,
                        x.evals_used,
                        x.best_makespan,
                        x.best_found_eval,
                        x.anytime_cmax_auc,
                        x.total_elapsed_ns / 1e6,
                    )
                )
            for rep, x in zip(reps, r4, strict=True):
                if rep == 0:
                    s = decode_obr(parsed.jobs, x.best_chromosome)
                    validate_schedule(parsed.jobs, s)
                inst_rows.append(
                    _make_row(
                        inst,
                        "P4_FrozenA_FrozenC",
                        rep,
                        master_seeds[rep],
                        budget,
                        x.evals_used,
                        x.best_makespan,
                        x.best_found_eval,
                        x.anytime_cmax_auc,
                        x.total_elapsed_ns / 1e6,
                    )
                )
            for rep, x in zip(reps, r5, strict=True):
                inst_rows.append(
                    _make_row(
                        inst,
                        "P5_Integrated_A_C_B",
                        rep,
                        master_seeds[rep],
                        budget,
                        x.evals_used,
                        x.final_best_makespan,
                        x.final_best_eval,
                        x.anytime_cmax_auc,
                        x.total_elapsed_ns / 1e6,
                        stage_c_best=x.stage_c_best_makespan,
                    )
                )

            for row in inst_rows:
                all_rows.append(row)
                fh.write(json.dumps(row) + "\n")

            dt = time.perf_counter() - t0
            best_row = min(inst_rows, key=lambda r: r["best_makespan"])
            print(
                f"  [Final {idx:2d}/{len(common_instances)}] {inst['instance_id']:6s} "
                f"({inst['split']:8s}, {inst['jobs']:2d}x{inst['machines']:2d}, B={budget:5d}): "
                f"100 runs in {dt:6.2f}s | best Cmax={best_row['best_makespan']} "
                f"(BKS={inst['best_known_value']}, {best_row['pipeline_id']})",
                flush=True,
            )

    dev_rows = [r for r in all_rows if r["split"] == "development"]
    ho_rows = [r for r in all_rows if r["split"] == "held_out"]

    dev_sum, dev_stats = _summarize_split(dev_rows)
    ho_sum, ho_stats = _summarize_split(ho_rows)

    _write_csv(FINAL_ROOT / "analysis" / "final_dev_summary.csv", dev_sum)
    _write_json(FINAL_ROOT / "analysis" / "final_dev_summary.json", dev_sum)
    _write_csv(FINAL_ROOT / "analysis" / "final_held_out_summary.csv", ho_sum)
    _write_json(FINAL_ROOT / "analysis" / "final_held_out_summary.json", ho_sum)

    # Family breakdown across all 32 instances
    fam_rows: list[dict[str, Any]] = []
    for fam in sorted({r["family"] for r in all_rows}):
        for pid in PIPELINE_IDS:
            sub = [r for r in all_rows if r["family"] == fam and r["pipeline_id"] == pid]
            fam_rows.append(
                {
                    "family": fam,
                    "pipeline_id": pid,
                    "n_runs": len(sub),
                    "best_rpd_mean": float(np.mean([r["best_rpd_bks"] for r in sub])),
                    "median_cmax_mean": float(np.median([r["best_makespan"] for r in sub])),
                    "hit_bks_rate": float(np.mean([1.0 if r["hit_bks"] else 0.0 for r in sub])),
                }
            )
    _write_csv(FINAL_ROOT / "analysis" / "final_family_breakdown.csv", fam_rows)
    _write_json(FINAL_ROOT / "analysis" / "final_family_breakdown.json", fam_rows)

    stats_payload = {
        "schema_version": "jssp-final-pipeline-stats-v1",
        "development_16": dev_stats,
        "held_out_16": ho_stats,
    }
    _write_json(FINAL_ROOT / "analysis" / "final_statistical_tests.json", stats_payload)
    _write_csv(
        FINAL_ROOT / "analysis" / "final_statistical_tests.csv",
        [dict(r, split="development") for r in dev_stats["paired_wilcoxon_holm"]]
        + [dict(r, split="held_out") for r in ho_stats["paired_wilcoxon_holm"]],
    )

    dev_sorted = sorted(dev_sum, key=lambda r: (r["median_rpd_mean"], r["friedman_rank"]))
    best_pipeline_id = dev_sorted[0]["pipeline_id"]
    ho_best_row = next(r for r in ho_sum if r["pipeline_id"] == best_pipeline_id)

    final_winner_doc: dict[str, Any] = {
        "schema_version": "jssp-final-pipeline-winner-v1",
        "frozen_at_utc": datetime.now(tz=UTC).isoformat(),
        "components": {"A_star": a_id, "B_star": b_id, "C_star": c_id},
        "best_overall_pipeline": best_pipeline_id,
        "development_metrics": dev_sorted[0],
        "held_out_validation_metrics": ho_best_row,
    }
    final_winner_doc["winner_lock_sha256"] = canonical_json_sha256(final_winner_doc)
    _write_json(FINAL_ROOT / "winner" / "frozen_final_pipeline.json", final_winner_doc)

    git_prov = collect_git_provenance(REPO_ROOT)
    env_prov = collect_environment_provenance(REPO_ROOT)
    manifest_doc = {
        "schema_version": "jssp-final-pipeline-manifest-v1",
        "phase": "final_integrated_pipeline",
        "started_utc": started_utc,
        "completed_utc": datetime.now(tz=UTC).isoformat(),
        "git": git_prov.model_dump(mode="json"),
        "environment": env_prov.model_dump(mode="json"),
        "rng": {
            "algorithm": rng_meta.algorithm,
            "seed_schedule_id": rng_meta.master_seed_schedule_id,
            "master_seeds_v1": list(master_seeds),
        },
        "components": {
            "frozen_winner_a": a_id,
            "frozen_winner_b": b_id,
            "frozen_winner_c": c_id,
        },
        "total_runs": len(all_rows),
        "raw_jsonl_sha256": _sha256_file(raw_path),
        "final_winner_sha256": final_winner_doc["winner_lock_sha256"],
    }
    _write_json(FINAL_ROOT / "manifests" / "final_pipeline_manifest.json", manifest_doc)

    # Master Final Research Results Markdown Report referencing frozen verified data
    pa_sum_uri = (REPORTS_ROOT / "phase_a_summary.md").as_posix()
    pb_sum_uri = (REPORTS_ROOT / "phase_b_summary.md").as_posix()
    pc_sum_uri = (REPORTS_ROOT / "phase_c_summary.md").as_posix()
    fin_man_uri = (FINAL_ROOT / "manifests/final_pipeline_manifest.json").as_posix()
    fin_win_uri = (FINAL_ROOT / "winner/frozen_final_pipeline.json").as_posix()

    fa_dev_rpd = frozen_a["development_metrics"]["ga_median_rpd_mean"]
    fa_ho_rpd = frozen_a["held_out_validation_metrics"]["ga_median_rpd_mean"]
    fb_dev_rpd = frozen_b["development_metrics"]["median_rpd_mean"]
    fb_ho_rpd = frozen_b["held_out_validation_metrics"]["median_rpd_mean"]
    fc_dev_rpd = frozen_c["development_metrics"]["median_rpd_mean"]
    fc_ho_rpd = frozen_c["held_out_validation_metrics"]["median_rpd_mean"]

    lines = [
        "# Final Research Results — Classical JSSP Sequential Optimization Pipeline",
        "",
        "## 1. Experimental Overview & Active Matrix Scope",
        (
            "- **Framework:** Deterministic classical rectangular JSSP ($\\min C_{\\max}$), OBR "
            "chromosomes, deterministic active Serial SGS decoder (`D01-A`), exact `O(L)` DAG "
            "evaluation (`D03-C`), 20 frozen SHA-256 seeds (`FROZEN_MASTER_SEEDS_V1`), strict "
            "Dev (`16`) vs. Held-Out (`16`) split inside the 32-instance Common Suite."
        ),
        "- **Approved Active Matrix (63 Configurations):**",
        (
            "  - **Phase A (`A01`–`A24`, 24 configs):** Initial population generation (PRNG, "
            "Deterministic MWR, Halton, Scrambled Halton, Sobol, Scrambled Sobol, Mixed QMC, "
            "MWR-biased QMC)."
        ),
        (
            "  - **Phase B (`B01`–`B24`, `B29`–`B32`, 28 active configs):** Critical-block "
            "disjunctive-graph neighborhoods (`N5`–`N8`) crossed with Steepest/First/Random "
            "Improvement, Tabu Search, ILS, SA, and Hybrid Metropolis-Filtered ILS "
            "(`B25`–`B28` excluded per `D04-A`)."
        ),
        (
            "  - **Phase C (`C01`–`C11`, 11 configs):** Classical GA, LGA, BGA, PLGA, "
            "Fixed/Adaptive Model A Dual-View GAs, unpaired dual-population control, and "
            "exploratory migration."
        ),
        "",
        "## 2. Sequential Frozen Component Winners",
        (
            f"- **Frozen Phase A Winner ($A^*$):** `{a_id}` — {frozen_a['winner_config_name']} "
            f"(Dev median GA RPD = `{fa_dev_rpd:.3f}%`, "
            f"Held-Out median GA RPD = `{fa_ho_rpd:.3f}%`)"
        ),
        (
            f"- **Frozen Phase B Winner ($B^*$):** `{b_id}` — {frozen_b['winner_strategy']} + "
            f"{frozen_b['winner_neighborhood']} (Dev median RPD = `{fb_dev_rpd:.3f}%`, "
            f"Held-Out median RPD = `{fb_ho_rpd:.3f}%`)"
        ),
        (
            f"- **Frozen Phase C Winner ($C^*$):** `{c_id}` — {frozen_c['winner_config_name']} "
            f"(Dev median RPD = `{fc_dev_rpd:.3f}%`, "
            f"Held-Out median RPD = `{fc_ho_rpd:.3f}%`)"
        ),
        (
            f"- **Best Integrated Pipeline Variant:** `{best_pipeline_id}` (Dev median RPD = "
            f"`{dev_sorted[0]['median_rpd_mean']:.3f}%`, Held-Out median RPD = "
            f"`{ho_best_row['median_rpd_mean']:.3f}%`)"
        ),
        "",
        "## 3. Integrated Pipeline Comparison (`P1`–`P5`, Equalized Evaluation Budget)",
        "",
        (
            "| Pipeline | Dev Med RPD | Dev Best RPD | Dev Friedman Rank | "
            "Held-Out Med RPD | Held-Out Best RPD | Held-Out Hit BKS | Mean ms |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ho_by_pid = {r["pipeline_id"]: r for r in ho_sum}
    for drow in dev_sum:
        pid = drow["pipeline_id"]
        hrow = ho_by_pid[pid]
        lines.append(
            f"| `{pid}` | {drow['median_rpd_mean']:.3f}% | {drow['best_rpd_mean']:.3f}% | "
            f"{drow['friedman_rank']:.2f} | {hrow['median_rpd_mean']:.3f}% | "
            f"{hrow['best_rpd_mean']:.3f}% | {hrow['hit_bks_rate']:.3f} | "
            f"{drow['mean_elapsed_ms']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## 4. Planned Statistical Contrasts (Held-Out 16 Instances, Holm-Adjusted)",
            "",
            "| Contrast | p (raw) | p (Holm) | Sig (0.05) | Vargha-Delaney A12 | Cliff's d |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for wr in ho_stats["paired_wilcoxon_holm"]:
        lines.append(
            f"| `{wr['method_a']}` vs `{wr['method_b']}` | {wr['p_value_raw']:.3e} | "
            f"{wr['p_value_holm']:.3e} | {wr['significant_0_05']} | "
            f"{wr['vargha_delaney_a12']:.3f} | {wr['cliffs_delta']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 5. Linked Phase Summaries & Frozen Artifacts",
            f"- Phase A Summary: [`reports/phase_a_summary.md`](file://{pa_sum_uri})",
            f"- Phase B Summary: [`reports/phase_b_summary.md`](file://{pb_sum_uri})",
            f"- Phase C Summary: [`reports/phase_c_summary.md`](file://{pc_sum_uri})",
            (
                "- Final Pipeline Manifest: "
                f"[`results/final_pipeline/manifests/final_pipeline_manifest.json`](file://{fin_man_uri})"
            ),
            (
                "- Final Winner Lock: "
                f"[`results/final_pipeline/winner/frozen_final_pipeline.json`](file://{fin_win_uri})"
            ),
        ]
    )
    (REPORTS_ROOT / "final_research_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"\nFinal Pipeline COMPLETE. Best overall pipeline: {best_pipeline_id}")


if __name__ == "__main__":
    main()
