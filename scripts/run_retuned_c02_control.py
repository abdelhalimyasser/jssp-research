"""Execute the matched single-view C02_RETUNED_SA control experiment (Dev -> Lock -> Held-Out)."""

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
)
from jssp_research.benchmarks import parse_jssp_file
from jssp_research.experiments.manifests import canonical_json_sha256
from jssp_research.native import (
    default_phase_c_budget,
    get_frozen_master_seeds,
    get_rng_metadata,
    run_dual_view_exploratory_batch,
)
from jssp_research.utils.provenance import (
    collect_environment_provenance,
    collect_git_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_JSON = REPO_ROOT / "data/metadata/selected_benchmark_suite.json"
PHASE_C_RAW_JSONL = REPO_ROOT / "results/phase_c/raw/phase_c_common_32.jsonl"
DV_RAW_JSONL = REPO_ROOT / "results/exploratory_dv_ls/raw/exploratory_dv_ls_runs.jsonl"
DV_SEL_JSON = REPO_ROOT / "results/exploratory_dv_ls/selection/frozen_dv_ls_selection.json"
EXP_ROOT = REPO_ROOT / "results/exploratory_dv_ls"
NUM_REPLICATES = 20

COMPARISON_CONTRASTS = [
    ("matched_tuned_dv_vs_c02", "DV_STAR", "C02_RETUNED_SA"),
    ("retuned_c02_vs_frozen_c02", "C02_RETUNED_SA", "C02_LGA"),
    ("retuned_c02_vs_ctrl_c06", "C02_RETUNED_SA", "DV_CTRL_C06"),
    ("dv_star_vs_frozen_c02", "DV_STAR", "C02_LGA"),
    ("dv_star_vs_ctrl_c06", "DV_STAR", "DV_CTRL_C06"),
    ("ctrl_c06_vs_frozen_c02", "DV_CTRL_C06", "C02_LGA"),
]


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
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _row_from_record(
    inst: dict[str, Any],
    replicate: int,
    seed: int,
    budget: int,
    rec: Any,
    *,
    stage: str,
) -> dict[str, Any]:
    bks = int(inst["best_known_value"])
    lb = inst["lower_bound"]
    lam = rec.lam_branch
    bald = rec.bald_branch
    tot_ls_evals = lam.ls_evals + bald.ls_evals
    ls_share = 100.0 * float(tot_ls_evals) / float(rec.evals_used) if rec.evals_used > 0 else 0.0
    return {
        "stage": stage,
        "experiment_id": rec.experiment_id,
        "base_config_id": rec.base_config_id,
        "initializer_id": rec.initializer_id,
        "ls_neighborhood": rec.ls_neighborhood,
        "ls_strategy": rec.ls_strategy,
        "max_ls_evals": rec.max_ls_evals,
        "candidate_fraction": rec.candidate_fraction,
        "genotype_capacity": rec.genotype_capacity,
        "instance_id": inst["instance_id"],
        "family": inst["benchmark_family"],
        "split": inst["split"],
        "jobs": inst["jobs"],
        "machines": inst["machines"],
        "operation_count": inst["operation_count"],
        "bks": bks,
        "lb": lb,
        "replicate_index": replicate,
        "master_seed": seed,
        "evaluation_budget": budget,
        "evals_used": rec.evals_used,
        "init_evals": rec.init_evals,
        "search_evals": rec.search_evals,
        "decode_evals": lam.decode_evals + bald.decode_evals,
        "ls_evals": tot_ls_evals,
        "ls_budget_share_pct": ls_share,
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
        "lam_decode_evals": lam.decode_evals,
        "lam_ls_evals": lam.ls_evals,
        "lam_ls_calls": lam.ls_calls,
        "lam_improving_calls": lam.improving_calls,
        "lam_evaluated_moves": lam.evaluated_moves,
        "lam_feasible_moves": lam.feasible_moves,
        "lam_rejected_cyclic_moves": lam.rejected_cyclic_moves,
        "lam_improving_moves": lam.improving_moves,
        "lam_accepted_moves": lam.accepted_moves,
        "lam_rejected_moves": lam.rejected_moves,
        "bald_decode_evals": bald.decode_evals,
        "bald_ls_evals": bald.ls_evals,
        "bald_ls_calls": bald.ls_calls,
        "bald_improving_calls": bald.improving_calls,
        "bald_evaluated_moves": bald.evaluated_moves,
        "bald_feasible_moves": bald.feasible_moves,
        "bald_rejected_cyclic_moves": bald.rejected_cyclic_moves,
        "bald_improving_moves": bald.improving_moves,
        "bald_accepted_moves": bald.accepted_moves,
        "bald_rejected_moves": bald.rejected_moves,
        "diversity_trajectory": [list(pt) for pt in rec.diversity_trajectory],
        "total_elapsed_ms": rec.total_elapsed_ns / 1e6,
    }


def _phase_c_ref_row(raw_row: dict[str, Any], alias_id: str) -> dict[str, Any]:
    decode_ev = int(raw_row["decode_evals"])
    ls_ev = int(raw_row["ls_evals"])
    ev_used = int(raw_row["evals_used"])
    ls_share = 100.0 * float(ls_ev) / float(ev_used) if ev_used > 0 else 0.0
    return {
        "stage": "phase_c_reference",
        "experiment_id": alias_id,
        "base_config_id": raw_row["config_id"],
        "initializer_id": raw_row["initializer_id"],
        "ls_neighborhood": raw_row["ls_neighborhood"],
        "ls_strategy": raw_row["ls_strategy"],
        "max_ls_evals": 50,
        "candidate_fraction": 1.0,
        "instance_id": raw_row["instance_id"],
        "family": raw_row["family"],
        "split": raw_row["split"],
        "jobs": raw_row["jobs"],
        "machines": raw_row["machines"],
        "operation_count": raw_row["operation_count"],
        "bks": raw_row["bks"],
        "lb": raw_row["lb"],
        "replicate_index": raw_row["replicate_index"],
        "master_seed": raw_row["master_seed"],
        "evaluation_budget": raw_row["evaluation_budget"],
        "evals_used": ev_used,
        "decode_evals": decode_ev,
        "ls_evals": ls_ev,
        "ls_budget_share_pct": ls_share,
        "generations_executed": raw_row["generations_executed"],
        "best_makespan": raw_row["best_makespan"],
        "best_rpd_bks": raw_row["best_rpd_bks"],
        "hit_bks": raw_row["hit_bks"],
        "best_found_eval": raw_row["best_found_eval"],
        "best_found_generation": raw_row["best_found_generation"],
        "feasible_final": raw_row["feasible_final"],
        "anytime_rpd_auc": raw_row["anytime_rpd_auc"],
        "final_obr_diversity": raw_row["final_obr_diversity"],
        "final_lam_diversity": raw_row.get("final_lam_diversity"),
        "final_bald_diversity": raw_row.get("final_bald_diversity"),
        "final_gate_p": raw_row.get("final_gate_p"),
        "lam_best_makespan": raw_row.get("lam_best_makespan"),
        "bald_best_makespan": raw_row.get("bald_best_makespan"),
        "writeback_count": raw_row["writeback_count"],
        "total_elapsed_ms": raw_row["total_elapsed_ms"],
    }


def _relabel(rows: list[dict[str, Any]], new_id: str) -> list[dict[str, Any]]:
    return [dict(r, experiment_id=new_id) for r in rows]


def _opt_mean(vals: list[float | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None


def _run_c02_batch(
    instances: list[dict[str, Any]],
    configs: list[tuple[str, str, str, str, int, float]],
    seeds: list[int],
    *,
    stage_label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reps = list(range(len(seeds)))
    for idx, inst in enumerate(instances, start=1):
        parsed = parse_jssp_file(
            REPO_ROOT / inst["raw_relative_path"], instance_id=inst["instance_id"]
        )
        budget = default_phase_c_budget(inst["jobs"], inst["machines"])
        tasks = [
            (
                exp_id,
                base_cfg,
                nb,
                strat,
                b_ls,
                f_ls,
                rep,
                None,
                "paired_init",
                "phase_c_shared_init",
                base_cfg,
            )
            for (exp_id, base_cfg, nb, strat, b_ls, f_ls) in configs
            for rep in reps
        ]
        t0 = time.perf_counter()
        recs = run_dual_view_exploratory_batch(
            parsed.jobs,
            inst["instance_id"],
            tasks,
            initializer_id="A11",
            evaluation_budget=budget,
            num_threads=0,
        )
        dt = time.perf_counter() - t0
        for rec_idx, rec in enumerate(recs):
            rep = rec_idx % len(seeds)
            row = _row_from_record(
                inst,
                rep,
                seeds[rep],
                budget,
                rec,
                stage=stage_label,
            )
            _assert_single_view_conservation(row)
            rows.append(row)
        best_r = min(rows[-len(recs) :], key=lambda r: r["best_makespan"])
        print(
            f"  [{stage_label} {idx:2d}/{len(instances)}] {inst['instance_id']:6s} "
            f"({inst['jobs']:2d}x{inst['machines']:2d}, B={budget:5d}): "
            f"{len(recs):2d} runs in {dt:5.2f}s | best Cmax={best_r['best_makespan']} "
            f"({best_r['experiment_id']})",
            flush=True,
        )
    return rows


def _assert_single_view_conservation(row: dict[str, Any]) -> None:
    assert row["feasible_final"] is True
    assert row["evals_used"] <= row["evaluation_budget"]
    assert row["init_evals"] + row["search_evals"] == row["evals_used"]
    assert row["decode_evals"] + row["ls_evals"] == row["evals_used"]
    assert row["lam_decode_evals"] == row["decode_evals"]
    assert row["lam_ls_evals"] == row["ls_evals"]
    assert row["bald_decode_evals"] == 0
    assert row["bald_ls_evals"] == 0
    assert row["final_gate_p"] is None
    if row["candidate_fraction"] < 1.0 - 1e-12:
        assert row["lam_ls_calls"] < row["lam_decode_evals"]


def _assert_c02_parity(rows: list[dict[str, Any]], c02_ref: list[dict[str, Any]]) -> None:
    by_key = {(r["instance_id"], r["replicate_index"]): r for r in c02_ref}
    checked = 0
    for r in rows:
        ref = by_key[(r["instance_id"], r["replicate_index"])]
        assert r["best_makespan"] == ref["best_makespan"], (
            f"C02 parity mismatch on {r['instance_id']} r{r['replicate_index']}"
        )
        assert r["evals_used"] == ref["evals_used"]
        assert r["decode_evals"] == ref["decode_evals"]
        assert r["ls_evals"] == ref["ls_evals"]
        assert r["generations_executed"] == ref["generations_executed"]
        assert r["writeback_count"] == ref["writeback_count"]
        checked += 1
    assert checked == len(rows) and checked > 0


def _summarize_comparison(
    rows: list[dict[str, Any]],
    exp_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inst_ids = sorted({r["instance_id"] for r in rows})
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        by_pair.setdefault((r["experiment_id"], r["instance_id"]), []).append(r)

    med_rpd_mat = np.zeros((len(inst_ids), len(exp_ids)), dtype=float)
    scores_by_id: dict[str, np.ndarray] = {}
    summary: list[dict[str, Any]] = []

    for c_idx, eid in enumerate(exp_ids):
        inst_med: list[float] = []
        inst_mean: list[float] = []
        inst_best: list[float] = []
        inst_auc: list[float] = []
        inst_hit: list[float] = []
        inst_gens: list[float] = []
        inst_ms: list[float] = []
        inst_evals: list[float] = []
        inst_decode: list[float] = []
        inst_ls_eval: list[float] = []
        inst_ls_share: list[float] = []
        inst_div: list[float] = []
        inst_wb: list[float] = []
        inst_gp: list[float | None] = []
        all_eid: list[dict[str, Any]] = []

        for i_idx, iid in enumerate(inst_ids):
            grp = by_pair[(eid, iid)]
            all_eid.extend(grp)
            rpds = np.array([g["best_rpd_bks"] for g in grp], dtype=float)
            med = float(np.median(rpds))
            med_rpd_mat[i_idx, c_idx] = med
            inst_med.append(med)
            inst_mean.append(float(np.mean(rpds)))
            inst_best.append(float(np.min(rpds)))
            inst_auc.append(float(np.median([g["anytime_rpd_auc"] for g in grp])))
            inst_hit.append(float(np.mean([1.0 if g["hit_bks"] else 0.0 for g in grp])))
            inst_gens.append(float(np.mean([g["generations_executed"] for g in grp])))
            inst_ms.append(float(np.mean([g["total_elapsed_ms"] for g in grp])))
            inst_evals.append(float(np.mean([g["evals_used"] for g in grp])))
            inst_decode.append(float(np.mean([g["decode_evals"] for g in grp])))
            inst_ls_eval.append(float(np.mean([g["ls_evals"] for g in grp])))
            inst_ls_share.append(float(np.mean([g["ls_budget_share_pct"] for g in grp])))
            inst_div.append(float(np.mean([g["final_obr_diversity"] for g in grp])))
            inst_wb.append(float(np.mean([g["writeback_count"] for g in grp])))
            inst_gp.append(_opt_mean([g.get("final_gate_p") for g in grp]))

        scores_by_id[eid] = med_rpd_mat[:, c_idx].copy()
        sample = all_eid[0]

        def _bmean(field: str, subset: list[dict[str, Any]] = all_eid) -> float | None:
            vals = [float(r[field]) for r in subset if r.get(field) is not None]
            return float(np.mean(vals)) if vals else None

        summary.append(
            {
                "experiment_id": eid,
                "base_config_id": sample["base_config_id"],
                "ls_strategy": sample["ls_strategy"],
                "ls_neighborhood": sample["ls_neighborhood"],
                "max_ls_evals": int(sample["max_ls_evals"]),
                "candidate_fraction": float(sample["candidate_fraction"]),
                "n_instances": len(inst_ids),
                "n_runs": len(all_eid),
                "feasible_rate": 1.0,
                "median_rpd_mean": float(np.mean(inst_med)),
                "mean_rpd_mean": float(np.mean(inst_mean)),
                "best_rpd_mean": float(np.mean(inst_best)),
                "anytime_rpd_auc_mean": float(np.mean(inst_auc)),
                "hit_bks_rate": float(np.mean(inst_hit)),
                "mean_generations": float(np.mean(inst_gens)),
                "mean_elapsed_ms": float(np.mean(inst_ms)),
                "mean_evals_used": float(np.mean(inst_evals)),
                "mean_decode_evals": float(np.mean(inst_decode)),
                "mean_ls_evals": float(np.mean(inst_ls_eval)),
                "mean_ls_budget_share_pct": float(np.mean(inst_ls_share)),
                "mean_writebacks": float(np.mean(inst_wb)),
                "final_obr_div_mean": float(np.mean(inst_div)),
                "final_gate_p_mean": _opt_mean(inst_gp),
                "lam_ls_calls_mean": _bmean("lam_ls_calls"),
                "lam_improving_calls_mean": _bmean("lam_improving_calls"),
                "lam_accepted_moves_mean": _bmean("lam_accepted_moves"),
                "bald_ls_evals_mean": _bmean("bald_ls_evals"),
            }
        )

    fr = run_friedman_analysis(exp_ids, med_rpd_mat)
    for r in summary:
        r["friedman_rank"] = fr.mean_ranks[r["experiment_id"]]
    summary.sort(
        key=lambda r: (
            r["median_rpd_mean"],
            r["friedman_rank"],
            r["anytime_rpd_auc_mean"],
            r["mean_elapsed_ms"],
        )
    )
    paired = paired_comparisons_table(scores_by_id, COMPARISON_CONTRASTS, lower_is_better=True)
    return summary, {
        "friedman_med_rpd": {
            "statistic": fr.statistic,
            "p_value": fr.p_value,
            "kendall_w": fr.kendall_w,
            "num_instances": fr.num_instances,
            "num_methods": fr.num_methods,
            "mean_ranks": fr.mean_ranks,
        },
        "paired_contrasts": paired,
    }


def _diversity_milestones(rows: list[dict[str, Any]], exp_ids: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for eid in exp_ids:
        sub = [r for r in rows if r["experiment_id"] == eid and r.get("diversity_trajectory")]
        if not sub:
            continue
        max_gen = max(r["generations_executed"] for r in sub)
        for g in sorted({0, 5, 10, 15, 20, 30, 50, max_gen}):
            samples = [pt for r in sub for pt in r["diversity_trajectory"] if int(pt[0]) == g]
            if samples:
                out.append(
                    {
                        "experiment_id": eid,
                        "generation": g,
                        "n_active_runs": len(samples),
                        "mean_obr_diversity": float(np.mean([s[1] for s in samples])),
                    }
                )
        out.append(
            {
                "experiment_id": eid,
                "generation": -1,
                "n_active_runs": len(sub),
                "mean_obr_diversity": float(np.mean([r["final_obr_diversity"] for r in sub])),
            }
        )
    return out


def main() -> None:
    started_utc = datetime.now(tz=UTC).isoformat()
    suite = json.loads(SUITE_JSON.read_text(encoding="utf-8"))
    instances = [x for x in suite["instances"] if x.get("in_common_pipeline_suite")]
    dev_instances = [x for x in instances if x["split"] == "development"]
    ho_instances = [x for x in instances if x["split"] == "held_out"]
    assert len(dev_instances) == 16 and len(ho_instances) == 16
    seeds = list(get_frozen_master_seeds())[:NUM_REPLICATES]
    rng_meta = get_rng_metadata()

    dv_sel = json.loads(DV_SEL_JSON.read_text(encoding="utf-8"))["selected_dv_star"]
    assert int(dv_sel["max_ls_evals"]) == 10
    assert abs(float(dv_sel["candidate_fraction"]) - 0.60) < 1e-12
    assert dv_sel["ls_strategy"] == "budget_limited_sa"

    # Load read-only frozen C02 references and previous DV exploratory rows
    phase_c_raw = [json.loads(line) for line in PHASE_C_RAW_JSONL.open(encoding="utf-8")]
    c02_dev_raw = [
        r for r in phase_c_raw if r["config_id"] == "C02" and r["split"] == "development"
    ]
    c02_ho_raw = [r for r in phase_c_raw if r["config_id"] == "C02" and r["split"] == "held_out"]
    c02_dev_ref = [_phase_c_ref_row(r, "C02_LGA") for r in c02_dev_raw]
    c02_ho_ref = [_phase_c_ref_row(r, "C02_LGA") for r in c02_ho_raw]

    dv_raw = [json.loads(line) for line in DV_RAW_JSONL.open(encoding="utf-8")]
    dv_star_dev = _relabel(
        [
            r
            for r in dv_raw
            if r["split"] == "development" and r["experiment_id"] == dv_sel["stage3_winner_id"]
        ],
        "DV_STAR",
    )
    dv_ctrl_dev = _relabel(
        [r for r in dv_raw if r["split"] == "development" and r["experiment_id"] == "DV_A4_B50"],
        "DV_CTRL_C06",
    )
    dv_star_ho = [
        r for r in dv_raw if r["stage"] == "Stage4_HeldOut" and r["experiment_id"] == "DV_STAR"
    ]
    dv_ctrl_ho = [
        r for r in dv_raw if r["stage"] == "Stage4_HeldOut" and r["experiment_id"] == "DV_CTRL_C06"
    ]
    assert len(dv_star_dev) == 320 and len(dv_ctrl_dev) == 320
    assert len(dv_star_ho) == 320 and len(dv_ctrl_ho) == 320

    cfgs = [
        ("C02_CTRL_SD50", "C02", "N5", "steepest_descent", 50, 1.0),
        ("C02_RETUNED_SA", "C02", "N5", "budget_limited_sa", 10, 0.60),
    ]

    # ------------------------------------------------------------------
    # STAGE 1: Development Split Verification & Parity Check (Dev First)
    # ------------------------------------------------------------------
    print("\n=== Stage 1: Retuned C02 Control on Dev Split (16 x 20 seeds) ===", flush=True)
    dev_rows = _run_c02_batch(dev_instances, cfgs, seeds, stage_label="RetunedC02_Dev")
    c02_ctrl_dev = [r for r in dev_rows if r["experiment_id"] == "C02_CTRL_SD50"]
    c02_ret_dev = [r for r in dev_rows if r["experiment_id"] == "C02_RETUNED_SA"]
    _assert_c02_parity(c02_ctrl_dev, c02_dev_raw)
    print(
        "  [Verified] C02_CTRL_SD50 matches frozen C02 on all 320 Dev runs bit-for-bit.",
        flush=True,
    )

    exp_ids = ["DV_STAR", "C02_RETUNED_SA", "C02_LGA", "DV_CTRL_C06"]
    dev_sum, dev_stats = _summarize_comparison(
        dv_star_dev + c02_ret_dev + c02_dev_ref + dv_ctrl_dev,
        exp_ids,
    )

    # Freeze comparison protocol & Dev lock strictly BEFORE Held-Out runs
    lock_doc: dict[str, Any] = {
        "schema_version": "jssp-exploratory-retuned-c02-lock-v1",
        "frozen_before_held_out_utc": datetime.now(tz=UTC).isoformat(),
        "selection_split": "development_16_only",
        "dv_star_lock_sha256": canonical_json_sha256(
            json.loads(DV_SEL_JSON.read_text(encoding="utf-8"))
        ),
        "control_specification": {
            "experiment_id": "C02_RETUNED_SA",
            "base_config_id": "C02",
            "initializer_id": "A11",
            "neighborhood": "N5",
            "ls_strategy": "budget_limited_sa",
            "max_ls_evals": 10,
            "candidate_fraction": 0.60,
            "population_size": 64,
            "elitism_count": 2,
            "offspring_per_gen": 62,
            "ls_candidates_per_gen_after_t0": 38,
            "ls_candidates_at_t0": 39,
            "candidate_selection_key": "(raw_makespan, cohort_idx)",
            "equalized_budget_formula": "clamp(64 * ceil(2.5 * J * M), 6400, 64000)",
        },
        "development_summary": dev_sum,
    }
    lock_doc["comparison_lock_sha256"] = canonical_json_sha256(lock_doc)
    _write_json(EXP_ROOT / "selection" / "frozen_retuned_c02_comparison_lock.json", lock_doc)

    # ------------------------------------------------------------------
    # STAGE 2: Frozen Held-Out Validation (16 x 20 seeds, Zero Tuning)
    # ------------------------------------------------------------------
    print("\n=== Stage 2: Retuned C02 Control on Held-Out Split (16 x 20 seeds) ===", flush=True)
    ho_rows = _run_c02_batch(ho_instances, cfgs, seeds, stage_label="RetunedC02_HeldOut")
    c02_ctrl_ho = [r for r in ho_rows if r["experiment_id"] == "C02_CTRL_SD50"]
    c02_ret_ho = [r for r in ho_rows if r["experiment_id"] == "C02_RETUNED_SA"]
    _assert_c02_parity(c02_ctrl_ho, c02_ho_raw)
    print(
        "  [Verified] C02_CTRL_SD50 matches frozen C02 on all 320 Held-Out runs bit-for-bit.",
        flush=True,
    )

    ho_sum, ho_stats = _summarize_comparison(
        dv_star_ho + c02_ret_ho + c02_ho_ref + dv_ctrl_ho,
        exp_ids,
    )
    div_traj = _diversity_milestones(
        dv_star_ho + c02_ret_ho + dv_ctrl_ho, ["DV_STAR", "C02_RETUNED_SA", "DV_CTRL_C06"]
    )

    # Write separate raw JSONL and analysis artifacts
    all_new_rows = dev_rows + ho_rows
    raw_path = EXP_ROOT / "raw" / "retuned_c02_control_runs.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as fh:
        for row in all_new_rows:
            fh.write(json.dumps(row) + "\n")

    _write_csv(EXP_ROOT / "analysis" / "retuned_c02_dev_summary.csv", dev_sum)
    _write_json(EXP_ROOT / "analysis" / "retuned_c02_dev_summary.json", dev_sum)
    _write_csv(EXP_ROOT / "analysis" / "retuned_c02_held_out_summary.csv", ho_sum)
    _write_json(EXP_ROOT / "analysis" / "retuned_c02_held_out_summary.json", ho_sum)
    _write_csv(EXP_ROOT / "analysis" / "retuned_c02_diversity_trajectories.csv", div_traj)
    _write_json(EXP_ROOT / "analysis" / "retuned_c02_diversity_trajectories.json", div_traj)

    stats_doc = {
        "schema_version": "jssp-exploratory-retuned-c02-stats-v1",
        "development_16": dev_stats,
        "held_out_16": ho_stats,
    }
    _write_json(EXP_ROOT / "analysis" / "retuned_c02_statistical_tests.json", stats_doc)
    _write_csv(
        EXP_ROOT / "analysis" / "retuned_c02_statistical_tests.csv",
        [dict(r, split="development") for r in dev_stats["paired_contrasts"]]
        + [dict(r, split="held_out") for r in ho_stats["paired_contrasts"]],
    )

    git_prov = collect_git_provenance(REPO_ROOT)
    env_prov = collect_environment_provenance(REPO_ROOT, compiler_profile="release_editable")
    manifest_doc = {
        "schema_version": "jssp-exploratory-retuned-c02-manifest-v1",
        "track": "exploratory_retuned_c02_single_view_control",
        "started_utc": started_utc,
        "completed_utc": datetime.now(tz=UTC).isoformat(),
        "git": git_prov.model_dump(mode="json"),
        "environment": env_prov.model_dump(mode="json"),
        "rng": {
            "algorithm": rng_meta.algorithm,
            "seed_schedule_id": rng_meta.master_seed_schedule_id,
            "master_seeds_v1": seeds,
            "pairing_mode": "paired_init",
            "pairing_group": "phase_c_shared_init",
        },
        "total_new_runs": len(all_new_rows),
        "raw_jsonl_sha256": _sha256_file(raw_path),
        "phase_c_reference_raw_sha256": _sha256_file(PHASE_C_RAW_JSONL),
        "dv_exploratory_raw_sha256": _sha256_file(DV_RAW_JSONL),
        "comparison_lock_sha256": lock_doc["comparison_lock_sha256"],
    }
    _write_json(EXP_ROOT / "manifests" / "retuned_c02_control_manifest.json", manifest_doc)

    print("\n=== Held-Out Comparison Summary ===", flush=True)
    for row in ho_sum:
        print(
            f"  {row['experiment_id']:15s}: MedRPD={row['median_rpd_mean']:.3f}% "
            f"MeanRPD={row['mean_rpd_mean']:.3f}% BestRPD={row['best_rpd_mean']:.3f}% "
            f"Rank={row['friedman_rank']:.2f} Gens={row['mean_generations']:.2f} "
            f"LSShare={row['mean_ls_budget_share_pct']:.1f}% ms={row['mean_elapsed_ms']:.1f}",
            flush=True,
        )
    print("\nRetuned C02 Control Experiment COMPLETE.", flush=True)


if __name__ == "__main__":
    main()
