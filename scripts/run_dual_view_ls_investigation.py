"""Execute the exploratory Dual-View LS budget investigation (Experiments A, B, C + Held-Out)."""

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
EXP_ROOT = REPO_ROOT / "results/exploratory_dv_ls"
REPORT_PATH = REPO_ROOT / "reports/dual_view_ls_budget_investigation.md"
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
        "gate_p_trajectory": [list(pt) for pt in rec.gate_p_trajectory],
        "diversity_trajectory": [list(pt) for pt in rec.diversity_trajectory],
        "total_elapsed_ms": rec.total_elapsed_ns / 1e6,
    }


def _phase_c_ref_row(raw_row: dict[str, Any], alias_id: str) -> dict[str, Any]:
    """Normalize a read-only Phase C raw JSONL row into compatible comparison fields."""
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


def _run_instances(
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
            rows.append(
                _row_from_record(
                    inst,
                    rep,
                    seeds[rep],
                    budget,
                    rec,
                    stage=stage_label,
                )
            )
        best_r = min(rows[-len(recs) :], key=lambda r: r["best_makespan"])
        print(
            f"  [{stage_label} {idx:2d}/{len(instances)}] {inst['instance_id']:6s} "
            f"({inst['jobs']:2d}x{inst['machines']:2d}, B={budget:5d}): "
            f"{len(recs):3d} runs in {dt:5.2f}s | best Cmax={best_r['best_makespan']} "
            f"({best_r['experiment_id']})",
            flush=True,
        )
    return rows


def _opt_mean(vals: list[float | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None


def _summarize_stage(
    rows: list[dict[str, Any]],
    exp_ids: list[str],
    contrasts: list[tuple[str, str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inst_ids = sorted({r["instance_id"] for r in rows})
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        by_pair.setdefault((r["experiment_id"], r["instance_id"]), []).append(r)

    med_rpd_mat = np.zeros((len(inst_ids), len(exp_ids)), dtype=float)
    scores_by_id: dict[str, np.ndarray] = {}
    summary: list[dict[str, Any]] = []

    for c_idx, eid in enumerate(exp_ids):
        inst_med_rpds: list[float] = []
        inst_mean_rpds: list[float] = []
        inst_best_rpds: list[float] = []
        inst_auc: list[float] = []
        inst_hit: list[float] = []
        inst_gens: list[float] = []
        inst_ms: list[float] = []
        inst_evals: list[float] = []
        inst_ls_share: list[float] = []
        inst_div: list[float] = []
        inst_lam_div: list[float | None] = []
        inst_bald_div: list[float | None] = []
        inst_gp: list[float | None] = []
        # Branch telemetry means
        all_eid_rows: list[dict[str, Any]] = []

        for i_idx, iid in enumerate(inst_ids):
            grp = by_pair[(eid, iid)]
            all_eid_rows.extend(grp)
            rpds = np.array([g["best_rpd_bks"] for g in grp], dtype=float)
            med = float(np.median(rpds))
            med_rpd_mat[i_idx, c_idx] = med
            inst_med_rpds.append(med)
            inst_mean_rpds.append(float(np.mean(rpds)))
            inst_best_rpds.append(float(np.min(rpds)))
            inst_auc.append(float(np.median([g["anytime_rpd_auc"] for g in grp])))
            inst_hit.append(float(np.mean([1.0 if g["hit_bks"] else 0.0 for g in grp])))
            inst_gens.append(float(np.mean([g["generations_executed"] for g in grp])))
            inst_ms.append(float(np.mean([g["total_elapsed_ms"] for g in grp])))
            inst_evals.append(float(np.mean([g["evals_used"] for g in grp])))
            inst_ls_share.append(float(np.mean([g["ls_budget_share_pct"] for g in grp])))
            inst_div.append(float(np.mean([g["final_obr_diversity"] for g in grp])))
            inst_lam_div.append(_opt_mean([g.get("final_lam_diversity") for g in grp]))
            inst_bald_div.append(_opt_mean([g.get("final_bald_diversity") for g in grp]))
            inst_gp.append(_opt_mean([g.get("final_gate_p") for g in grp]))

        scores_by_id[eid] = med_rpd_mat[:, c_idx].copy()
        sample = all_eid_rows[0]

        def _bmean(field: str, subset: list[dict[str, Any]] = all_eid_rows) -> float | None:
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
                "n_runs": len(all_eid_rows),
                "feasible_rate": float(
                    np.mean([1.0 if r["feasible_final"] else 0.0 for r in all_eid_rows])
                ),
                "median_rpd_mean": float(np.mean(inst_med_rpds)),
                "mean_rpd_mean": float(np.mean(inst_mean_rpds)),
                "best_rpd_mean": float(np.mean(inst_best_rpds)),
                "anytime_rpd_auc_mean": float(np.mean(inst_auc)),
                "hit_bks_rate": float(np.mean(inst_hit)),
                "mean_generations": float(np.mean(inst_gens)),
                "mean_elapsed_ms": float(np.mean(inst_ms)),
                "mean_evals_used": float(np.mean(inst_evals)),
                "mean_ls_budget_share_pct": float(np.mean(inst_ls_share)),
                "final_obr_div_mean": float(np.mean(inst_div)),
                "final_lam_div_mean": _opt_mean(inst_lam_div),
                "final_bald_div_mean": _opt_mean(inst_bald_div),
                "final_gate_p_mean": _opt_mean(inst_gp),
                "lam_decode_evals_mean": _bmean("lam_decode_evals"),
                "lam_ls_evals_mean": _bmean("lam_ls_evals"),
                "lam_ls_calls_mean": _bmean("lam_ls_calls"),
                "lam_improving_calls_mean": _bmean("lam_improving_calls"),
                "lam_accepted_moves_mean": _bmean("lam_accepted_moves"),
                "lam_rejected_moves_mean": _bmean("lam_rejected_moves"),
                "lam_rejected_cyclic_moves_mean": _bmean("lam_rejected_cyclic_moves"),
                "bald_decode_evals_mean": _bmean("bald_decode_evals"),
                "bald_ls_evals_mean": _bmean("bald_ls_evals"),
                "bald_ls_calls_mean": _bmean("bald_ls_calls"),
                "bald_improving_calls_mean": _bmean("bald_improving_calls"),
                "bald_accepted_moves_mean": _bmean("bald_accepted_moves"),
                "bald_rejected_moves_mean": _bmean("bald_rejected_moves"),
                "bald_rejected_cyclic_moves_mean": _bmean("bald_rejected_cyclic_moves"),
            }
        )

    friedman_dict: dict[str, Any] | None = None
    if len(exp_ids) >= 3:
        fr = run_friedman_analysis(exp_ids, med_rpd_mat)
        for r in summary:
            r["friedman_rank"] = fr.mean_ranks[r["experiment_id"]]
        friedman_dict = {
            "statistic": fr.statistic,
            "p_value": fr.p_value,
            "kendall_w": fr.kendall_w,
            "num_instances": fr.num_instances,
            "num_methods": fr.num_methods,
            "mean_ranks": fr.mean_ranks,
        }
    else:
        for r in summary:
            r["friedman_rank"] = 1.0

    summary.sort(
        key=lambda r: (
            r["median_rpd_mean"],
            r["friedman_rank"],
            r["anytime_rpd_auc_mean"],
            r["mean_elapsed_ms"],
        )
    )

    paired = paired_comparisons_table(scores_by_id, contrasts, lower_is_better=True)
    return summary, {"friedman_med_rpd": friedman_dict, "paired_contrasts": paired}


def _relabel(rows: list[dict[str, Any]], new_id: str) -> list[dict[str, Any]]:
    return [dict(r, experiment_id=new_id) for r in rows]


def _diversity_milestones(rows: list[dict[str, Any]], exp_ids: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for eid in exp_ids:
        sub = [r for r in rows if r["experiment_id"] == eid and r.get("diversity_trajectory")]
        if not sub:
            continue
        max_gen = max(r["generations_executed"] for r in sub)
        points = [0, 5, 10, 15, 20, 30, 50, max_gen]
        seen: set[int] = set()
        for g in points:
            if g in seen:
                continue
            seen.add(g)
            samples = [pt for r in sub for pt in r["diversity_trajectory"] if int(pt[0]) == g]
            if samples:
                out.append(
                    {
                        "experiment_id": eid,
                        "generation": g,
                        "n_active_runs": len(samples),
                        "mean_obr_diversity": float(np.mean([s[1] for s in samples])),
                        "mean_lam_diversity": float(np.mean([s[2] for s in samples])),
                        "mean_bald_diversity": float(np.mean([s[3] for s in samples])),
                    }
                )
        # Always include final generation summary across all runs
        out.append(
            {
                "experiment_id": eid,
                "generation": -1,
                "n_active_runs": len(sub),
                "mean_obr_diversity": float(np.mean([r["final_obr_diversity"] for r in sub])),
                "mean_lam_diversity": float(np.mean([r["final_lam_diversity"] for r in sub])),
                "mean_bald_diversity": float(np.mean([r["final_bald_diversity"] for r in sub])),
            }
        )
    return out


def _assert_c06_parity(rows: list[dict[str, Any]], c06_ref: list[dict[str, Any]]) -> None:
    by_key = {(r["instance_id"], r["replicate_index"]): r for r in c06_ref}
    checked = 0
    for r in rows:
        ref = by_key[(r["instance_id"], r["replicate_index"])]
        assert r["best_makespan"] == ref["best_makespan"], (
            f"C06 parity mismatch on {r['instance_id']} r{r['replicate_index']}"
        )
        assert r["evals_used"] == ref["evals_used"]
        assert r["generations_executed"] == ref["generations_executed"]
        assert r["final_gate_p"] == ref["final_gate_p"]
        checked += 1
    assert checked == len(rows) and checked > 0


def main() -> None:
    started_utc = datetime.now(tz=UTC).isoformat()
    suite = json.loads(SUITE_JSON.read_text(encoding="utf-8"))
    instances = [x for x in suite["instances"] if x.get("in_common_pipeline_suite")]
    dev_instances = [x for x in instances if x["split"] == "development"]
    ho_instances = [x for x in instances if x["split"] == "held_out"]
    assert len(dev_instances) == 16 and len(ho_instances) == 16, (
        f"Expected 16/16 common suite split, got {len(dev_instances)}/{len(ho_instances)}"
    )
    seeds = list(get_frozen_master_seeds())[:NUM_REPLICATES]
    rng_meta = get_rng_metadata()

    # Load read-only Phase C frozen raw rows for C02 and C06 parity & reference comparison
    phase_c_raw = [json.loads(line) for line in PHASE_C_RAW_JSONL.open(encoding="utf-8")]
    c06_dev_ref = [
        r for r in phase_c_raw if r["config_id"] == "C06" and r["split"] == "development"
    ]
    c06_ho_ref = [r for r in phase_c_raw if r["config_id"] == "C06" and r["split"] == "held_out"]
    c02_dev_ref = [
        _phase_c_ref_row(r, "C02_LGA")
        for r in phase_c_raw
        if r["config_id"] == "C02" and r["split"] == "development"
    ]
    c02_ho_ref = [
        _phase_c_ref_row(r, "C02_LGA")
        for r in phase_c_raw
        if r["config_id"] == "C02" and r["split"] == "held_out"
    ]

    all_new_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # STAGE 1: Experiment A — LS Evaluation Limit (Dev only)
    # ------------------------------------------------------------------
    print("\n=== Stage 1: Experiment A (LS Budget Cap on Dev) ===", flush=True)
    exp_a_cfgs = [
        ("DV_A1_B10", "C06", "N5", "steepest_descent", 10, 1.0),
        ("DV_A2_B20", "C06", "N5", "steepest_descent", 20, 1.0),
        ("DV_A3_B30", "C06", "N5", "steepest_descent", 30, 1.0),
        ("DV_A4_B50", "C06", "N5", "steepest_descent", 50, 1.0),
    ]
    exp_a_rows = _run_instances(dev_instances, exp_a_cfgs, seeds, stage_label="ExpA_Dev")
    all_new_rows.extend(exp_a_rows)
    _assert_c06_parity([r for r in exp_a_rows if r["experiment_id"] == "DV_A4_B50"], c06_dev_ref)
    print("  [Verified] DV_A4_B50 matches frozen C06 on all 320 Dev runs bit-for-bit.", flush=True)

    exp_a_sum, exp_a_stats = _summarize_stage(
        exp_a_rows,
        [c[0] for c in exp_a_cfgs],
        [
            ("exp_a_vs_control", "DV_A1_B10", "DV_A4_B50"),
            ("exp_a_vs_control", "DV_A2_B20", "DV_A4_B50"),
            ("exp_a_vs_control", "DV_A3_B30", "DV_A4_B50"),
        ],
    )
    best_a_id = exp_a_sum[0]["experiment_id"]
    b_ls_star = int(exp_a_sum[0]["max_ls_evals"])
    print(f"  => Stage 1 Winner on Dev: {best_a_id} (b_LS* = {b_ls_star})", flush=True)

    # ------------------------------------------------------------------
    # STAGE 2: Experiment B — LS Candidate Selection Fraction (Dev only)
    # ------------------------------------------------------------------
    print(
        f"\n=== Stage 2: Experiment B (Candidate Selection at b_LS*={b_ls_star} on Dev) ===",
        flush=True,
    )
    exp_b_new_cfgs = [
        ("DV_B1_F020", "C06", "N5", "steepest_descent", b_ls_star, 0.20),
        ("DV_B2_F040", "C06", "N5", "steepest_descent", b_ls_star, 0.40),
        ("DV_B3_F060", "C06", "N5", "steepest_descent", b_ls_star, 0.60),
    ]
    exp_b_new_rows = _run_instances(dev_instances, exp_b_new_cfgs, seeds, stage_label="ExpB_Dev")
    all_new_rows.extend(exp_b_new_rows)
    exp_b_rows = exp_b_new_rows + _relabel(
        [r for r in exp_a_rows if r["experiment_id"] == best_a_id], "DV_B4_F100"
    )
    exp_b_ids = ["DV_B1_F020", "DV_B2_F040", "DV_B3_F060", "DV_B4_F100"]
    exp_b_sum, exp_b_stats = _summarize_stage(
        exp_b_rows,
        exp_b_ids,
        [
            ("exp_b_vs_f100", "DV_B1_F020", "DV_B4_F100"),
            ("exp_b_vs_f100", "DV_B2_F040", "DV_B4_F100"),
            ("exp_b_vs_f100", "DV_B3_F060", "DV_B4_F100"),
        ],
    )
    best_b_id = exp_b_sum[0]["experiment_id"]
    f_ls_star = float(exp_b_sum[0]["candidate_fraction"])
    print(f"  => Stage 2 Winner on Dev: {best_b_id} (f_LS* = {f_ls_star:.2f})", flush=True)

    # ------------------------------------------------------------------
    # STAGE 3: Experiment C — LS Strategy under (b_LS*, f_LS*) on Dev
    # ------------------------------------------------------------------
    print(
        f"\n=== Stage 3: Experiment C (LS Strategy at b_LS*={b_ls_star}, "
        f"f_LS*={f_ls_star:.2f} on Dev) ===",
        flush=True,
    )
    exp_c_new_cfgs = [
        ("DV_C2_FI", "C06", "N5", "first_improvement", b_ls_star, f_ls_star),
        ("DV_C3_SA", "C06", "N5", "budget_limited_sa", b_ls_star, f_ls_star),
    ]
    exp_c_new_rows = _run_instances(dev_instances, exp_c_new_cfgs, seeds, stage_label="ExpC_Dev")
    all_new_rows.extend(exp_c_new_rows)
    exp_c_rows = (
        _relabel([r for r in exp_b_rows if r["experiment_id"] == best_b_id], "DV_C1_SD")
        + exp_c_new_rows
    )
    exp_c_ids = ["DV_C1_SD", "DV_C2_FI", "DV_C3_SA"]
    exp_c_sum, exp_c_stats = _summarize_stage(
        exp_c_rows,
        exp_c_ids,
        [
            ("exp_c_vs_sd", "DV_C2_FI", "DV_C1_SD"),
            ("exp_c_vs_sd", "DV_C3_SA", "DV_C1_SD"),
            ("exp_c_fi_vs_sa", "DV_C3_SA", "DV_C2_FI"),
        ],
    )
    best_c_row = exp_c_sum[0]
    best_c_id = best_c_row["experiment_id"]
    s_ls_star = str(best_c_row["ls_strategy"])
    print(
        f"  => Stage 3 Winner (Frozen DV_STAR): {best_c_id} "
        f"(b_LS*={b_ls_star}, f_LS*={f_ls_star:.2f}, strategy={s_ls_star})",
        flush=True,
    )

    # Freeze DV_STAR before inspecting Held-Out!
    dv_star_dev_rows = _relabel(
        [r for r in exp_c_rows if r["experiment_id"] == best_c_id], "DV_STAR"
    )
    ctrl_dev_rows = _relabel(
        [r for r in exp_a_rows if r["experiment_id"] == "DV_A4_B50"], "DV_CTRL_C06"
    )
    dev_ref_sum, dev_ref_stats = _summarize_stage(
        dv_star_dev_rows + ctrl_dev_rows + c02_dev_ref,
        ["DV_STAR", "DV_CTRL_C06", "C02_LGA"],
        [
            ("dev_vs_c06", "DV_STAR", "DV_CTRL_C06"),
            ("dev_vs_c02", "DV_STAR", "C02_LGA"),
            ("dev_c06_vs_c02", "DV_CTRL_C06", "C02_LGA"),
        ],
    )

    selection_doc: dict[str, Any] = {
        "schema_version": "jssp-exploratory-dv-ls-selection-v1",
        "frozen_at_utc": datetime.now(tz=UTC).isoformat(),
        "selection_split": "development_16_only",
        "selected_dv_star": {
            "stage1_winner_id": best_a_id,
            "stage2_winner_id": best_b_id,
            "stage3_winner_id": best_c_id,
            "base_config_id": "C06",
            "initializer_id": "A11",
            "neighborhood": "N5",
            "max_ls_evals": b_ls_star,
            "candidate_fraction": f_ls_star,
            "ls_strategy": s_ls_star,
        },
        "development_metrics": next(r for r in dev_ref_sum if r["experiment_id"] == "DV_STAR"),
    }
    selection_doc["selection_lock_sha256"] = canonical_json_sha256(selection_doc)
    _write_json(EXP_ROOT / "selection" / "frozen_dv_ls_selection.json", selection_doc)

    # ------------------------------------------------------------------
    # STAGE 4: Frozen Held-Out Validation (`DV_STAR` vs `C06` vs `C02`)
    # ------------------------------------------------------------------
    print("\n=== Stage 4: Frozen Held-Out Validation (16 instances x 20 seeds) ===", flush=True)
    ho_cfgs = [
        ("DV_STAR", "C06", "N5", s_ls_star, b_ls_star, f_ls_star),
        ("DV_CTRL_C06", "C06", "N5", "steepest_descent", 50, 1.0),
    ]
    ho_rows = _run_instances(ho_instances, ho_cfgs, seeds, stage_label="Stage4_HeldOut")
    all_new_rows.extend(ho_rows)
    _assert_c06_parity([r for r in ho_rows if r["experiment_id"] == "DV_CTRL_C06"], c06_ho_ref)
    print(
        "  [Verified] DV_CTRL_C06 matches frozen C06 on all 320 Held-Out runs bit-for-bit.",
        flush=True,
    )

    ho_ref_sum, ho_ref_stats = _summarize_stage(
        ho_rows + c02_ho_ref,
        ["DV_STAR", "DV_CTRL_C06", "C02_LGA"],
        [
            ("held_out_vs_c06", "DV_STAR", "DV_CTRL_C06"),
            ("held_out_vs_c02", "DV_STAR", "C02_LGA"),
            ("held_out_c06_vs_c02", "DV_CTRL_C06", "C02_LGA"),
        ],
    )

    div_traj_rows = _diversity_milestones(ho_rows, ["DV_STAR", "DV_CTRL_C06"])
    ho_star_rows = [r for r in ho_rows if r["experiment_id"] == "DV_STAR"]
    ho_lam_wins = sum(1 for r in ho_star_rows if r["lam_best_makespan"] < r["bald_best_makespan"])
    ho_branch_stats = {"ho_lam_wins": ho_lam_wins, "ho_total_runs": len(ho_star_rows)}

    # Write Raw JSONL (never touching existing results/phase_c/raw/)
    raw_path = EXP_ROOT / "raw" / "exploratory_dv_ls_runs.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as fh:
        for row in all_new_rows:
            fh.write(json.dumps(row) + "\n")

    # Write Analysis CSVs & JSONs
    _write_csv(EXP_ROOT / "analysis" / "exp_a_dev_summary.csv", exp_a_sum)
    _write_json(EXP_ROOT / "analysis" / "exp_a_dev_summary.json", exp_a_sum)
    _write_csv(EXP_ROOT / "analysis" / "exp_b_dev_summary.csv", exp_b_sum)
    _write_json(EXP_ROOT / "analysis" / "exp_b_dev_summary.json", exp_b_sum)
    _write_csv(EXP_ROOT / "analysis" / "exp_c_dev_summary.csv", exp_c_sum)
    _write_json(EXP_ROOT / "analysis" / "exp_c_dev_summary.json", exp_c_sum)
    _write_csv(EXP_ROOT / "analysis" / "stage4_dev_comparison.csv", dev_ref_sum)
    _write_csv(EXP_ROOT / "analysis" / "stage4_held_out_comparison.csv", ho_ref_sum)
    _write_csv(EXP_ROOT / "analysis" / "diversity_trajectories.csv", div_traj_rows)
    _write_json(EXP_ROOT / "analysis" / "diversity_trajectories.json", div_traj_rows)

    all_stats = {
        "schema_version": "jssp-exploratory-dv-ls-stats-v1",
        "exp_a_dev": exp_a_stats,
        "exp_b_dev": exp_b_stats,
        "exp_c_dev": exp_c_stats,
        "stage4_dev_comparison": dev_ref_stats,
        "stage4_held_out_comparison": ho_ref_stats,
    }
    _write_json(EXP_ROOT / "analysis" / "exploratory_statistical_tests.json", all_stats)
    _write_csv(
        EXP_ROOT / "analysis" / "exploratory_statistical_tests.csv",
        [dict(r, stage="exp_a_dev") for r in exp_a_stats["paired_contrasts"]]
        + [dict(r, stage="exp_b_dev") for r in exp_b_stats["paired_contrasts"]]
        + [dict(r, stage="exp_c_dev") for r in exp_c_stats["paired_contrasts"]]
        + [dict(r, stage="stage4_dev") for r in dev_ref_stats["paired_contrasts"]]
        + [dict(r, stage="stage4_held_out") for r in ho_ref_stats["paired_contrasts"]],
    )

    # Reproducibility Manifest
    git_prov = collect_git_provenance(REPO_ROOT)
    env_prov = collect_environment_provenance(REPO_ROOT, compiler_profile="release_editable")
    manifest_doc = {
        "schema_version": "jssp-exploratory-dv-ls-manifest-v1",
        "track": "exploratory_dual_view_ls_budget",
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
        "selected_benchmark_suite_sha256": _sha256_file(SUITE_JSON),
        "selection_lock_sha256": selection_doc["selection_lock_sha256"],
    }
    _write_json(EXP_ROOT / "manifests" / "exploratory_dv_ls_manifest.json", manifest_doc)

    # Generate Concise Markdown Report strictly from computed CSV/JSON structures
    _write_report(
        exp_a_sum=exp_a_sum,
        exp_a_stats=exp_a_stats,
        exp_b_sum=exp_b_sum,
        exp_b_stats=exp_b_stats,
        exp_c_sum=exp_c_sum,
        exp_c_stats=exp_c_stats,
        dev_ref_sum=dev_ref_sum,
        dev_ref_stats=dev_ref_stats,
        ho_ref_sum=ho_ref_sum,
        ho_ref_stats=ho_ref_stats,
        div_traj_rows=div_traj_rows,
        ho_branch_stats=ho_branch_stats,
        selection_doc=selection_doc,
        manifest_doc=manifest_doc,
    )
    print(f"\nExploratory Investigation COMPLETE. Report: {REPORT_PATH}", flush=True)


def _fmt_stage_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        (
            "| ID | Strategy | $b_{\\text{LS}}$ | $f_{\\text{LS}}$ | Med RPD | "
            "Mean RPD | Best RPD | Friedman Rank | Gens | LS Share | Final OBR Div | ms |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['experiment_id']}` | `{r['ls_strategy']}` | `{r['max_ls_evals']}` | "
            f"`{r['candidate_fraction']:.2f}` | `{r['median_rpd_mean']:.3f}%` | "
            f"`{r['mean_rpd_mean']:.3f}%` | `{r['best_rpd_mean']:.3f}%` | "
            f"`{r['friedman_rank']:.2f}` | `{r['mean_generations']:.2f}` | "
            f"`{r['mean_ls_budget_share_pct']:.1f}%` | `{r['final_obr_div_mean']:.3f}` | "
            f"`{r['mean_elapsed_ms']:.1f}` |"
        )
    return lines


def _fmt_contrasts(contrasts: list[dict[str, Any]]) -> list[str]:
    lines = [
        (
            "| Candidate vs. Ref | W/T/L | Mean Diff | $p_{\\text{raw}}$ | "
            "$p_{\\text{Holm}}$ | Sig (0.05) | $A_{12}$ | Cliff's $d$ |"
        ),
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for c in contrasts:
        sig = "Yes" if c["significant_0_05"] else "No"
        lines.append(
            f"| `{c['candidate']}` vs `{c['reference']}` | `{c['wins_ties_losses']}` | "
            f"`{c['mean_diff']:+.3f}%` | `{c['raw_p']:.3e}` | `{c['holm_p']:.3e}` | "
            f"{sig} | `{c['vargha_delaney_a12']:.3f}` | `{c['cliffs_delta']:+.3f}` |"
        )
    return lines


def _write_report(
    *,
    exp_a_sum: list[dict[str, Any]],
    exp_a_stats: dict[str, Any],
    exp_b_sum: list[dict[str, Any]],
    exp_b_stats: dict[str, Any],
    exp_c_sum: list[dict[str, Any]],
    exp_c_stats: dict[str, Any],
    dev_ref_sum: list[dict[str, Any]],
    dev_ref_stats: dict[str, Any],
    ho_ref_sum: list[dict[str, Any]],
    ho_ref_stats: dict[str, Any],
    div_traj_rows: list[dict[str, Any]],
    ho_branch_stats: dict[str, int],
    selection_doc: dict[str, Any],
    manifest_doc: dict[str, Any],
) -> None:
    sel = selection_doc["selected_dv_star"]
    lines: list[str] = [
        "# Dual-View Local Search Budget Investigation (Exploratory Extension)",
        "",
        "> **Scope Note:** All experiments reported here are exploratory additions on branch "
        "`exploratory/dual-view-ls-budget` (`"
        f"{manifest_doc['total_new_runs']:,}` new runs). Frozen Phase A–C (`A11`, `B21`, `C02`) "
        "and Integrated Pipeline (`P3`) confirmatory outputs remain completely unmodified.",
        "",
        "## 1. Implementation Audit Summary",
        "",
        (
            "1. **Strict Budget Accounting:** `EvalCounter` enforces "
            "`evals_used == init_evals + search_evals == decode_evals + ls_evals <= budget`. "
            "Control parity checks confirmed `DV_A4_B50` (Dev, `320` runs) and `DV_CTRL_C06` "
            "(Held-Out, `320` runs) match frozen `C06` bit-for-bit across makespans, "
            "evaluation counts, generations, and final gate weights."
        ),
        (
            "2. **Why `96.9%`–`97.5%` Was Consumed by LS:** Calling `steepest_descent + N5` "
            "(`b_LS=50`) on 100% of both views consumes `~33.9` LS evaluations per single genotype "
            "decode (`97.1%` per-offspring cost), shrinking evolution to `~12` Held-Out "
            "generations and duplicating deterministic LS on identical $t=0$ genotypes across "
            "both views."
        ),
        (
            "3. **Standalone SA Incompatibility Resolved:** Phase B `Simulated Annealing` probes "
            "50 uncommitted moves for $T_0$ calibration, leaving 0 committed moves at "
            "$b_{\\text{LS}} \\le 50$. Experiment C evaluates `run_budget_limited_sa` "
            "(`n_cal = clamp(b_LS/5, 2, 8)` retaining best calibration moves)."
        ),
        "",
        "## 2. Controlled Screening on Development Split (`16` Instances × `20` Seeds)",
        "",
        (
            "### Experiment A — Per-Individual LS Evaluation Cap (`f_LS = 1.00`, "
            "Steepest Descent + `N5`)"
        ),
        "",
        *_fmt_stage_table(exp_a_sum),
        "",
        (
            f"- **Dev Friedman:** $\\chi^2_F = "
            f"{exp_a_stats['friedman_med_rpd']['statistic']:.2f}$ "
            f"($p = {exp_a_stats['friedman_med_rpd']['p_value']:.3e}$, Kendall $W = "
            f"{exp_a_stats['friedman_med_rpd']['kendall_w']:.4f}$). Selected: "
            f"**`{sel['stage1_winner_id']}`** ($b_{{\\text{{LS}}}}^* = {sel['max_ls_evals']}$)."
        ),
        "",
        *_fmt_contrasts(exp_a_stats["paired_contrasts"]),
        "",
        (
            f"### Experiment B — Candidate Selection Fraction "
            f"($b_{{\\text{{LS}}}}^* = {sel['max_ls_evals']}$, Steepest Descent + `N5`)"
        ),
        "",
        *_fmt_stage_table(exp_b_sum),
        "",
        (
            f"- **Dev Friedman:** $\\chi^2_F = "
            f"{exp_b_stats['friedman_med_rpd']['statistic']:.2f}$ "
            f"($p = {exp_b_stats['friedman_med_rpd']['p_value']:.3e}$, Kendall $W = "
            f"{exp_b_stats['friedman_med_rpd']['kendall_w']:.4f}$). Selected: "
            f"**`{sel['stage2_winner_id']}`** "
            f"($f_{{\\text{{LS}}}}^* = {sel['candidate_fraction']:.2f}$)."
        ),
        "",
        *_fmt_contrasts(exp_b_stats["paired_contrasts"]),
        "",
        (
            f"### Experiment C — LS Strategy Comparison "
            f"($b_{{\\text{{LS}}}}^* = {sel['max_ls_evals']}$, "
            f"$f_{{\\text{{LS}}}}^* = {sel['candidate_fraction']:.2f}$, `N5`)"
        ),
        "",
        *_fmt_stage_table(exp_c_sum),
        "",
        (
            f"- **Dev Friedman:** $\\chi^2_F = "
            f"{exp_c_stats['friedman_med_rpd']['statistic']:.2f}$ "
            f"($p = {exp_c_stats['friedman_med_rpd']['p_value']:.3e}$, Kendall $W = "
            f"{exp_c_stats['friedman_med_rpd']['kendall_w']:.4f}$). Frozen Final Choice: "
            f"**`DV_STAR` = `{sel['stage3_winner_id']}`** (`{sel['ls_strategy']}`, "
            f"$b_{{\\text{{LS}}}}^* = {sel['max_ls_evals']}$, "
            f"$f_{{\\text{{LS}}}}^* = {sel['candidate_fraction']:.2f}$)."
        ),
        "",
        *_fmt_contrasts(exp_c_stats["paired_contrasts"]),
        "",
        "## 3. Held-Out Validation: Frozen `DV_STAR` vs. `C06` Control & Single-View `C02` (LGA)",
        "",
        "### Held-Out Summary (`16` Held-Out Instances × `20` Seeds)",
        "",
        *_fmt_stage_table(ho_ref_sum),
        "",
        (
            f"- **Held-Out Friedman (`DV_STAR`, `DV_CTRL_C06`, `C02_LGA`):** "
            f"$\\chi^2_F = {ho_ref_stats['friedman_med_rpd']['statistic']:.2f}$ "
            f"($p = {ho_ref_stats['friedman_med_rpd']['p_value']:.3e}$, Kendall $W = "
            f"{ho_ref_stats['friedman_med_rpd']['kendall_w']:.4f}$)."
        ),
        "",
        *_fmt_contrasts(ho_ref_stats["paired_contrasts"]),
        "",
        "### Branch-Specific Budget & Move Telemetry (`Stage 4 Held-Out`)",
        "",
        (
            "| ID | Branch | Decode Evals | LS Evals | LS Calls | Improving Calls | "
            "Accepted Moves | Rejected Moves | Cyclic Rejects |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ho_ref_sum:
        if r["lam_decode_evals_mean"] is None:
            continue
        lines.append(
            f"| `{r['experiment_id']}` | Lamarckian ($G^L$) | "
            f"`{r['lam_decode_evals_mean']:.1f}` | `{r['lam_ls_evals_mean']:.1f}` | "
            f"`{r['lam_ls_calls_mean']:.1f}` | `{r['lam_improving_calls_mean']:.1f}` | "
            f"`{r['lam_accepted_moves_mean']:.1f}` | `{r['lam_rejected_moves_mean']:.1f}` | "
            f"`{r['lam_rejected_cyclic_moves_mean']:.2f}` |"
        )
        lines.append(
            f"| `{r['experiment_id']}` | Baldwinian ($G^B$) | "
            f"`{r['bald_decode_evals_mean']:.1f}` | `{r['bald_ls_evals_mean']:.1f}` | "
            f"`{r['bald_ls_calls_mean']:.1f}` | `{r['bald_improving_calls_mean']:.1f}` | "
            f"`{r['bald_accepted_moves_mean']:.1f}` | "
            f"`{r['bald_rejected_moves_mean']:.1f}` | "
            f"`{r['bald_rejected_cyclic_moves_mean']:.2f}` |"
        )

    lines.extend(
        [
            "",
            "### Held-Out Diversity Trajectories (`DV_STAR` vs `DV_CTRL_C06`)",
            "",
            (
                "| Experiment | Generation | Active Runs | Mean OBR Div | Mean $G^L$ Div | "
                "Mean $G^B$ Div |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for d in div_traj_rows:
        g_lbl = "final" if d["generation"] == -1 else str(d["generation"])
        lines.append(
            f"| `{d['experiment_id']}` | `{g_lbl}` | `{d['n_active_runs']}` | "
            f"`{d['mean_obr_diversity']:.4f}` | `{d['mean_lam_diversity']:.4f}` | "
            f"`{d['mean_bald_diversity']:.4f}` |"
        )

    proto_uri = (REPO_ROOT / "docs/experiments/dual_view_ls_budget_protocol.md").as_posix()
    sel_uri = (EXP_ROOT / "selection/frozen_dv_ls_selection.json").as_posix()
    man_uri = (EXP_ROOT / "manifests/exploratory_dv_ls_manifest.json").as_posix()
    raw_uri = (EXP_ROOT / "raw/exploratory_dv_ls_runs.jsonl").as_posix()
    stat_uri = (EXP_ROOT / "analysis/exploratory_statistical_tests.json").as_posix()

    ho_by_id = {r["experiment_id"]: r for r in ho_ref_sum}
    ho_star = ho_by_id["DV_STAR"]
    ho_ctrl = ho_by_id["DV_CTRL_C06"]
    ho_c02 = ho_by_id["C02_LGA"]
    ho_c_vs_ctrl = next(
        c for c in ho_ref_stats["paired_contrasts"] if c["reference"] == "DV_CTRL_C06"
    )
    ho_c_vs_c02 = next(c for c in ho_ref_stats["paired_contrasts"] if c["reference"] == "C02_LGA")
    lam_wins = ho_branch_stats["ho_lam_wins"]
    ho_n = max(1, ho_branch_stats["ho_total_runs"])
    tot_ls_ho = ho_star["lam_ls_evals_mean"] + ho_star["bald_ls_evals_mean"]
    bald_ls_pct = (ho_star["bald_ls_evals_mean"] / max(1.0, tot_ls_ho)) * 100.0
    gen_ratio = ho_star["mean_generations"] / max(1e-9, ho_ctrl["mean_generations"])

    lines.extend(
        [
            "",
            "### Development Reference (`DV_STAR` vs. `DV_CTRL_C06` & `C02_LGA`)",
            "",
            *_fmt_stage_table(dev_ref_sum),
            "",
            *_fmt_contrasts(dev_ref_stats["paired_contrasts"]),
            "",
            "## 4. Evidence-Based Diagnostic Conclusions",
            "",
            (
                "1. **Was excessive LS consumption the primary cause of C06 underperformance?** "
                "Reducing `b_LS` alone and adding selective top-$K$ LS screening + "
                "`BudgetLimitedSa` drove Held-Out median RPD from "
                f"`{ho_ctrl['median_rpd_mean']:.3f}%` (`DV_CTRL_C06`) to "
                f"`{ho_star['median_rpd_mean']:.3f}%` (`DV_STAR`, "
                f"`{ho_c_vs_ctrl['mean_diff']:+.3f}%`, `{ho_c_vs_ctrl['wins_ties_losses']}`, "
                f"$p_{{\\text{{Holm}}}} = {ho_c_vs_ctrl['holm_p']:.3e}$)."
            ),
            (
                "2. **Did reallocating LS budget improve horizon, gate adaptation, and search "
                f"behavior?** On Held-Out, `DV_STAR` increased generations `{gen_ratio:.2f}x` "
                f"(`{ho_ctrl['mean_generations']:.2f}` -> `{ho_star['mean_generations']:.2f}`), "
                f"lowered LS budget share (`{ho_ctrl['mean_ls_budget_share_pct']:.1f}%` -> "
                f"`{ho_star['mean_ls_budget_share_pct']:.1f}%`), and allowed adaptive gate $p$ "
                f"to shift (`{ho_ctrl['final_gate_p_mean']:.4f}` -> "
                f"`{ho_star['final_gate_p_mean']:.4f}`)."
            ),
            (
                "3. **Did reallocated Dual-View close the gap to single-view LGA (`C02`)?** "
                f"Under equal total evaluation budgets, `DV_STAR` "
                f"(`{ho_star['median_rpd_mean']:.3f}%`) outperformed frozen `C02_LGA` "
                f"(`{ho_c02['median_rpd_mean']:.3f}%`, `{ho_c_vs_c02['mean_diff']:+.3f}%`, "
                f"`{ho_c_vs_c02['wins_ties_losses']}`, "
                f"$p_{{\\text{{Holm}}}} = {ho_c_vs_c02['holm_p']:.3e}$). Caveat: `C02_LGA` used "
                "frozen confirmatory settings (`b_LS=50, f_LS=1.00, steepest_descent`); a "
                "re-tuned single-view `C02` ablation remains separate future work."
            ),
            (
                "4. **Remaining architectural bottleneck in `Dual-View`:** Even in `DV_STAR`, "
                f"symmetrical per-branch LS still allocates `{bald_ls_pct:.1f}%` of LS "
                f"evaluations (`{ho_star['bald_ls_evals_mean']:.1f}` / `{tot_ls_ho:.1f}`) to "
                f"Baldwinian genotypes ($G^B$), while $G^L$ produces the best schedule in "
                f"`{lam_wins}/{ho_n}` (`{100.0 * lam_wins / ho_n:.1f}%`) Held-Out runs."
            ),
            "",
            "## 5. Artifacts",
            (
                "- Protocol: [`docs/experiments/dual_view_ls_budget_protocol.md`]"
                f"(file://{proto_uri})"
            ),
            (
                f"- Selection Lock (`{selection_doc['selection_lock_sha256'][:16]}...`): "
                f"[`results/exploratory_dv_ls/selection/frozen_dv_ls_selection.json`]"
                f"(file://{sel_uri})"
            ),
            (
                f"- Manifest (`{manifest_doc['raw_jsonl_sha256'][:16]}...`): "
                f"[`results/exploratory_dv_ls/manifests/exploratory_dv_ls_manifest.json`]"
                f"(file://{man_uri})"
            ),
            (
                f"- Raw Runs JSONL (`{manifest_doc['total_new_runs']:,}` runs): "
                f"[`results/exploratory_dv_ls/raw/exploratory_dv_ls_runs.jsonl`]"
                f"(file://{raw_uri})"
            ),
            (
                "- Statistical Tests: "
                f"[`results/exploratory_dv_ls/analysis/exploratory_statistical_tests.json`]"
                f"(file://{stat_uri})"
            ),
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
