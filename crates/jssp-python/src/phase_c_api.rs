//! PyO3 entrypoints for Phase C Memetic & Dual-View GAs (`C01`–`C11`) and Final Pipeline (`P5`).

use std::time::Instant;

use jssp_core::decoder::decode_obr as core_decode_obr;
use jssp_core::feasibility::validate_schedule as core_validate_schedule;
use jssp_core::initialization::InitializerId;
use jssp_core::memetic::{
    PhaseCBatchTask, PhaseCConfigId, PhaseCGaConfig,
    default_phase_c_budget as core_default_c_budget, run_phase_c_batch as core_run_phase_c_batch,
    run_phase_c_ga as core_run_phase_c_ga,
};
use jssp_core::metaheuristics::{
    PhaseBConfigId, run_phase_b_from_schedule as core_run_phase_b_from_schedule,
};
use jssp_core::neighborhoods::NeighborhoodId;
use jssp_core::representation::ObrChromosome;
use pyo3::prelude::*;

use crate::instance_api::{jssp_err_to_py, parse_instance};
use crate::phase_a_api::build_seed_context;
use crate::result_api::MachineSequenceList;

type PhaseCConfigMetaTuple = (&'static str, &'static str, bool, bool);

type PhaseCSummaryCoreTuple = (
    &'static str,
    &'static str,
    Option<&'static str>,
    Option<&'static str>,
    usize,
    usize,
    usize,
    usize,
    usize,
);

type PhaseCSummaryPerfTuple = (usize, usize, usize, u64, usize, usize, bool, f64, u64);

type PhaseCDiversityTuple = (
    f64,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<u64>,
    Option<u64>,
    usize,
    usize,
    usize,
    usize,
);

type PhaseCRunTuple = (
    PhaseCSummaryCoreTuple,
    PhaseCSummaryPerfTuple,
    PhaseCDiversityTuple,
    Vec<(usize, f64)>,
    Vec<(usize, u64)>,
    Vec<usize>,
    MachineSequenceList,
);

type PhaseCBatchTaskSpec = (String, u32, Option<u64>, String, String, String);

type FinalPipelineRunTuple = (
    &'static str,
    usize,
    usize,
    u64,
    u64,
    usize,
    usize,
    bool,
    f64,
    u64,
);

fn parse_strategy(name_or_code: &str) -> PyResult<jssp_core::metaheuristics::PhaseBStrategy> {
    use jssp_core::metaheuristics::PhaseBStrategy;
    match name_or_code {
        "steepest_descent" | "Steepest Descent" => Ok(PhaseBStrategy::SteepestDescent),
        "first_improvement" | "First Improvement" => Ok(PhaseBStrategy::FirstImprovement),
        "random_improvement" | "Random Improvement" => Ok(PhaseBStrategy::RandomImprovement),
        "tabu_search" | "Tabu Search" => Ok(PhaseBStrategy::TabuSearch),
        "ils" | "Iterated Local Search (ILS)" => Ok(PhaseBStrategy::IteratedLocalSearch),
        "simulated_annealing" | "Simulated Annealing (SA)" => {
            Ok(PhaseBStrategy::SimulatedAnnealing)
        }
        "ils_mh" | "Metropolis-Filtered ILS + Steepest Descent" => {
            Ok(PhaseBStrategy::MetropolisFilteredIls)
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown Phase B strategy '{other}'"
        ))),
    }
}

fn phase_c_result_to_tuple(res: jssp_core::memetic::PhaseCRunResult) -> PhaseCRunTuple {
    let summary_core: PhaseCSummaryCoreTuple = (
        res.config_id,
        res.initializer_id,
        res.ls_neighborhood,
        res.ls_strategy,
        res.genotype_capacity,
        res.evaluation_budget,
        res.evals_used,
        res.init_evals,
        res.search_evals,
    );
    let summary_perf: PhaseCSummaryPerfTuple = (
        res.decode_evals,
        res.ls_evals,
        res.generations_executed,
        res.best_makespan,
        res.best_found_eval,
        res.best_found_generation,
        res.feasible_final,
        res.anytime_cmax_auc,
        res.total_elapsed_ns,
    );
    let div: PhaseCDiversityTuple = (
        res.final_obr_diversity,
        res.final_lam_diversity,
        res.final_bald_diversity,
        res.final_gate_p,
        res.lam_best_makespan,
        res.bald_best_makespan,
        res.crossover_count,
        res.mutation_count,
        res.writeback_count,
        res.migrations_applied,
    );
    (
        summary_core,
        summary_perf,
        div,
        res.gate_p_trajectory,
        res.best_milestones,
        res.best_chromosome,
        res.best_machine_sequences,
    )
}

/// Returns `(code, name, is_memetic, is_dual_view_paired)` for all 11 Phase C configurations.
#[pyfunction]
pub fn list_phase_c_configs() -> Vec<PhaseCConfigMetaTuple> {
    PhaseCConfigId::ALL
        .iter()
        .map(|&c| (c.code(), c.name(), c.is_memetic(), c.is_dual_view_paired()))
        .collect()
}

/// Computes Phase C default total objective-evaluation budget matching Phase A (`B_GA`).
#[pyfunction]
pub fn default_phase_c_budget(num_jobs: usize, num_machines: usize) -> usize {
    core_default_c_budget(num_jobs, num_machines)
}

/// Executes a single Phase C (`C01`–`C11`) run inside Rust.
#[pyfunction]
#[pyo3(signature = (
    jobs,
    config_id,
    instance_id,
    initializer_id="A07",
    ls_neighborhood="N7",
    ls_strategy="steepest_descent",
    max_ls_evals=50,
    pop_size=64,
    tournament_size=2,
    crossover_rate=0.85,
    mutation_rate=0.15,
    elitism_count=2,
    evaluation_budget=None,
    replicate_index=0,
    master_seed_override=None,
    pairing_mode="paired_init",
    pairing_group="phase_c_shared_init",
    configuration_id=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn run_phase_c_single(
    jobs: Vec<Vec<(usize, u64)>>,
    config_id: &str,
    instance_id: String,
    initializer_id: &str,
    ls_neighborhood: &str,
    ls_strategy: &str,
    max_ls_evals: usize,
    pop_size: usize,
    tournament_size: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    elitism_count: usize,
    evaluation_budget: Option<usize>,
    replicate_index: u32,
    master_seed_override: Option<u64>,
    pairing_mode: &str,
    pairing_group: &str,
    configuration_id: Option<String>,
) -> PyResult<PhaseCRunTuple> {
    let instance = parse_instance(jobs)?;
    let cfg_id = PhaseCConfigId::from_code(config_id).map_err(jssp_err_to_py)?;
    let init_id = InitializerId::from_code(initializer_id).map_err(jssp_err_to_py)?;
    let nb_id = NeighborhoodId::from_code(ls_neighborhood).map_err(jssp_err_to_py)?;
    let strat = parse_strategy(ls_strategy)?;
    let budget = evaluation_budget
        .unwrap_or_else(|| core_default_c_budget(instance.num_jobs(), instance.num_machines()));
    let config = PhaseCGaConfig {
        config_id: cfg_id,
        initializer: init_id,
        population_size: pop_size,
        tournament_size,
        crossover_rate,
        mutation_rate,
        elitism_count,
        evaluation_budget: budget,
        local_search: jssp_core::memetic::MemeticLsConfig {
            neighborhood: nb_id,
            strategy: strat,
            max_ls_evals,
        },
    };
    let cfg_label = configuration_id.unwrap_or_else(|| config_id.to_owned());
    let context = build_seed_context(
        replicate_index,
        master_seed_override,
        instance_id,
        pairing_mode,
        pairing_group.to_owned(),
        cfg_label,
    )?;
    let mut streams = context.all_streams();
    let res = core_run_phase_c_ga(&instance, &config, &mut streams).map_err(jssp_err_to_py)?;
    Ok(phase_c_result_to_tuple(res))
}

/// Runs a multi-threaded Rust batch of Phase C (`C01`–`C11`) runs releasing the GIL (`py.detach`).
#[pyfunction]
#[pyo3(signature = (
    jobs,
    instance_id,
    tasks,
    initializer_id="A07",
    ls_neighborhood="N7",
    ls_strategy="steepest_descent",
    max_ls_evals=50,
    pop_size=64,
    tournament_size=2,
    crossover_rate=0.85,
    mutation_rate=0.15,
    elitism_count=2,
    evaluation_budget=None,
    num_threads=0,
))]
#[allow(clippy::too_many_arguments)]
pub fn run_phase_c_batch(
    py: Python<'_>,
    jobs: Vec<Vec<(usize, u64)>>,
    instance_id: String,
    tasks: Vec<PhaseCBatchTaskSpec>,
    initializer_id: &str,
    ls_neighborhood: &str,
    ls_strategy: &str,
    max_ls_evals: usize,
    pop_size: usize,
    tournament_size: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    elitism_count: usize,
    evaluation_budget: Option<usize>,
    num_threads: usize,
) -> PyResult<Vec<PhaseCRunTuple>> {
    let instance = parse_instance(jobs)?;
    let init_id = InitializerId::from_code(initializer_id).map_err(jssp_err_to_py)?;
    let nb_id = NeighborhoodId::from_code(ls_neighborhood).map_err(jssp_err_to_py)?;
    let strat = parse_strategy(ls_strategy)?;
    let budget = evaluation_budget
        .unwrap_or_else(|| core_default_c_budget(instance.num_jobs(), instance.num_machines()));

    let mut batch_tasks = Vec::with_capacity(tasks.len());
    for (cfg_code, rep_idx, seed_override, pairing_mode, pairing_group, cfg_label) in tasks {
        let cfg_id = PhaseCConfigId::from_code(&cfg_code).map_err(jssp_err_to_py)?;
        let config = PhaseCGaConfig {
            config_id: cfg_id,
            initializer: init_id,
            population_size: pop_size,
            tournament_size,
            crossover_rate,
            mutation_rate,
            elitism_count,
            evaluation_budget: budget,
            local_search: jssp_core::memetic::MemeticLsConfig {
                neighborhood: nb_id,
                strategy: strat,
                max_ls_evals,
            },
        };
        config.validate().map_err(jssp_err_to_py)?;
        let context = build_seed_context(
            rep_idx,
            seed_override,
            instance_id.clone(),
            &pairing_mode,
            pairing_group,
            cfg_label,
        )?;
        batch_tasks.push(PhaseCBatchTask {
            config,
            seed_context: context,
        });
    }

    let raw_results = py
        .detach(|| core_run_phase_c_batch(&instance, &batch_tasks, num_threads))
        .map_err(jssp_err_to_py)?;

    Ok(raw_results
        .into_iter()
        .map(phase_c_result_to_tuple)
        .collect())
}

/// Executes the Final Integrated Optimization Pipeline (`P5: A* -> C* (85%) -> B* (15%)`)
/// in a single thread-parallel Rust batch under exact total `evaluation_budget`, independently
/// M2-validating both intermediate `C*` and final `B*` polished schedules.
#[pyfunction]
#[pyo3(signature = (
    jobs,
    instance_id,
    replicates,
    frozen_a_id,
    frozen_b_id,
    frozen_c_id,
    max_ls_evals=50,
    evaluation_budget=None,
    stage_c_fraction=0.85,
    num_threads=0,
))]
#[allow(clippy::too_many_arguments)]
pub fn run_integrated_acb_batch(
    py: Python<'_>,
    jobs: Vec<Vec<(usize, u64)>>,
    instance_id: String,
    replicates: Vec<u32>,
    frozen_a_id: &str,
    frozen_b_id: &str,
    frozen_c_id: &str,
    max_ls_evals: usize,
    evaluation_budget: Option<usize>,
    stage_c_fraction: f64,
    num_threads: usize,
) -> PyResult<Vec<FinalPipelineRunTuple>> {
    let instance = parse_instance(jobs)?;
    let init_id = InitializerId::from_code(frozen_a_id).map_err(jssp_err_to_py)?;
    let b_cfg = PhaseBConfigId::from_code(frozen_b_id).map_err(jssp_err_to_py)?;
    let c_cfg_id = PhaseCConfigId::from_code(frozen_c_id).map_err(jssp_err_to_py)?;
    if !(0.1..=0.95).contains(&stage_c_fraction) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "stage_c_fraction must lie in [0.1, 0.95]",
        ));
    }

    let total_budget = evaluation_budget
        .unwrap_or_else(|| core_default_c_budget(instance.num_jobs(), instance.num_machines()));
    let c_budget = ((total_budget as f64) * stage_c_fraction).round() as usize;
    let c_budget = c_budget.max(64).min(total_budget.saturating_sub(1));

    let results = py
        .detach(|| {
            let resolved_threads = if num_threads == 0 {
                std::thread::available_parallelism()
                    .map_or(1, std::num::NonZeroUsize::get)
                    .min(replicates.len().max(1))
            } else {
                num_threads.min(replicates.len().max(1))
            };
            let run_one = |rep: u32| -> Result<FinalPipelineRunTuple, jssp_core::error::JsspError> {
                let t0 = Instant::now();
                let ctx = build_seed_context_rust(rep, &instance_id, "P5_Integrated_A_C_B")?;
                let mut streams = ctx.all_streams();
                let cfg_c = PhaseCGaConfig::standard(
                    c_cfg_id,
                    init_id,
                    c_budget,
                    b_cfg.neighborhood(),
                    jssp_core::metaheuristics::PhaseBStrategy::SteepestDescent,
                    max_ls_evals,
                );
                let c_res = core_run_phase_c_ga(&instance, &cfg_c, &mut streams)?;
                let remaining = total_budget.saturating_sub(c_res.evals_used).max(1);

                let c_obr = ObrChromosome::from_job_indices(&c_res.best_chromosome);
                let c_sched = core_decode_obr(&instance, &c_obr)?;
                core_validate_schedule(&instance, &c_sched)?;

                let b_res = core_run_phase_b_from_schedule(
                    &instance,
                    b_cfg,
                    &c_sched,
                    remaining,
                    &mut streams,
                )?;
                let total_evals = c_res.evals_used + b_res.evals_used;
                let final_best = b_res.best_makespan.min(c_res.best_makespan);
                let polish_delta = c_res.best_makespan.saturating_sub(final_best);
                let elapsed_ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
                let comb_auc = if total_evals > 0 {
                    (c_res.anytime_cmax_auc * (c_res.evals_used as f64)
                        + b_res.anytime_cmax_auc * (b_res.evals_used as f64))
                        / (total_evals as f64)
                } else {
                    final_best as f64
                };
                Ok((
                    "P5_Integrated_A_C_B",
                    total_budget,
                    total_evals,
                    c_res.best_makespan,
                    final_best,
                    c_res.best_found_eval,
                    c_res.evals_used + b_res.best_found_eval,
                    b_res.feasible_final
                        && c_res.feasible_final
                        && polish_delta <= c_res.best_makespan,
                    comb_auc,
                    elapsed_ns,
                ))
            };

            if resolved_threads <= 1 || replicates.len() <= 1 {
                replicates.iter().map(|&r| run_one(r)).collect()
            } else {
                let chunk_sz = replicates.len().div_ceil(resolved_threads);
                let chunk_out: Vec<
                    Result<Vec<FinalPipelineRunTuple>, jssp_core::error::JsspError>,
                > = std::thread::scope(|s| {
                    let handles: Vec<_> = replicates
                        .chunks(chunk_sz)
                        .map(|ch| {
                            s.spawn(move || {
                                let mut v = Vec::with_capacity(ch.len());
                                for &r in ch {
                                    v.push(run_one(r)?);
                                }
                                Ok(v)
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| match h.join() {
                            Ok(r) => r,
                            Err(_) => {
                                Err(jssp_core::error::JsspError::InvalidGeneticConfiguration {
                                    detail: "P5 worker panic".to_owned(),
                                })
                            }
                        })
                        .collect()
                });
                let mut flat = Vec::with_capacity(replicates.len());
                for chunk_res in chunk_out {
                    flat.extend(chunk_res?);
                }
                Ok(flat)
            }
        })
        .map_err(jssp_err_to_py)?;
    Ok(results)
}

fn build_seed_context_rust(
    replicate_index: u32,
    instance_id: &str,
    cfg_id: &str,
) -> Result<jssp_core::rng::SeedDerivationContext, jssp_core::error::JsspError> {
    jssp_core::rng::SeedDerivationContext::from_frozen_replicate(
        replicate_index,
        instance_id,
        jssp_core::rng::PairingPolicy::PairedInitialAndScramble {
            pairing_group: "final_pipeline_shared_init".to_owned(),
            configuration_id: cfg_id.to_owned(),
        },
    )
}
