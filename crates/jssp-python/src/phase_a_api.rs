//! PyO3 entrypoints for Phase A initializers (`A01`–`A24`) and Classical GA (`C01`).

use jssp_core::genetic::{
    ClassicalGaConfig, PhaseABatchTask, default_phase_a_ga_budget as core_default_budget,
    evaluate_initializer_only as core_eval_initializer_only,
    run_phase_a_batch as core_run_phase_a_batch, run_phase_a_ga as core_run_phase_a_ga,
};
use jssp_core::initialization::InitializerId;
use jssp_core::rng::{SeedDerivationContext, master_seed_for_replicate};
use jssp_core::telemetry::{GenerationSnapshot, PhaseARunResult, PopulationDiagnostics};
use pyo3::prelude::*;

use crate::instance_api::{jssp_err_to_py, parse_instance};
use crate::run_api::build_pairing_policy;

type PopDiagTuple = (
    (usize, usize, usize, usize),
    (u64, u64, f64, f64, f64),
    (f64, f64, f64, f64, Option<f64>),
    (u64, u64),
);

type RunSummaryTuple = (
    &'static str,
    usize,
    usize,
    usize,
    usize,
    u64,
    usize,
    usize,
    String,
    f64,
);

type TrajectoryTuple = (usize, usize, u64, f64, f64, usize);

type PhaseARunTuple = (
    RunSummaryTuple,
    (usize, usize, u64),
    PopDiagTuple,
    PopDiagTuple,
    Vec<usize>,
    Vec<TrajectoryTuple>,
);

type BatchTaskSpec = (String, u32, Option<u64>, String, String, String);
type InitializerMetaTuple = (&'static str, &'static str, Option<f64>, bool, bool, bool);

fn diag_to_tuple(diag: &PopulationDiagnostics) -> PopDiagTuple {
    (
        (
            diag.population_size,
            diag.feasible_count,
            diag.unique_chromosomes,
            diag.unique_schedules,
        ),
        (
            diag.best_makespan,
            diag.worst_makespan,
            diag.mean_makespan,
            diag.median_makespan,
            diag.std_makespan,
        ),
        (
            diag.obr_duplicate_rate,
            diag.schedule_duplicate_rate,
            diag.obr_hamming_diversity,
            diag.schedule_diversity,
            diag.qmc_1d_ks_discrepancy,
        ),
        (diag.generation_elapsed_ns, diag.decode_elapsed_ns),
    )
}

fn select_trajectory(
    trajectory: &[GenerationSnapshot],
    trajectory_mode: &str,
) -> PyResult<Vec<TrajectoryTuple>> {
    match trajectory_mode {
        "none" => Ok(Vec::new()),
        "full" => Ok(trajectory
            .iter()
            .map(|s| {
                (
                    s.generation,
                    s.evals_used,
                    s.best_makespan,
                    s.mean_makespan,
                    s.obr_hamming_diversity,
                    s.unique_chromosomes,
                )
            })
            .collect()),
        "sampled" => {
            if trajectory.len() <= 21 {
                return select_trajectory(trajectory, "full");
            }
            let last_idx = trajectory.len() - 1;
            let mut out = Vec::with_capacity(21);
            let mut prev_picked = usize::MAX;
            for step in 0..=20 {
                let idx = (step * last_idx) / 20;
                if idx != prev_picked {
                    let s = &trajectory[idx];
                    out.push((
                        s.generation,
                        s.evals_used,
                        s.best_makespan,
                        s.mean_makespan,
                        s.obr_hamming_diversity,
                        s.unique_chromosomes,
                    ));
                    prev_picked = idx;
                }
            }
            Ok(out)
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown trajectory_mode '{other}', expected 'full', 'sampled', or 'none'"
        ))),
    }
}

fn result_to_tuple(res: PhaseARunResult, trajectory_mode: &str) -> PyResult<PhaseARunTuple> {
    let traj = select_trajectory(&res.trajectory, trajectory_mode)?;
    Ok((
        (
            res.initializer_id,
            res.pop_size,
            res.evaluation_budget,
            res.evals_used,
            res.generations_executed,
            res.best_makespan,
            res.best_found_eval,
            res.best_found_generation,
            res.best_schedule_hash,
            res.anytime_cmax_auc,
        ),
        (
            res.crossover_count,
            res.mutation_count,
            res.total_elapsed_ns,
        ),
        diag_to_tuple(&res.initial),
        diag_to_tuple(&res.final_pop),
        res.best_chromosome,
        traj,
    ))
}

fn build_seed_context(
    replicate_index: u32,
    master_seed_override: Option<u64>,
    instance_id: String,
    pairing_mode: &str,
    pairing_group: String,
    configuration_id: String,
) -> PyResult<SeedDerivationContext> {
    let policy = build_pairing_policy(pairing_mode, pairing_group, configuration_id)?;
    match master_seed_override {
        Some(seed) => SeedDerivationContext::with_explicit_master_seed(
            seed,
            replicate_index,
            instance_id,
            policy,
        )
        .map_err(jssp_err_to_py),
        None => {
            let seed = master_seed_for_replicate(replicate_index).map_err(jssp_err_to_py)?;
            SeedDerivationContext::with_explicit_master_seed(
                seed,
                replicate_index,
                instance_id,
                policy,
            )
            .map_err(jssp_err_to_py)
        }
    }
}

/// Returns metadata for all 24 Phase A initializers (`A01`–`A24`).
#[pyfunction]
pub fn list_phase_a_initializers() -> Vec<InitializerMetaTuple> {
    InitializerId::ALL
        .iter()
        .map(|&id| {
            (
                id.code(),
                id.name(),
                id.alpha(),
                id.is_deterministic(),
                id.uses_sobol(),
                id.has_dyadic_sobol_net_balance(64),
            )
        })
        .collect()
}

/// Computes the frozen size-scaled Phase A GA objective evaluation budget.
#[pyfunction]
pub fn default_phase_a_ga_budget(num_jobs: usize, num_machines: usize) -> usize {
    core_default_budget(num_jobs, num_machines)
}

/// Generates and M2-decodes a single initial population (Layer A standalone inspection).
#[pyfunction]
#[pyo3(signature = (
    jobs,
    initializer_id,
    instance_id,
    pop_size=64,
    replicate_index=0,
    master_seed_override=None,
    pairing_mode="paired_init",
    pairing_group="phase_a_shared_init",
    configuration_id=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_phase_a_initializer(
    jobs: Vec<Vec<(usize, u64)>>,
    initializer_id: String,
    instance_id: String,
    pop_size: usize,
    replicate_index: u32,
    master_seed_override: Option<u64>,
    pairing_mode: &str,
    pairing_group: &str,
    configuration_id: Option<String>,
) -> PyResult<(PopDiagTuple, Vec<Vec<usize>>, Vec<u64>)> {
    let instance = parse_instance(jobs)?;
    let init = InitializerId::from_code(&initializer_id).map_err(jssp_err_to_py)?;
    let cfg_id = configuration_id.unwrap_or_else(|| initializer_id.clone());
    let context = build_seed_context(
        replicate_index,
        master_seed_override,
        instance_id,
        pairing_mode,
        pairing_group.to_owned(),
        cfg_id,
    )?;
    let mut streams = context.all_streams();
    let (diag, chrs, scheds) = core_eval_initializer_only(&instance, init, pop_size, &mut streams)
        .map_err(jssp_err_to_py)?;
    let raw_chrs: Vec<Vec<usize>> = chrs.iter().map(|c| c.to_job_indices()).collect();
    let cmaxes: Vec<u64> = scheds.iter().map(|s| s.makespan()).collect();
    Ok((diag_to_tuple(&diag), raw_chrs, cmaxes))
}

/// Executes a single Phase A `(initializer -> Classical GA C01)` experiment run.
#[pyfunction]
#[pyo3(signature = (
    jobs,
    initializer_id,
    instance_id,
    pop_size=64,
    tournament_size=2,
    crossover_rate=0.85,
    mutation_rate=0.15,
    elitism_count=2,
    evaluation_budget=None,
    replicate_index=0,
    master_seed_override=None,
    pairing_mode="paired_init",
    pairing_group="phase_a_shared_init",
    configuration_id=None,
    trajectory_mode="full",
))]
#[allow(clippy::too_many_arguments)]
pub fn run_phase_a_single(
    jobs: Vec<Vec<(usize, u64)>>,
    initializer_id: String,
    instance_id: String,
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
    trajectory_mode: &str,
) -> PyResult<PhaseARunTuple> {
    let instance = parse_instance(jobs)?;
    let init = InitializerId::from_code(&initializer_id).map_err(jssp_err_to_py)?;
    let budget = evaluation_budget
        .unwrap_or_else(|| core_default_budget(instance.num_jobs(), instance.num_machines()));
    let config = ClassicalGaConfig {
        population_size: pop_size,
        tournament_size,
        crossover_rate,
        mutation_rate,
        elitism_count,
        evaluation_budget: budget,
    };
    let cfg_id = configuration_id.unwrap_or_else(|| initializer_id.clone());
    let context = build_seed_context(
        replicate_index,
        master_seed_override,
        instance_id,
        pairing_mode,
        pairing_group.to_owned(),
        cfg_id,
    )?;
    let mut streams = context.all_streams();
    let res =
        core_run_phase_a_ga(&instance, init, &config, &mut streams).map_err(jssp_err_to_py)?;
    result_to_tuple(res, trajectory_mode)
}

/// Executes a thread-parallel batch of Phase A `(initializer, replicate)` runs on one instance
/// while releasing the Python GIL (`py.detach`).
#[pyfunction]
#[pyo3(signature = (
    jobs,
    instance_id,
    tasks,
    pop_size=64,
    tournament_size=2,
    crossover_rate=0.85,
    mutation_rate=0.15,
    elitism_count=2,
    evaluation_budget=None,
    num_threads=0,
    trajectory_mode="sampled",
))]
#[allow(clippy::too_many_arguments)]
pub fn run_phase_a_batch(
    py: Python<'_>,
    jobs: Vec<Vec<(usize, u64)>>,
    instance_id: String,
    tasks: Vec<BatchTaskSpec>,
    pop_size: usize,
    tournament_size: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    elitism_count: usize,
    evaluation_budget: Option<usize>,
    num_threads: usize,
    trajectory_mode: &str,
) -> PyResult<Vec<PhaseARunTuple>> {
    let instance = parse_instance(jobs)?;
    let budget = evaluation_budget
        .unwrap_or_else(|| core_default_budget(instance.num_jobs(), instance.num_machines()));
    let config = ClassicalGaConfig {
        population_size: pop_size,
        tournament_size,
        crossover_rate,
        mutation_rate,
        elitism_count,
        evaluation_budget: budget,
    };
    config.validate().map_err(jssp_err_to_py)?;

    let mut batch_tasks = Vec::with_capacity(tasks.len());
    for (init_code, rep_idx, seed_override, pairing_mode, pairing_group, cfg_id) in tasks {
        let init = InitializerId::from_code(&init_code).map_err(jssp_err_to_py)?;
        let context = build_seed_context(
            rep_idx,
            seed_override,
            instance_id.clone(),
            &pairing_mode,
            pairing_group,
            cfg_id,
        )?;
        batch_tasks.push(PhaseABatchTask {
            initializer: init,
            seed_context: context,
            config,
        });
    }

    let raw_results = py
        .detach(|| core_run_phase_a_batch(&instance, &batch_tasks, num_threads))
        .map_err(jssp_err_to_py)?;

    let mut out = Vec::with_capacity(raw_results.len());
    for res in raw_results {
        out.push(result_to_tuple(res, trajectory_mode)?);
    }
    Ok(out)
}
