//! PyO3 entrypoints for Phase B neighborhoods, local search, and metaheuristics (`B01`–`B24`, `B29`–`B32`).

use jssp_core::decoder::decode_obr as core_decode_obr;
use jssp_core::initialization::InitializerId;
use jssp_core::local_search::PhaseBRunResult;
use jssp_core::metaheuristics::{
    PhaseBBatchTask, PhaseBConfigId, default_phase_b_budget as core_default_phase_b_budget,
    generate_matched_phase_b_start as core_generate_matched_phase_b_start,
    run_phase_b_batch as core_run_phase_b_batch,
    run_phase_b_from_schedule as core_run_phase_b_from_schedule,
};
use jssp_core::representation::ObrChromosome;
use pyo3::prelude::*;

use crate::instance_api::{jssp_err_to_py, parse_instance};
use crate::phase_a_api::build_seed_context;
use crate::result_api::{DecodedSchedulePayload, MachineSequenceList, schedule_to_py_payload};

type PhaseBConfigMetaTuple = (&'static str, &'static str, &'static str);

type PhaseBSummaryCoreTuple = (
    &'static str,
    &'static str,
    usize,
    usize,
    u64,
    u64,
    u64,
    usize,
);

type PhaseBStatusMetaTuple = (bool, bool, &'static str, f64, u64);

type PhaseBCountersPart1 = (usize, usize, usize, usize, usize, usize, usize);
type PhaseBCountersPart2 = (usize, usize, usize, usize, usize, usize, usize);

type PhaseBRunTuple = (
    PhaseBSummaryCoreTuple,
    PhaseBStatusMetaTuple,
    PhaseBCountersPart1,
    PhaseBCountersPart2,
    Vec<(usize, u64)>,
    MachineSequenceList,
);

type PhaseBBatchTaskSpec = (String, u32, Option<u64>, String, String, String);

fn phase_b_result_to_tuple(res: PhaseBRunResult) -> PhaseBRunTuple {
    let core_summary: PhaseBSummaryCoreTuple = (
        res.config_id,
        res.neighborhood_id,
        res.evaluation_budget,
        res.evals_used,
        res.initial_makespan,
        res.final_makespan,
        res.best_makespan,
        res.best_found_eval,
    );
    let status_meta: PhaseBStatusMetaTuple = (
        res.feasible_final,
        res.reached_local_optimum,
        res.terminated_reason.as_str(),
        res.anytime_cmax_auc,
        res.total_elapsed_ns,
    );
    let c = res.counters;
    let c1: PhaseBCountersPart1 = (
        c.iterations,
        c.generated_moves,
        c.duplicate_pruned_moves,
        c.infeasible_precond_moves,
        c.clipped_moves,
        c.evaluated_moves,
        c.feasible_moves,
    );
    let c2: PhaseBCountersPart2 = (
        c.rejected_cyclic_moves,
        c.improving_moves,
        c.accepted_moves,
        c.adapter_left_shift_count,
        c.perturbations_applied,
        c.perturbations_accepted,
        c.restarts_triggered,
    );
    (
        core_summary,
        status_meta,
        c1,
        c2,
        res.best_milestones,
        res.best_machine_sequences,
    )
}

/// Returns metadata `(config_id, strategy_name, neighborhood_code)` for all 28 active
/// Phase B configurations (`B01`–`B24`, `B29`–`B32`; `B25`–`B28` excluded per `D04-A`).
#[pyfunction]
pub fn list_phase_b_configs() -> Vec<PhaseBConfigMetaTuple> {
    PhaseBConfigId::ALL
        .iter()
        .map(|&id| (id.code(), id.strategy().name(), id.neighborhood().code()))
        .collect()
}

/// Computes Phase B budget `multiplier * J * M` (`multiplier = 100` screening, `500` confirmatory).
#[pyfunction]
#[pyo3(signature = (num_jobs, num_machines, multiplier=100))]
pub fn default_phase_b_budget(num_jobs: usize, num_machines: usize, multiplier: usize) -> usize {
    core_default_phase_b_budget(num_jobs, num_machines, multiplier)
}

/// Generates and decodes the matched starting `(obr_genes, schedule)` from frozen Phase A
/// winner `initializer_id` using the same seed derivation context as Phase B runs.
#[pyfunction]
#[pyo3(signature = (
    jobs,
    initializer_id="A07",
    instance_id="instance",
    pop_size=64,
    replicate_index=0,
    master_seed_override=None,
    pairing_mode="paired_init",
    pairing_group="phase_b_matched_start",
    configuration_id=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn generate_matched_phase_b_start(
    jobs: Vec<Vec<(usize, u64)>>,
    initializer_id: &str,
    instance_id: &str,
    pop_size: usize,
    replicate_index: u32,
    master_seed_override: Option<u64>,
    pairing_mode: &str,
    pairing_group: &str,
    configuration_id: Option<String>,
) -> PyResult<(Vec<usize>, DecodedSchedulePayload)> {
    let instance = parse_instance(jobs)?;
    let init = InitializerId::from_code(initializer_id).map_err(jssp_err_to_py)?;
    let cfg_id = configuration_id.unwrap_or_else(|| initializer_id.to_owned());
    let context = build_seed_context(
        replicate_index,
        master_seed_override,
        instance_id.to_owned(),
        pairing_mode,
        pairing_group.to_owned(),
        cfg_id,
    )?;
    let mut streams = context.all_streams();
    let (obr, sched) = core_generate_matched_phase_b_start(&instance, init, pop_size, &mut streams)
        .map_err(jssp_err_to_py)?;
    Ok((obr.to_job_indices(), schedule_to_py_payload(&sched)))
}

/// Executes one Phase B configuration from either a matched `initializer_id` start or an explicit
/// `start_chromosome`, verifying DAG acyclicity and M2 schedule feasibility.
#[pyfunction]
#[pyo3(signature = (
    jobs,
    config_id,
    instance_id,
    initializer_id="A07",
    init_pop_size=64,
    evaluation_budget=None,
    budget_multiplier=100,
    replicate_index=0,
    master_seed_override=None,
    pairing_mode="paired_init",
    pairing_group="phase_b_matched_start",
    configuration_id=None,
    start_chromosome=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn run_phase_b_single(
    jobs: Vec<Vec<(usize, u64)>>,
    config_id: &str,
    instance_id: String,
    initializer_id: &str,
    init_pop_size: usize,
    evaluation_budget: Option<usize>,
    budget_multiplier: usize,
    replicate_index: u32,
    master_seed_override: Option<u64>,
    pairing_mode: &str,
    pairing_group: &str,
    configuration_id: Option<String>,
    start_chromosome: Option<Vec<usize>>,
) -> PyResult<PhaseBRunTuple> {
    let instance = parse_instance(jobs)?;
    let cfg = PhaseBConfigId::from_code(config_id).map_err(jssp_err_to_py)?;
    let init = InitializerId::from_code(initializer_id).map_err(jssp_err_to_py)?;
    let budget = evaluation_budget.unwrap_or_else(|| {
        core_default_phase_b_budget(
            instance.num_jobs(),
            instance.num_machines(),
            budget_multiplier,
        )
    });
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

    let start_schedule = match start_chromosome {
        Some(genes) => {
            let obr = ObrChromosome::from_job_indices(&genes);
            core_decode_obr(&instance, &obr).map_err(jssp_err_to_py)?
        }
        None => {
            let (_, sched) =
                core_generate_matched_phase_b_start(&instance, init, init_pop_size, &mut streams)
                    .map_err(jssp_err_to_py)?;
            sched
        }
    };

    let res = core_run_phase_b_from_schedule(&instance, cfg, &start_schedule, budget, &mut streams)
        .map_err(jssp_err_to_py)?;
    Ok(phase_b_result_to_tuple(res))
}

/// Runs a multi-threaded Rust batch of Phase B `(config_id, replicate)` tasks on one instance
/// while releasing the Python GIL (`py.detach`).
#[pyfunction]
#[pyo3(signature = (
    jobs,
    instance_id,
    tasks,
    initializer_id="A07",
    init_pop_size=64,
    evaluation_budget=None,
    budget_multiplier=100,
    num_threads=0,
))]
#[allow(clippy::too_many_arguments)]
pub fn run_phase_b_batch(
    py: Python<'_>,
    jobs: Vec<Vec<(usize, u64)>>,
    instance_id: String,
    tasks: Vec<PhaseBBatchTaskSpec>,
    initializer_id: &str,
    init_pop_size: usize,
    evaluation_budget: Option<usize>,
    budget_multiplier: usize,
    num_threads: usize,
) -> PyResult<Vec<PhaseBRunTuple>> {
    let instance = parse_instance(jobs)?;
    let init = InitializerId::from_code(initializer_id).map_err(jssp_err_to_py)?;
    let budget = evaluation_budget.unwrap_or_else(|| {
        core_default_phase_b_budget(
            instance.num_jobs(),
            instance.num_machines(),
            budget_multiplier,
        )
    });

    let mut batch_tasks = Vec::with_capacity(tasks.len());
    for (cfg_code, rep_idx, seed_override, pairing_mode, pairing_group, cfg_label) in tasks {
        let cfg = PhaseBConfigId::from_code(&cfg_code).map_err(jssp_err_to_py)?;
        let context = build_seed_context(
            rep_idx,
            seed_override,
            instance_id.clone(),
            &pairing_mode,
            pairing_group,
            cfg_label,
        )?;
        batch_tasks.push(PhaseBBatchTask {
            config_id: cfg,
            seed_context: context,
            evaluation_budget: budget,
        });
    }

    let raw_results = py
        .detach(|| {
            core_run_phase_b_batch(&instance, init, init_pop_size, &batch_tasks, num_threads)
        })
        .map_err(jssp_err_to_py)?;

    Ok(raw_results
        .into_iter()
        .map(phase_b_result_to_tuple)
        .collect())
}
