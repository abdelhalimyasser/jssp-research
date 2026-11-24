//! Phase B local search & metaheuristic orchestration (`B01`–`B24`, `B29`–`B32`).

pub mod ils;
pub mod ils_mh;
pub mod metropolis_hastings;
pub mod simulated_annealing;
pub mod tabu_search;

use std::time::Instant;

use crate::decoder::SerialDecoderWorkspace;
use crate::error::JsspError;
use crate::feasibility::validate_schedule;
use crate::initialization::{InitializerId, generate_initial_population};
use crate::local_search::descent::{
    PhaseBRunResult, SearchCounters, SearchTerminationReason, compute_stepwise_anytime_auc,
    run_first_improvement, run_random_improvement, run_steepest_descent,
};
use crate::local_search::evaluation::FastGraphEvaluator;
use crate::model::{JsspInstance, Schedule};
use crate::neighborhoods::NeighborhoodId;
use crate::representation::ObrChromosome;
use crate::rng::{RunRngStreams, SeedDerivationContext};
use crate::telemetry::EvalCounter;

use self::ils::run_ils;
use self::ils_mh::run_ils_mh;
use self::metropolis_hastings::pruned_standalone_mh_note;
use self::simulated_annealing::run_simulated_annealing;
use self::tabu_search::run_tabu_search;

/// Search strategy factor across the 7 active Phase B strategy rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PhaseBStrategy {
    SteepestDescent,
    FirstImprovement,
    RandomImprovement,
    TabuSearch,
    IteratedLocalSearch,
    SimulatedAnnealing,
    MetropolisFilteredIls,
}

impl PhaseBStrategy {
    /// Human-readable strategy name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SteepestDescent => "Steepest Descent",
            Self::FirstImprovement => "First Improvement",
            Self::RandomImprovement => "Random Improvement",
            Self::TabuSearch => "Tabu Search",
            Self::IteratedLocalSearch => "Iterated Local Search (ILS)",
            Self::SimulatedAnnealing => "Simulated Annealing (SA)",
            Self::MetropolisFilteredIls => "Metropolis-Filtered ILS + Steepest Descent",
        }
    }
}

/// Complete 28-configuration active Phase B matrix (`B01`–`B24`, `B29`–`B32`; `B25`–`B28` pruned per `D04-A`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PhaseBConfigId {
    B01,
    B02,
    B03,
    B04,
    B05,
    B06,
    B07,
    B08,
    B09,
    B10,
    B11,
    B12,
    B13,
    B14,
    B15,
    B16,
    B17,
    B18,
    B19,
    B20,
    B21,
    B22,
    B23,
    B24,
    B29,
    B30,
    B31,
    B32,
}

impl PhaseBConfigId {
    /// Ordered array of all 28 active Phase B configurations.
    pub const ALL: [Self; 28] = [
        Self::B01,
        Self::B02,
        Self::B03,
        Self::B04,
        Self::B05,
        Self::B06,
        Self::B07,
        Self::B08,
        Self::B09,
        Self::B10,
        Self::B11,
        Self::B12,
        Self::B13,
        Self::B14,
        Self::B15,
        Self::B16,
        Self::B17,
        Self::B18,
        Self::B19,
        Self::B20,
        Self::B21,
        Self::B22,
        Self::B23,
        Self::B24,
        Self::B29,
        Self::B30,
        Self::B31,
        Self::B32,
    ];

    /// Parses a Phase B configuration code (`"B01"`..=`"B24"`, `"B29"`..=`"B32"`),
    /// rejecting `"B25"`..=`"B28"` explicitly per `D04-A`.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidInitializerConfiguration`] on unknown or pruned IDs.
    pub fn from_code(code: &str) -> Result<Self, JsspError> {
        if let Some(reason) = pruned_standalone_mh_note(code) {
            return Err(JsspError::InvalidInitializerConfiguration {
                detail: format!("{code} is excluded: {reason}"),
            });
        }
        Self::ALL
            .iter()
            .copied()
            .find(|id| id.code() == code)
            .ok_or_else(|| JsspError::InvalidInitializerConfiguration {
                detail: format!("unknown Phase B configuration ID '{code}'"),
            })
    }

    /// Canonical string code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::B01 => "B01",
            Self::B02 => "B02",
            Self::B03 => "B03",
            Self::B04 => "B04",
            Self::B05 => "B05",
            Self::B06 => "B06",
            Self::B07 => "B07",
            Self::B08 => "B08",
            Self::B09 => "B09",
            Self::B10 => "B10",
            Self::B11 => "B11",
            Self::B12 => "B12",
            Self::B13 => "B13",
            Self::B14 => "B14",
            Self::B15 => "B15",
            Self::B16 => "B16",
            Self::B17 => "B17",
            Self::B18 => "B18",
            Self::B19 => "B19",
            Self::B20 => "B20",
            Self::B21 => "B21",
            Self::B22 => "B22",
            Self::B23 => "B23",
            Self::B24 => "B24",
            Self::B29 => "B29",
            Self::B30 => "B30",
            Self::B31 => "B31",
            Self::B32 => "B32",
        }
    }

    /// Search strategy associated with `self`.
    #[must_use]
    pub const fn strategy(self) -> PhaseBStrategy {
        match self {
            Self::B01 | Self::B02 | Self::B03 | Self::B04 => PhaseBStrategy::SteepestDescent,
            Self::B05 | Self::B06 | Self::B07 | Self::B08 => PhaseBStrategy::FirstImprovement,
            Self::B09 | Self::B10 | Self::B11 | Self::B12 => PhaseBStrategy::RandomImprovement,
            Self::B13 | Self::B14 | Self::B15 | Self::B16 => PhaseBStrategy::TabuSearch,
            Self::B17 | Self::B18 | Self::B19 | Self::B20 => PhaseBStrategy::IteratedLocalSearch,
            Self::B21 | Self::B22 | Self::B23 | Self::B24 => PhaseBStrategy::SimulatedAnnealing,
            Self::B29 | Self::B30 | Self::B31 | Self::B32 => PhaseBStrategy::MetropolisFilteredIls,
        }
    }

    /// Neighborhood (`N5`–`N8`) associated with `self`.
    #[must_use]
    pub const fn neighborhood(self) -> NeighborhoodId {
        match self {
            Self::B01 | Self::B05 | Self::B09 | Self::B13 | Self::B17 | Self::B21 | Self::B29 => {
                NeighborhoodId::N5
            }
            Self::B02 | Self::B06 | Self::B10 | Self::B14 | Self::B18 | Self::B22 | Self::B30 => {
                NeighborhoodId::N6
            }
            Self::B03 | Self::B07 | Self::B11 | Self::B15 | Self::B19 | Self::B23 | Self::B31 => {
                NeighborhoodId::N7
            }
            Self::B04 | Self::B08 | Self::B12 | Self::B16 | Self::B20 | Self::B24 | Self::B32 => {
                NeighborhoodId::N8
            }
        }
    }
}

/// Applies one symmetric 3-cycle OBR permutation over three positions carrying pairwise
/// distinct job IDs (`D04-B` Option B, `perturbation` stream). Falls back to distinct-job
/// 2-swap only on degenerate 2-job fixtures.
///
/// # Errors
/// Returns [`JsspError`] if `genes` length < 2 or RNG sampling fails.
pub fn perturb_obr_three_cycle(
    genes: &mut [usize],
    num_jobs: usize,
    rng: &mut crate::rng::NamedRngStream,
) -> Result<(), JsspError> {
    let len = genes.len();
    if num_jobs < 2 || len < 2 {
        return Err(JsspError::InvalidDisjunctiveGraph {
            detail: "OBR perturbation requires >= 2 distinct jobs".to_owned(),
        });
    }
    if num_jobs == 2 {
        let i = rng.gen_range_usize(len)?;
        let mut j = rng.gen_range_usize(len)?;
        while genes[j] == genes[i] {
            j = rng.gen_range_usize(len)?;
        }
        genes.swap(i, j);
        return Ok(());
    }

    // Sample 3 distinct positions uniformly until pairwise job IDs are all distinct:
    // preserves exact uniform 1 / (2 * T) distribution over all 2T distinct 3-cycles.
    loop {
        let p0 = rng.gen_range_usize(len)?;
        let mut p1 = rng.gen_range_usize(len)?;
        while p1 == p0 {
            p1 = rng.gen_range_usize(len)?;
        }
        let mut p2 = rng.gen_range_usize(len)?;
        while p2 == p0 || p2 == p1 {
            p2 = rng.gen_range_usize(len)?;
        }
        let j0 = genes[p0];
        let j1 = genes[p1];
        let j2 = genes[p2];
        if j0 != j1 && j1 != j2 && j0 != j2 {
            let mut idx = [p0, p1, p2];
            idx.sort_unstable();
            let (a, b, c) = (idx[0], idx[1], idx[2]);
            let (va, vb, vc) = (genes[a], genes[b], genes[c]);
            if (rng.next_u64() & 1) == 0 {
                genes[a] = vc;
                genes[b] = va;
                genes[c] = vb;
            } else {
                genes[a] = vb;
                genes[b] = vc;
                genes[c] = va;
            }
            return Ok(());
        }
    }
}

/// Computes Phase B default evaluation budget (`100 * L` screening or `500 * L` confirmatory).
#[must_use]
pub const fn default_phase_b_budget(
    num_jobs: usize,
    num_machines: usize,
    multiplier: usize,
) -> usize {
    num_jobs
        .saturating_mul(num_machines)
        .saturating_mul(multiplier)
}

/// Generates the matched initial starting schedule using frozen Phase A winner `initializer`
/// (evaluating its `pop_size` chromosomes and picking the best makespan tie-broken by lowest index).
///
/// # Errors
/// Returns [`JsspError`] if generation, decoding, or M2 validation fails.
pub fn generate_matched_phase_b_start(
    instance: &JsspInstance,
    initializer: InitializerId,
    pop_size: usize,
    streams: &mut RunRngStreams,
) -> Result<(ObrChromosome, Schedule), JsspError> {
    let pop = generate_initial_population(initializer, instance, pop_size, streams)?;
    let mut ws = SerialDecoderWorkspace::new(instance);
    let mut best_pair: Option<(ObrChromosome, Schedule)> = None;

    for chr in pop.chromosomes {
        let sched = ws.decode(instance, &chr, true)?;
        validate_schedule(instance, &sched)?;
        let is_better = match &best_pair {
            None => true,
            Some((_, bs)) => sched.makespan() < bs.makespan(),
        };
        if is_better {
            best_pair = Some((chr, sched));
        }
    }

    best_pair.ok_or_else(|| JsspError::InvalidInitializerConfiguration {
        detail: "matched Phase B start requires pop_size >= 1".to_owned(),
    })
}

/// Executes one Phase B configuration from a pre-built matched `start_schedule` within
/// exact `evaluation_budget` (charging search evaluations to [`EvalCounter`]).
///
/// # Errors
/// Returns [`JsspError`] if DAG validation, search execution, or final verification fails.
pub fn run_phase_b_from_schedule(
    instance: &JsspInstance,
    config_id: PhaseBConfigId,
    start_schedule: &Schedule,
    evaluation_budget: usize,
    streams: &mut RunRngStreams,
) -> Result<PhaseBRunResult, JsspError> {
    let run_start = Instant::now();
    let neighborhood = config_id.neighborhood();
    let strategy = config_id.strategy();

    let mut evaluator = FastGraphEvaluator::new(instance)?;
    evaluator.load_from_schedule(start_schedule)?;
    let mut decoder_ws = SerialDecoderWorkspace::new(instance);
    let mut counter = EvalCounter::new(evaluation_budget)?;
    let mut counters = SearchCounters::default();

    let initial_makespan = start_schedule.makespan();
    let mut best_makespan = initial_makespan;
    let mut best_found_eval = 0usize;
    let mut best_seqs = evaluator.snapshot_machine_seqs();
    let mut best_milestones = vec![(0usize, initial_makespan)];

    let (final_cmax, reason) = match strategy {
        PhaseBStrategy::SteepestDescent => run_steepest_descent(
            &mut evaluator,
            neighborhood,
            &mut counter,
            &mut counters,
            &mut best_makespan,
            &mut best_found_eval,
            &mut best_seqs,
            &mut best_milestones,
        )?,
        PhaseBStrategy::FirstImprovement => run_first_improvement(
            &mut evaluator,
            neighborhood,
            &mut counter,
            &mut counters,
            &mut best_makespan,
            &mut best_found_eval,
            &mut best_seqs,
            &mut best_milestones,
        )?,
        PhaseBStrategy::RandomImprovement => run_random_improvement(
            &mut evaluator,
            neighborhood,
            &mut counter,
            &mut counters,
            &mut streams.local_search_sample,
            &mut best_makespan,
            &mut best_found_eval,
            &mut best_seqs,
            &mut best_milestones,
        )?,
        PhaseBStrategy::TabuSearch => run_tabu_search(
            instance,
            &mut evaluator,
            &mut decoder_ws,
            neighborhood,
            &mut counter,
            &mut counters,
            streams,
            &mut best_makespan,
            &mut best_found_eval,
            &mut best_seqs,
            &mut best_milestones,
        )?,
        PhaseBStrategy::IteratedLocalSearch => run_ils(
            instance,
            &mut evaluator,
            &mut decoder_ws,
            neighborhood,
            &mut counter,
            &mut counters,
            streams,
            &mut best_makespan,
            &mut best_found_eval,
            &mut best_seqs,
            &mut best_milestones,
        )?,
        PhaseBStrategy::SimulatedAnnealing => run_simulated_annealing(
            &mut evaluator,
            neighborhood,
            &mut counter,
            &mut counters,
            streams,
            &mut best_makespan,
            &mut best_found_eval,
            &mut best_seqs,
            &mut best_milestones,
        )?,
        PhaseBStrategy::MetropolisFilteredIls => run_ils_mh(
            instance,
            &mut evaluator,
            &mut decoder_ws,
            neighborhood,
            &mut counter,
            &mut counters,
            streams,
            &mut best_makespan,
            &mut best_found_eval,
            &mut best_seqs,
            &mut best_milestones,
        )?,
    };

    // Independently verify best DAG acyclicity + M2 schedule projection feasibility
    evaluator.restore_machine_seqs(&best_seqs);
    let verified_state =
        evaluator
            .analyze_current()?
            .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                detail: "Phase B final best machine sequence contains cycle".to_owned(),
            })?;
    if verified_state.makespan != best_makespan {
        return Err(JsspError::MakespanMismatch {
            reported: best_makespan,
            recomputed: verified_state.makespan,
        });
    }
    let (_, best_sched, best_proj_left_shifts) =
        evaluator.project_and_validate(instance, &mut decoder_ws)?;
    counters.adapter_left_shift_count += best_proj_left_shifts;
    let feasible_final = best_sched.makespan() <= best_makespan;

    let num_m = instance.num_machines();
    let best_machine_sequences: Vec<Vec<(usize, usize)>> = best_seqs
        .iter()
        .map(|seq| {
            seq.iter()
                .map(|&flat| ((flat as usize) / num_m, (flat as usize) % num_m))
                .collect()
        })
        .collect();

    let anytime_cmax_auc = compute_stepwise_anytime_auc(&best_milestones, evaluation_budget);
    let reached_local_optimum = matches!(reason, SearchTerminationReason::LocalOptimum);
    let total_elapsed_ns = u64::try_from(run_start.elapsed().as_nanos()).unwrap_or(u64::MAX);

    Ok(PhaseBRunResult {
        config_id: config_id.code(),
        neighborhood_id: neighborhood.code(),
        evaluation_budget,
        evals_used: counter.total(),
        initial_makespan,
        final_makespan: final_cmax,
        best_makespan,
        best_found_eval,
        feasible_final,
        reached_local_optimum,
        terminated_reason: reason,
        counters,
        anytime_cmax_auc,
        best_milestones,
        best_machine_sequences,
        total_elapsed_ns,
    })
}

/// Specification for one task inside [`run_phase_b_batch`].
#[derive(Debug, Clone)]
pub struct PhaseBBatchTask {
    pub config_id: PhaseBConfigId,
    pub seed_context: SeedDerivationContext,
    pub evaluation_budget: usize,
}

/// Runs a batch of Phase B tasks on `instance` in parallel across OS threads while
/// guaranteeing bit-identical matched starting schedules per `replicate_index` from `initializer`.
///
/// # Errors
/// Returns the first encountered [`JsspError`].
pub fn run_phase_b_batch(
    instance: &JsspInstance,
    initializer: InitializerId,
    init_pop_size: usize,
    tasks: &[PhaseBBatchTask],
    num_threads: usize,
) -> Result<Vec<PhaseBRunResult>, JsspError> {
    if tasks.is_empty() {
        return Ok(Vec::new());
    }
    let resolved_threads = if num_threads == 0 {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(tasks.len())
    } else {
        num_threads.min(tasks.len())
    };

    if resolved_threads <= 1 {
        return tasks
            .iter()
            .map(|t| {
                let mut streams = t.seed_context.all_streams();
                let (_, start_sched) = generate_matched_phase_b_start(
                    instance,
                    initializer,
                    init_pop_size,
                    &mut streams,
                )?;
                run_phase_b_from_schedule(
                    instance,
                    t.config_id,
                    &start_sched,
                    t.evaluation_budget,
                    &mut streams,
                )
            })
            .collect();
    }

    let chunk_size = tasks.len().div_ceil(resolved_threads);
    let chunk_results: Vec<Result<Vec<PhaseBRunResult>, JsspError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = tasks
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(move || {
                    let mut local = Vec::with_capacity(chunk.len());
                    for t in chunk {
                        let mut streams = t.seed_context.all_streams();
                        let (_, start_sched) = generate_matched_phase_b_start(
                            instance,
                            initializer,
                            init_pop_size,
                            &mut streams,
                        )?;
                        local.push(run_phase_b_from_schedule(
                            instance,
                            t.config_id,
                            &start_sched,
                            t.evaluation_budget,
                            &mut streams,
                        )?);
                    }
                    Ok(local)
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|h| match h.join() {
                Ok(res) => res,
                Err(_) => Err(JsspError::InvalidDisjunctiveGraph {
                    detail: "worker thread panicked during Phase B batch".to_owned(),
                }),
            })
            .collect()
    });

    let mut ordered = Vec::with_capacity(tasks.len());
    for res in chunk_results {
        ordered.extend(res?);
    }
    Ok(ordered)
}
