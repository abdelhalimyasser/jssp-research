//! Classical Genetic Algorithm (`C01`) baseline runner and thread-parallel batch execution.

use std::time::Instant;

use crate::decoder::SerialDecoderWorkspace;
use crate::error::JsspError;
use crate::feasibility::validate_schedule;
use crate::genetic::crossover::pox_crossover;
use crate::genetic::elitism::ranked_population_indices;
use crate::genetic::mutation::mutate_obr;
use crate::genetic::selection::tournament_select;
use crate::initialization::{InitializerId, generate_initial_population};
use crate::model::{JsspInstance, Schedule};
use crate::representation::ObrChromosome;
use crate::rng::{RunRngStreams, SeedDerivationContext};
use crate::telemetry::{
    EvalCounter, GenerationSnapshot, PhaseARunResult, PopulationDiagnostics,
    compute_anytime_cmax_auc, compute_obr_hamming_diversity, count_unique_chromosomes,
    schedule_machine_sequence_hash, summarize_population_diagnostics,
};

/// Frozen Classical GA baseline (`C01`) parameters (`D07-A`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassicalGaConfig {
    pub population_size: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub elitism_count: usize,
    pub evaluation_budget: usize,
}

impl ClassicalGaConfig {
    /// Constructs the frozen `C01` configuration (`N=64, k=2, p_c=0.85, p_m=0.15, E=2`)
    /// with the specified `evaluation_budget`.
    #[must_use]
    pub const fn frozen_c01(evaluation_budget: usize) -> Self {
        Self {
            population_size: 64,
            tournament_size: 2,
            crossover_rate: 0.85,
            mutation_rate: 0.15,
            elitism_count: 2,
            evaluation_budget,
        }
    }

    /// Constructs the frozen `C01` config with the size-scaled Phase A evaluation budget
    /// `clamp(64 * ceil(2.5 * J * M), 6400, 64000)`.
    #[must_use]
    pub fn for_phase_a_instance(instance: &JsspInstance) -> Self {
        Self::frozen_c01(default_phase_a_ga_budget(
            instance.num_jobs(),
            instance.num_machines(),
        ))
    }

    /// Validates parameter ranges against Classical GA invariants.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidGeneticConfiguration`] if any constraint fails.
    pub fn validate(&self) -> Result<(), JsspError> {
        if self.population_size < 2 {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: format!("population_size must be >= 2, got {}", self.population_size),
            });
        }
        if self.tournament_size == 0 {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: "tournament_size must be >= 1".to_owned(),
            });
        }
        if !self.crossover_rate.is_finite() || !(0.0..=1.0).contains(&self.crossover_rate) {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: format!(
                    "crossover_rate must lie in [0, 1], got {}",
                    self.crossover_rate
                ),
            });
        }
        if !self.mutation_rate.is_finite() || !(0.0..=1.0).contains(&self.mutation_rate) {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: format!(
                    "mutation_rate must lie in [0, 1], got {}",
                    self.mutation_rate
                ),
            });
        }
        if self.elitism_count >= self.population_size {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: format!(
                    "elitism_count {} must be strictly less than population_size {}",
                    self.elitism_count, self.population_size
                ),
            });
        }
        if self.evaluation_budget < self.population_size {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: format!(
                    "evaluation_budget {} must be >= population_size {}",
                    self.evaluation_budget, self.population_size
                ),
            });
        }
        Ok(())
    }
}

/// Computes the frozen size-scaled Phase A Classical GA evaluation budget:
/// `clamp(64 * ceil(2.5 * J * M), 6400, 64000)`.
#[must_use]
pub fn default_phase_a_ga_budget(num_jobs: usize, num_machines: usize) -> usize {
    let l = num_jobs.saturating_mul(num_machines);
    let multiplier = ((l as f64) * 2.5).ceil() as usize;
    multiplier.saturating_mul(64).clamp(6_400, 64_000)
}

/// Standalone Layer A evaluation of an initial population (64 decodes + M2 feasibility verification).
///
/// # Errors
/// Returns [`JsspError`] if generation, decoding, or feasibility validation fails.
pub fn evaluate_initializer_only(
    instance: &JsspInstance,
    initializer: InitializerId,
    pop_size: usize,
    streams: &mut RunRngStreams,
) -> Result<(PopulationDiagnostics, Vec<ObrChromosome>, Vec<Schedule>), JsspError> {
    let generated = generate_initial_population(initializer, instance, pop_size, streams)?;
    let mut workspace = SerialDecoderWorkspace::new(instance);
    let decode_start = Instant::now();
    let mut schedules = Vec::with_capacity(generated.chromosomes.len());
    let mut feasible_count = 0usize;

    for chr in &generated.chromosomes {
        let sched = workspace.decode(instance, chr, true)?;
        validate_schedule(instance, &sched)?;
        feasible_count += 1;
        schedules.push(sched);
    }

    let decode_elapsed_ns = u64::try_from(decode_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let diagnostics = summarize_population_diagnostics(
        instance,
        &generated.chromosomes,
        &schedules,
        feasible_count,
        generated.first_dim_coords.as_deref(),
        generated.generation_elapsed_ns,
        decode_elapsed_ns,
    )?;

    Ok((diagnostics, generated.chromosomes, schedules))
}

/// Executes one complete Phase A `(initializer -> Classical GA C01)` experiment run.
///
/// # Errors
/// Returns [`JsspError`] if configuration validation, initialization, decoding,
/// genetic operations, or feasibility checks fail.
pub fn run_phase_a_ga(
    instance: &JsspInstance,
    initializer: InitializerId,
    config: &ClassicalGaConfig,
    streams: &mut RunRngStreams,
) -> Result<PhaseARunResult, JsspError> {
    config.validate()?;
    let run_start = Instant::now();
    let pop_size = config.population_size;
    let num_jobs = instance.num_jobs();
    let mut counter = EvalCounter::new(config.evaluation_budget)?;
    let mut workspace = SerialDecoderWorkspace::new(instance);

    // 1. Initial population generation and M2-validated Layer A decoding.
    let generated = generate_initial_population(initializer, instance, pop_size, streams)?;
    let decode_start = Instant::now();
    let mut chromosomes = generated.chromosomes;
    let mut initial_schedules = Vec::with_capacity(pop_size);
    let mut makespans = Vec::with_capacity(pop_size);
    let mut feasible_init = 0usize;

    let mut best_makespan = u64::MAX;
    let mut best_found_eval = 0usize;
    let mut best_found_generation = 0usize;
    let mut best_chromosome = chromosomes[0].clone();

    for chr in &chromosomes {
        if !counter.try_consume_init() {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: "evaluation budget exhausted during initial population".to_owned(),
            });
        }
        let sched = workspace.decode(instance, chr, true)?;
        validate_schedule(instance, &sched)?;
        feasible_init += 1;
        let cmax = sched.makespan();
        makespans.push(cmax);
        if cmax < best_makespan {
            best_makespan = cmax;
            best_found_eval = counter.total();
            best_chromosome = chr.clone();
        }
        initial_schedules.push(sched);
    }

    let init_decode_ns = u64::try_from(decode_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let initial_diag = summarize_population_diagnostics(
        instance,
        &chromosomes,
        &initial_schedules,
        feasible_init,
        generated.first_dim_coords.as_deref(),
        generated.generation_elapsed_ns,
        init_decode_ns,
    )?;

    let mut trajectory = Vec::new();
    trajectory.push(GenerationSnapshot {
        generation: 0,
        evals_used: counter.total(),
        best_makespan,
        mean_makespan: initial_diag.mean_makespan,
        obr_hamming_diversity: initial_diag.obr_hamming_diversity,
        unique_chromosomes: initial_diag.unique_chromosomes,
    });

    // 2. Generational Classical GA evolution loop (`C01`).
    let mut generation = 0usize;
    let mut crossover_count = 0usize;
    let mut mutation_count = 0usize;

    while counter.remaining() > 0 {
        generation += 1;
        let ranked = ranked_population_indices(&makespans);
        let nominal_new = pop_size - config.elitism_count;
        let num_new = nominal_new.min(counter.remaining());
        let carry_count = pop_size - num_new;

        let mut next_chromosomes = Vec::with_capacity(pop_size);
        let mut next_makespans = Vec::with_capacity(pop_size);

        for &idx in ranked.iter().take(carry_count) {
            next_chromosomes.push(chromosomes[idx].clone());
            next_makespans.push(makespans[idx]);
        }

        let mut produced = 0usize;
        while produced < num_new {
            let p1_idx =
                tournament_select(&makespans, config.tournament_size, &mut streams.selection)?;
            let p2_idx =
                tournament_select(&makespans, config.tournament_size, &mut streams.selection)?;
            let (mut child1, mut child2, did_cx) = pox_crossover(
                &chromosomes[p1_idx],
                &chromosomes[p2_idx],
                num_jobs,
                config.crossover_rate,
                &mut streams.crossover,
            )?;
            if did_cx {
                crossover_count += 1;
            }

            if mutate_obr(&mut child1, config.mutation_rate, &mut streams.mutation)? {
                mutation_count += 1;
            }
            if counter.try_consume_search() {
                let cmax1 = workspace.decode_makespan_only(instance, &child1)?;
                if cmax1 < best_makespan {
                    best_makespan = cmax1;
                    best_found_eval = counter.total();
                    best_found_generation = generation;
                    best_chromosome = child1.clone();
                }
                next_chromosomes.push(child1);
                next_makespans.push(cmax1);
                produced += 1;
            }

            if produced < num_new {
                if mutate_obr(&mut child2, config.mutation_rate, &mut streams.mutation)? {
                    mutation_count += 1;
                }
                if counter.try_consume_search() {
                    let cmax2 = workspace.decode_makespan_only(instance, &child2)?;
                    if cmax2 < best_makespan {
                        best_makespan = cmax2;
                        best_found_eval = counter.total();
                        best_found_generation = generation;
                        best_chromosome = child2.clone();
                    }
                    next_chromosomes.push(child2);
                    next_makespans.push(cmax2);
                    produced += 1;
                }
            }
        }

        chromosomes = next_chromosomes;
        makespans = next_makespans;

        let mean_cmax = makespans.iter().map(|&v| v as f64).sum::<f64>() / (makespans.len() as f64);
        let div = compute_obr_hamming_diversity(&chromosomes, num_jobs);
        let uniq_obr = count_unique_chromosomes(&chromosomes);
        trajectory.push(GenerationSnapshot {
            generation,
            evals_used: counter.total(),
            best_makespan,
            mean_makespan: mean_cmax,
            obr_hamming_diversity: div,
            unique_chromosomes: uniq_obr,
        });
    }

    // 3. Authoritative final population + best-schedule verification under full M2 validation.
    let final_decode_start = Instant::now();
    let mut final_schedules = Vec::with_capacity(pop_size);
    let mut feasible_final = 0usize;
    for chr in &chromosomes {
        let sched = workspace.decode(instance, chr, true)?;
        validate_schedule(instance, &sched)?;
        feasible_final += 1;
        final_schedules.push(sched);
    }
    let final_decode_ns =
        u64::try_from(final_decode_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let final_pop_diag = summarize_population_diagnostics(
        instance,
        &chromosomes,
        &final_schedules,
        feasible_final,
        None,
        0,
        final_decode_ns,
    )?;

    let best_schedule = workspace.decode(instance, &best_chromosome, true)?;
    validate_schedule(instance, &best_schedule)?;
    if best_schedule.makespan() != best_makespan {
        return Err(JsspError::MakespanMismatch {
            reported: best_makespan,
            recomputed: best_schedule.makespan(),
        });
    }
    let best_schedule_hash = schedule_machine_sequence_hash(&best_schedule);
    let anytime_cmax_auc = compute_anytime_cmax_auc(&trajectory, config.evaluation_budget);
    let total_elapsed_ns = u64::try_from(run_start.elapsed().as_nanos()).unwrap_or(u64::MAX);

    Ok(PhaseARunResult {
        initializer_id: initializer.code(),
        pop_size,
        evaluation_budget: config.evaluation_budget,
        evals_used: counter.total(),
        generations_executed: generation,
        initial: initial_diag,
        final_pop: final_pop_diag,
        best_makespan,
        best_found_eval,
        best_found_generation,
        best_chromosome: best_chromosome.to_job_indices(),
        best_schedule_hash,
        anytime_cmax_auc,
        crossover_count,
        mutation_count,
        total_elapsed_ns,
        trajectory,
    })
}

/// Specification for one task inside [`run_phase_a_batch`].
#[derive(Debug, Clone)]
pub struct PhaseABatchTask {
    pub initializer: InitializerId,
    pub seed_context: SeedDerivationContext,
    pub config: ClassicalGaConfig,
}

/// Executes a batch of Phase A `(initializer, replicate)` runs on one instance across
/// OS worker threads using `std::thread::scope`, returning results in input task order.
///
/// # Errors
/// Returns the first encountered [`JsspError`] if any task fails.
pub fn run_phase_a_batch(
    instance: &JsspInstance,
    tasks: &[PhaseABatchTask],
    num_threads: usize,
) -> Result<Vec<PhaseARunResult>, JsspError> {
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
            .map(|task| {
                let mut streams = task.seed_context.all_streams();
                run_phase_a_ga(instance, task.initializer, &task.config, &mut streams)
            })
            .collect();
    }

    let chunk_size = tasks.len().div_ceil(resolved_threads);
    let chunk_results: Vec<Result<Vec<PhaseARunResult>, JsspError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = tasks
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(move || {
                    let mut local = Vec::with_capacity(chunk.len());
                    for task in chunk {
                        let mut streams = task.seed_context.all_streams();
                        local.push(run_phase_a_ga(
                            instance,
                            task.initializer,
                            &task.config,
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
                Err(_) => Err(JsspError::InvalidGeneticConfiguration {
                    detail: "worker thread panicked during Phase A batch execution".to_owned(),
                }),
            })
            .collect()
    });

    let mut ordered = Vec::with_capacity(tasks.len());
    for chunk_res in chunk_results {
        ordered.extend(chunk_res?);
    }
    Ok(ordered)
}
