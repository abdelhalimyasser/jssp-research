//! Phase C Memetic & Dual-View Genetic Algorithms (`C01`–`C11`, `D06`, `D07`).

pub mod baldwinian;
pub mod dual_view;
pub mod lamarckian;
pub mod partial_lamarckian;

use std::time::Instant;

use crate::decoder::SerialDecoderWorkspace;
use crate::error::JsspError;
use crate::feasibility::validate_schedule;
use crate::genetic::{
    ClassicalGaConfig, PoxWorkspace, default_phase_a_ga_budget, mutate_obr,
    ranked_population_indices, run_phase_a_ga, tournament_select,
};
use crate::initialization::{InitializerId, generate_initial_population};
use crate::local_search::descent::{
    SearchCounters, compute_stepwise_anytime_auc, run_first_improvement, run_random_improvement,
    run_steepest_descent,
};
use crate::local_search::evaluation::FastGraphEvaluator;
use crate::metaheuristics::PhaseBStrategy;
use crate::metaheuristics::ils::run_ils;
use crate::metaheuristics::ils_mh::run_ils_mh;
use crate::metaheuristics::simulated_annealing::run_simulated_annealing;
use crate::metaheuristics::tabu_search::run_tabu_search;
use crate::model::{JobId, JsspInstance};
use crate::neighborhoods::NeighborhoodId;
use crate::representation::ObrChromosome;
use crate::rng::{RunRngStreams, SeedDerivationContext};
use crate::telemetry::{EvalCounter, compute_obr_hamming_diversity};

use self::baldwinian::apply_baldwinian_preservation;
use self::dual_view::{run_dual_view_model_a, run_unpaired_dual_pool_control};
use self::lamarckian::{MemeticEvalOutcome, apply_lamarckian_writeback};
use self::partial_lamarckian::apply_partial_lamarckian_even_ranks;

/// Complete 11-configuration Phase C matrix (`C01`–`C11`, `D07`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PhaseCConfigId {
    C01,
    C02,
    C03,
    C04,
    C05,
    C06,
    C07,
    C08,
    C09,
    C10,
    C11,
}

impl PhaseCConfigId {
    /// Ordered array of all 11 approved Phase C configurations.
    pub const ALL: [Self; 11] = [
        Self::C01,
        Self::C02,
        Self::C03,
        Self::C04,
        Self::C05,
        Self::C06,
        Self::C07,
        Self::C08,
        Self::C09,
        Self::C10,
        Self::C11,
    ];

    /// Parses `"C01"`..=`"C11"`.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidGeneticConfiguration`] on unknown IDs.
    pub fn from_code(code: &str) -> Result<Self, JsspError> {
        Self::ALL
            .iter()
            .copied()
            .find(|c| c.code() == code)
            .ok_or_else(|| JsspError::InvalidGeneticConfiguration {
                detail: format!("unknown Phase C configuration ID '{code}', expected C01..=C11"),
            })
    }

    /// Canonical string code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::C01 => "C01",
            Self::C02 => "C02",
            Self::C03 => "C03",
            Self::C04 => "C04",
            Self::C05 => "C05",
            Self::C06 => "C06",
            Self::C07 => "C07",
            Self::C08 => "C08",
            Self::C09 => "C09",
            Self::C10 => "C10",
            Self::C11 => "C11",
        }
    }

    /// Human-readable descriptive name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::C01 => "Classical GA Baseline",
            Self::C02 => "Lamarckian GA (LGA)",
            Self::C03 => "Baldwinian GA (BGA)",
            Self::C04 => "Partial Lamarckian GA (PLGA 50%)",
            Self::C05 => "Dual-View Fixed Gate (p=0.50)",
            Self::C06 => "Dual-View Adaptive Gate (w_perf=0.70, w_div=0.30)",
            Self::C07 => "Dual-View Fixed Gate (p=0.25)",
            Self::C08 => "Dual-View Fixed Gate (p=0.75)",
            Self::C09 => "Dual-View Adaptive Gate Ablation (w_div=0.00)",
            Self::C10 => "Unpaired Dual-Population Control (32 LGA + 32 BGA)",
            Self::C11 => "Dual-View Adaptive + Cross-View Migration (Exploratory)",
        }
    }

    /// True if configuration executes embedded local search (`C02`–`C11`).
    #[must_use]
    pub const fn is_memetic(self) -> bool {
        !matches!(self, Self::C01)
    }

    /// True if configuration uses paired Model A Dual-View representation (`C05`–`C09`, `C11`).
    #[must_use]
    pub const fn is_dual_view_paired(self) -> bool {
        matches!(
            self,
            Self::C05 | Self::C06 | Self::C07 | Self::C08 | Self::C09 | Self::C11
        )
    }
}

/// Embedded local-search operator transferred from Phase B into `C02`–`C11`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemeticLsConfig {
    pub neighborhood: NeighborhoodId,
    pub strategy: PhaseBStrategy,
    pub max_ls_evals: usize,
}

/// Complete resolved Phase C execution configuration (`C01`–`C11`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseCGaConfig {
    pub config_id: PhaseCConfigId,
    pub initializer: InitializerId,
    pub population_size: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub elitism_count: usize,
    pub evaluation_budget: usize,
    pub local_search: MemeticLsConfig,
}

impl PhaseCGaConfig {
    /// Builds a frozen Phase C configuration using equalized GA parameters
    /// (`N=64, k=2, p_c=0.85, p_m=0.15, E=2`) and local search budget `b_LS = 50`.
    #[must_use]
    pub const fn standard(
        config_id: PhaseCConfigId,
        initializer: InitializerId,
        evaluation_budget: usize,
        neighborhood: NeighborhoodId,
        strategy: PhaseBStrategy,
        max_ls_evals: usize,
    ) -> Self {
        Self {
            config_id,
            initializer,
            population_size: 64,
            tournament_size: 2,
            crossover_rate: 0.85,
            mutation_rate: 0.15,
            elitism_count: 2,
            evaluation_budget,
            local_search: MemeticLsConfig {
                neighborhood,
                strategy,
                max_ls_evals,
            },
        }
    }

    /// Validates configuration invariants against Phase C capacity and budget rules.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidGeneticConfiguration`] on invalid parameters.
    pub fn validate(&self) -> Result<(), JsspError> {
        ClassicalGaConfig {
            population_size: self.population_size,
            tournament_size: self.tournament_size,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            elitism_count: self.elitism_count,
            evaluation_budget: self.evaluation_budget,
        }
        .validate()?;
        if !self.population_size.is_multiple_of(2) || self.population_size < 4 {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: format!(
                    "Phase C population_size must be an even integer >= 4 for dual-view pairing, got {}",
                    self.population_size
                ),
            });
        }
        if self.config_id.is_memetic() && self.local_search.max_ls_evals == 0 {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: format!(
                    "{} requires positive local_search.max_ls_evals",
                    self.config_id.code()
                ),
            });
        }
        Ok(())
    }
}

/// Complete telemetry returned by one Phase C run (`C01`–`C11`).
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseCRunResult {
    pub config_id: &'static str,
    pub initializer_id: &'static str,
    pub ls_neighborhood: Option<&'static str>,
    pub ls_strategy: Option<&'static str>,
    pub genotype_capacity: usize,
    pub evaluation_budget: usize,
    pub evals_used: usize,
    pub init_evals: usize,
    pub search_evals: usize,
    pub decode_evals: usize,
    pub ls_evals: usize,
    pub generations_executed: usize,
    pub best_makespan: u64,
    pub best_found_eval: usize,
    pub best_found_generation: usize,
    pub feasible_final: bool,
    pub anytime_cmax_auc: f64,
    pub final_obr_diversity: f64,
    pub final_lam_diversity: Option<f64>,
    pub final_bald_diversity: Option<f64>,
    pub final_gate_p: Option<f64>,
    pub lam_best_makespan: Option<u64>,
    pub bald_best_makespan: Option<u64>,
    pub gate_p_trajectory: Vec<(usize, f64)>,
    pub best_milestones: Vec<(usize, u64)>,
    pub crossover_count: usize,
    pub mutation_count: usize,
    pub writeback_count: usize,
    pub migrations_applied: usize,
    pub best_chromosome: Vec<usize>,
    pub best_machine_sequences: Vec<Vec<(usize, usize)>>,
    pub total_elapsed_ns: u64,
}

/// Computes Phase C default evaluation budget matching `default_phase_a_ga_budget`:
/// `clamp(64 * ceil(2.5 * J * M), 6400, 64000)`.
#[must_use]
pub fn default_phase_c_budget(num_jobs: usize, num_machines: usize) -> usize {
    default_phase_a_ga_budget(num_jobs, num_machines)
}

pub(crate) type VerifiedBestBundle = (u64, Vec<usize>, Vec<Vec<(usize, usize)>>);
pub(crate) type DecodedLsBundle = (MemeticEvalOutcome, Vec<Vec<u32>>);

pub(crate) fn verify_final_best(
    instance: &JsspInstance,
    chr: &ObrChromosome,
    decoder_ws: &mut SerialDecoderWorkspace,
) -> Result<VerifiedBestBundle, JsspError> {
    let sched = decoder_ws.decode(instance, chr, true)?;
    validate_schedule(instance, &sched)?;
    let seqs: Vec<Vec<(usize, usize)>> = sched
        .machine_sequences()
        .into_iter()
        .map(|ms| {
            ms.operations
                .into_iter()
                .map(|op| (op.job.0, op.op_index.0))
                .collect()
        })
        .collect();
    Ok((sched.makespan(), chr.to_job_indices(), seqs))
}

/// Decodes `chr` (1 eval charged to `counter`) and applies bounded local search up to
/// `min(ls_cfg.max_ls_evals, counter.remaining())`, charging exact LS evaluations cleanly back
/// into the single shared `counter`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_and_local_search(
    instance: &JsspInstance,
    chr: &ObrChromosome,
    ls_cfg: MemeticLsConfig,
    is_init: bool,
    counter: &mut EvalCounter,
    decoder_ws: &mut SerialDecoderWorkspace,
    ls_decoder_ws: &mut SerialDecoderWorkspace,
    graph_eval: &mut FastGraphEvaluator,
    streams: &mut RunRngStreams,
) -> Result<Option<DecodedLsBundle>, JsspError> {
    let ok = if is_init {
        counter.try_consume_init()
    } else {
        counter.try_consume_search()
    };
    if !ok {
        return Ok(None);
    }

    let raw_sched = decoder_ws.decode(instance, chr, true)?;
    validate_schedule(instance, &raw_sched)?;
    let raw_cmax = raw_sched.makespan();
    graph_eval.load_from_schedule(&raw_sched)?;

    let allowed_ls = ls_cfg.max_ls_evals.min(counter.remaining());
    if allowed_ls == 0 {
        let seqs = graph_eval.snapshot_machine_seqs();
        return Ok(Some((
            MemeticEvalOutcome {
                original_genotype: chr.clone(),
                improved_genotype: chr.clone(),
                effective_genotype: chr.clone(),
                raw_makespan: raw_cmax,
                post_ls_makespan: raw_cmax,
                ls_evals_consumed: 0,
                best_schedule: raw_sched,
            },
            seqs,
        )));
    }

    let mut ls_counter = EvalCounter::new(allowed_ls)?;
    let mut dummy_counters = SearchCounters::default();
    let mut ls_best_cmax = raw_cmax;
    let mut dummy_best_eval = 0usize;
    let mut ls_best_seqs = graph_eval.snapshot_machine_seqs();
    let mut dummy_milestones = vec![(0usize, raw_cmax)];

    match ls_cfg.strategy {
        PhaseBStrategy::SteepestDescent => {
            let _ = run_steepest_descent(
                graph_eval,
                ls_cfg.neighborhood,
                &mut ls_counter,
                &mut dummy_counters,
                &mut ls_best_cmax,
                &mut dummy_best_eval,
                &mut ls_best_seqs,
                &mut dummy_milestones,
            )?;
        }
        PhaseBStrategy::FirstImprovement => {
            let _ = run_first_improvement(
                graph_eval,
                ls_cfg.neighborhood,
                &mut ls_counter,
                &mut dummy_counters,
                &mut ls_best_cmax,
                &mut dummy_best_eval,
                &mut ls_best_seqs,
                &mut dummy_milestones,
            )?;
        }
        PhaseBStrategy::RandomImprovement => {
            let _ = run_random_improvement(
                graph_eval,
                ls_cfg.neighborhood,
                &mut ls_counter,
                &mut dummy_counters,
                &mut streams.local_search_sample,
                &mut ls_best_cmax,
                &mut dummy_best_eval,
                &mut ls_best_seqs,
                &mut dummy_milestones,
            )?;
        }
        PhaseBStrategy::TabuSearch => {
            let _ = run_tabu_search(
                instance,
                graph_eval,
                ls_decoder_ws,
                ls_cfg.neighborhood,
                &mut ls_counter,
                &mut dummy_counters,
                streams,
                &mut ls_best_cmax,
                &mut dummy_best_eval,
                &mut ls_best_seqs,
                &mut dummy_milestones,
            )?;
        }
        PhaseBStrategy::IteratedLocalSearch => {
            let _ = run_ils(
                instance,
                graph_eval,
                ls_decoder_ws,
                ls_cfg.neighborhood,
                &mut ls_counter,
                &mut dummy_counters,
                streams,
                &mut ls_best_cmax,
                &mut dummy_best_eval,
                &mut ls_best_seqs,
                &mut dummy_milestones,
            )?;
        }
        PhaseBStrategy::SimulatedAnnealing => {
            let _ = run_simulated_annealing(
                graph_eval,
                ls_cfg.neighborhood,
                &mut ls_counter,
                &mut dummy_counters,
                streams,
                &mut ls_best_cmax,
                &mut dummy_best_eval,
                &mut ls_best_seqs,
                &mut dummy_milestones,
            )?;
        }
        PhaseBStrategy::MetropolisFilteredIls => {
            let _ = run_ils_mh(
                instance,
                graph_eval,
                ls_decoder_ws,
                ls_cfg.neighborhood,
                &mut ls_counter,
                &mut dummy_counters,
                streams,
                &mut ls_best_cmax,
                &mut dummy_best_eval,
                &mut ls_best_seqs,
                &mut dummy_milestones,
            )?;
        }
    }

    let ls_used = counter.consume_search_up_to(ls_counter.total());
    if ls_best_cmax < raw_cmax {
        graph_eval.restore_machine_seqs(&ls_best_seqs);
        let (improved_obr, ls_sched, _) =
            graph_eval.project_and_validate(instance, ls_decoder_ws)?;
        let post_cmax = ls_sched.makespan();
        graph_eval.load_from_schedule(&ls_sched)?;
        let final_seqs = graph_eval.snapshot_machine_seqs();
        Ok(Some((
            MemeticEvalOutcome {
                original_genotype: chr.clone(),
                improved_genotype: improved_obr.clone(),
                effective_genotype: improved_obr,
                raw_makespan: raw_cmax,
                post_ls_makespan: post_cmax,
                ls_evals_consumed: ls_used,
                best_schedule: ls_sched,
            },
            final_seqs,
        )))
    } else {
        Ok(Some((
            MemeticEvalOutcome {
                original_genotype: chr.clone(),
                improved_genotype: chr.clone(),
                effective_genotype: chr.clone(),
                raw_makespan: raw_cmax,
                post_ls_makespan: raw_cmax,
                ls_evals_consumed: ls_used,
                best_schedule: raw_sched,
            },
            ls_best_seqs,
        )))
    }
}

/// Runs single-view memetic GA (`C02` LGA, `C03` BGA, `C04` PLGA 50%).
#[allow(clippy::too_many_lines)]
fn run_single_view_memetic(
    instance: &JsspInstance,
    config: &PhaseCGaConfig,
    counter: &mut EvalCounter,
    decoder_ws: &mut SerialDecoderWorkspace,
    ls_decoder_ws: &mut SerialDecoderWorkspace,
    graph_eval: &mut FastGraphEvaluator,
    streams: &mut RunRngStreams,
) -> Result<PhaseCRunResult, JsspError> {
    let run_start = Instant::now();
    let pop_size = config.population_size;
    let elite_cnt = config.elitism_count;
    let num_jobs = instance.num_jobs();
    let chrom_len = instance.num_operations();

    let generated = generate_initial_population(config.initializer, instance, pop_size, streams)?;
    let mut chromosomes: Vec<ObrChromosome> = Vec::with_capacity(pop_size);
    let mut improved_opts: Vec<Option<ObrChromosome>> = Vec::with_capacity(pop_size);
    let mut fitnesses: Vec<u64> = Vec::with_capacity(pop_size);

    let mut decode_evals = 0usize;
    let mut ls_evals = 0usize;
    let mut writeback_count = 0usize;
    let mut crossover_count = 0usize;
    let mut mutation_count = 0usize;

    let mut best_makespan = u64::MAX;
    let mut best_chr = generated.chromosomes[0].clone();
    let mut best_found_eval = 0usize;
    let mut best_found_gen = 0usize;
    let mut milestones: Vec<(usize, u64)> = Vec::new();

    for c in &generated.chromosomes {
        let Some((mut out, _)) = decode_and_local_search(
            instance,
            c,
            config.local_search,
            true,
            counter,
            decoder_ws,
            ls_decoder_ws,
            graph_eval,
            streams,
        )?
        else {
            break;
        };
        decode_evals += 1;
        ls_evals += out.ls_evals_consumed;
        if out.post_ls_makespan < best_makespan {
            best_makespan = out.post_ls_makespan;
            best_chr = out.improved_genotype.clone();
            best_found_eval = counter.total();
        }
        match config.config_id {
            PhaseCConfigId::C02 => {
                apply_lamarckian_writeback(instance, &mut out)?;
                if out.post_ls_makespan < out.raw_makespan {
                    writeback_count += 1;
                }
                chromosomes.push(out.effective_genotype);
                improved_opts.push(None);
            }
            PhaseCConfigId::C03 => {
                apply_baldwinian_preservation(instance, &mut out)?;
                chromosomes.push(out.effective_genotype);
                improved_opts.push(None);
            }
            PhaseCConfigId::C04 => {
                let imp_opt = if out.post_ls_makespan < out.raw_makespan {
                    Some(out.improved_genotype.clone())
                } else {
                    None
                };
                chromosomes.push(out.original_genotype);
                improved_opts.push(imp_opt);
            }
            other => {
                return Err(JsspError::InvalidGeneticConfiguration {
                    detail: format!("{} is not a single-view memetic config", other.code()),
                });
            }
        }
        fitnesses.push(out.post_ls_makespan);
    }

    if matches!(config.config_id, PhaseCConfigId::C04) && !chromosomes.is_empty() {
        writeback_count += apply_partial_lamarckian_even_ranks(
            instance,
            &mut chromosomes,
            &improved_opts,
            &fitnesses,
        )?;
    }
    if best_makespan < u64::MAX {
        milestones.push((best_found_eval, best_makespan));
    }

    let mut pox_ws = PoxWorkspace::new(num_jobs, chrom_len);
    let mut buf1: Vec<JobId> = Vec::with_capacity(chrom_len);
    let mut buf2: Vec<JobId> = Vec::with_capacity(chrom_len);
    let mut generation = 0usize;

    while counter.remaining() > 0 && chromosomes.len() >= 2 {
        let ranked = ranked_population_indices(&fitnesses);
        let mut next_chrs: Vec<ObrChromosome> = Vec::with_capacity(pop_size);
        let mut next_imps: Vec<Option<ObrChromosome>> = Vec::with_capacity(pop_size);
        let mut next_fits: Vec<u64> = Vec::with_capacity(pop_size);

        for &e in ranked.iter().take(elite_cnt) {
            next_chrs.push(chromosomes[e].clone());
            next_imps.push(None);
            next_fits.push(fitnesses[e]);
        }

        let gen_next = generation + 1;
        while next_chrs.len() < pop_size && counter.remaining() > 0 {
            let p1 = tournament_select(&fitnesses, config.tournament_size, &mut streams.selection)?;
            let p2 = tournament_select(&fitnesses, config.tournament_size, &mut streams.selection)?;
            if pox_ws.crossover_into(
                (chromosomes[p1].genes(), chromosomes[p2].genes()),
                num_jobs,
                config.crossover_rate,
                &mut streams.crossover,
                (&mut buf1, &mut buf2),
            )? {
                crossover_count += 1;
            }

            for raw_buf in [&buf1, &buf2] {
                if next_chrs.len() >= pop_size || counter.remaining() == 0 {
                    break;
                }
                let mut child = ObrChromosome::new(raw_buf.clone());
                if mutate_obr(&mut child, config.mutation_rate, &mut streams.mutation)? {
                    mutation_count += 1;
                }
                let Some((mut out, _)) = decode_and_local_search(
                    instance,
                    &child,
                    config.local_search,
                    false,
                    counter,
                    decoder_ws,
                    ls_decoder_ws,
                    graph_eval,
                    streams,
                )?
                else {
                    break;
                };
                decode_evals += 1;
                ls_evals += out.ls_evals_consumed;
                if out.post_ls_makespan < best_makespan {
                    best_makespan = out.post_ls_makespan;
                    best_chr = out.improved_genotype.clone();
                    best_found_eval = counter.total();
                    best_found_gen = gen_next;
                    milestones.push((best_found_eval, best_makespan));
                }
                match config.config_id {
                    PhaseCConfigId::C02 => {
                        apply_lamarckian_writeback(instance, &mut out)?;
                        if out.post_ls_makespan < out.raw_makespan {
                            writeback_count += 1;
                        }
                        next_chrs.push(out.effective_genotype);
                        next_imps.push(None);
                    }
                    PhaseCConfigId::C03 => {
                        apply_baldwinian_preservation(instance, &mut out)?;
                        next_chrs.push(out.effective_genotype);
                        next_imps.push(None);
                    }
                    PhaseCConfigId::C04 => {
                        let imp_opt = if out.post_ls_makespan < out.raw_makespan {
                            Some(out.improved_genotype.clone())
                        } else {
                            None
                        };
                        next_chrs.push(out.original_genotype);
                        next_imps.push(imp_opt);
                    }
                    _ => unreachable!(),
                }
                next_fits.push(out.post_ls_makespan);
            }
        }

        if next_chrs.len() < 2 {
            break;
        }
        if matches!(config.config_id, PhaseCConfigId::C04) {
            writeback_count += apply_partial_lamarckian_even_ranks(
                instance,
                &mut next_chrs,
                &next_imps,
                &next_fits,
            )?;
        }
        generation = gen_next;
        chromosomes = next_chrs;
        fitnesses = next_fits;
    }

    let (ver_cmax, ver_genes, ver_seqs) = verify_final_best(instance, &best_chr, decoder_ws)?;
    if ver_cmax < best_makespan {
        best_makespan = ver_cmax;
    }
    let anytime_cmax_auc = compute_stepwise_anytime_auc(&milestones, config.evaluation_budget);
    let total_elapsed_ns = u64::try_from(run_start.elapsed().as_nanos()).unwrap_or(u64::MAX);

    Ok(PhaseCRunResult {
        config_id: config.config_id.code(),
        initializer_id: config.initializer.code(),
        ls_neighborhood: Some(config.local_search.neighborhood.code()),
        ls_strategy: Some(config.local_search.strategy.name()),
        genotype_capacity: pop_size,
        evaluation_budget: config.evaluation_budget,
        evals_used: counter.total(),
        init_evals: counter.init_evals(),
        search_evals: counter.search_evals(),
        decode_evals,
        ls_evals,
        generations_executed: generation,
        best_makespan,
        best_found_eval,
        best_found_generation: best_found_gen,
        feasible_final: true,
        anytime_cmax_auc,
        final_obr_diversity: compute_obr_hamming_diversity(&chromosomes, num_jobs),
        final_lam_diversity: None,
        final_bald_diversity: None,
        final_gate_p: None,
        lam_best_makespan: None,
        bald_best_makespan: None,
        gate_p_trajectory: Vec::new(),
        best_milestones: milestones,
        crossover_count,
        mutation_count,
        writeback_count,
        migrations_applied: 0,
        best_chromosome: ver_genes,
        best_machine_sequences: ver_seqs,
        total_elapsed_ns,
    })
}

/// Executes any single Phase C configuration (`C01`–`C11`) cleanly inside Rust under equalized
/// budget accounting (`EvalCounter`).
///
/// # Errors
/// Returns [`JsspError`] if config validation, decoding, local search, or M2 validation fails.
pub fn run_phase_c_ga(
    instance: &JsspInstance,
    config: &PhaseCGaConfig,
    streams: &mut RunRngStreams,
) -> Result<PhaseCRunResult, JsspError> {
    config.validate()?;
    let run_start = Instant::now();

    if matches!(config.config_id, PhaseCConfigId::C01) {
        let ga_cfg = ClassicalGaConfig {
            population_size: config.population_size,
            tournament_size: config.tournament_size,
            crossover_rate: config.crossover_rate,
            mutation_rate: config.mutation_rate,
            elitism_count: config.elitism_count,
            evaluation_budget: config.evaluation_budget,
        };
        let res = run_phase_a_ga(instance, config.initializer, &ga_cfg, streams)?;
        let mut decoder_ws = SerialDecoderWorkspace::new(instance);
        let best_obr = ObrChromosome::from_job_indices(&res.best_chromosome);
        let (ver_cmax, ver_genes, ver_seqs) =
            verify_final_best(instance, &best_obr, &mut decoder_ws)?;
        let mut milestones: Vec<(usize, u64)> = res
            .trajectory
            .iter()
            .map(|s| (s.evals_used, s.best_makespan))
            .collect();
        if milestones.is_empty() {
            milestones.push((res.best_found_eval, ver_cmax));
        }
        return Ok(PhaseCRunResult {
            config_id: "C01",
            initializer_id: config.initializer.code(),
            ls_neighborhood: None,
            ls_strategy: None,
            genotype_capacity: config.population_size,
            evaluation_budget: res.evaluation_budget,
            evals_used: res.evals_used,
            init_evals: res.pop_size.min(res.evals_used),
            search_evals: res
                .evals_used
                .saturating_sub(res.pop_size.min(res.evals_used)),
            decode_evals: res.evals_used,
            ls_evals: 0,
            generations_executed: res.generations_executed,
            best_makespan: ver_cmax,
            best_found_eval: res.best_found_eval,
            best_found_generation: res.best_found_generation,
            feasible_final: res.final_pop.feasible_count > 0,
            anytime_cmax_auc: res.anytime_cmax_auc,
            final_obr_diversity: res.final_pop.obr_hamming_diversity,
            final_lam_diversity: None,
            final_bald_diversity: None,
            final_gate_p: None,
            lam_best_makespan: None,
            bald_best_makespan: None,
            gate_p_trajectory: Vec::new(),
            best_milestones: milestones,
            crossover_count: res.crossover_count,
            mutation_count: res.mutation_count,
            writeback_count: 0,
            migrations_applied: 0,
            best_chromosome: ver_genes,
            best_machine_sequences: ver_seqs,
            total_elapsed_ns: res.total_elapsed_ns,
        });
    }

    let mut counter = EvalCounter::new(config.evaluation_budget)?;
    let mut decoder_ws = SerialDecoderWorkspace::new(instance);
    let mut ls_decoder_ws = SerialDecoderWorkspace::new(instance);
    let mut graph_eval = FastGraphEvaluator::new(instance)?;

    if matches!(
        config.config_id,
        PhaseCConfigId::C02 | PhaseCConfigId::C03 | PhaseCConfigId::C04
    ) {
        return run_single_view_memetic(
            instance,
            config,
            &mut counter,
            &mut decoder_ws,
            &mut ls_decoder_ws,
            &mut graph_eval,
            streams,
        );
    }

    let summary = if matches!(config.config_id, PhaseCConfigId::C10) {
        run_unpaired_dual_pool_control(
            instance,
            config,
            &mut counter,
            &mut decoder_ws,
            &mut ls_decoder_ws,
            &mut graph_eval,
            streams,
        )?
    } else {
        run_dual_view_model_a(
            instance,
            config,
            &mut counter,
            &mut decoder_ws,
            &mut ls_decoder_ws,
            &mut graph_eval,
            streams,
        )?
    };

    let anytime_cmax_auc =
        compute_stepwise_anytime_auc(&summary.best_milestones, config.evaluation_budget);
    let total_elapsed_ns = u64::try_from(run_start.elapsed().as_nanos()).unwrap_or(u64::MAX);

    Ok(PhaseCRunResult {
        config_id: config.config_id.code(),
        initializer_id: config.initializer.code(),
        ls_neighborhood: Some(config.local_search.neighborhood.code()),
        ls_strategy: Some(config.local_search.strategy.name()),
        genotype_capacity: config.population_size,
        evaluation_budget: config.evaluation_budget,
        evals_used: counter.total(),
        init_evals: counter.init_evals(),
        search_evals: counter.search_evals(),
        decode_evals: summary.decode_evals,
        ls_evals: summary.ls_evals,
        generations_executed: summary.generations_executed,
        best_makespan: summary.best_makespan,
        best_found_eval: summary.best_found_eval,
        best_found_generation: summary.best_found_generation,
        feasible_final: summary.feasible_final,
        anytime_cmax_auc,
        final_obr_diversity: summary.final_obr_diversity,
        final_lam_diversity: Some(summary.final_lam_diversity),
        final_bald_diversity: Some(summary.final_bald_diversity),
        final_gate_p: summary.final_gate_p,
        lam_best_makespan: Some(summary.lam_best_makespan),
        bald_best_makespan: Some(summary.bald_best_makespan),
        gate_p_trajectory: summary.gate_p_trajectory,
        best_milestones: summary.best_milestones,
        crossover_count: summary.crossover_count,
        mutation_count: summary.mutation_count,
        writeback_count: summary.writeback_count,
        migrations_applied: summary.migrations_applied,
        best_chromosome: summary.best_chromosome,
        best_machine_sequences: summary.best_machine_sequences,
        total_elapsed_ns,
    })
}

/// Specification for one task inside [`run_phase_c_batch`].
#[derive(Debug, Clone)]
pub struct PhaseCBatchTask {
    pub config: PhaseCGaConfig,
    pub seed_context: SeedDerivationContext,
}

/// Executes a batch of Phase C tasks in parallel across OS threads preserving per-task determinism.
///
/// # Errors
/// Returns the first encountered [`JsspError`].
pub fn run_phase_c_batch(
    instance: &JsspInstance,
    tasks: &[PhaseCBatchTask],
    num_threads: usize,
) -> Result<Vec<PhaseCRunResult>, JsspError> {
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
                run_phase_c_ga(instance, &t.config, &mut streams)
            })
            .collect();
    }

    let chunk_size = tasks.len().div_ceil(resolved_threads);
    let chunk_results: Vec<Result<Vec<PhaseCRunResult>, JsspError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = tasks
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(move || {
                    let mut local = Vec::with_capacity(chunk.len());
                    for t in chunk {
                        let mut streams = t.seed_context.all_streams();
                        local.push(run_phase_c_ga(instance, &t.config, &mut streams)?);
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
                    detail: "worker thread panicked during Phase C batch".to_owned(),
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
