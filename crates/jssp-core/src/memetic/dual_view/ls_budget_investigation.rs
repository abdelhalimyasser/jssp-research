//! Exploratory Dual-View Local Search Budget Investigation (`exploratory/dual-view-ls-budget`).
//!
//! Isolates per-individual LS budget (`b_LS`), deterministic pre-LS candidate selection fraction
//! (`f_LS`), and short-horizon LS strategy (`SteepestDescent`, `FirstImprovement`,
//! `BudgetLimitedSa`) while preserving frozen confirmatory `C01`–`C11` behavior untouched.

use std::time::Instant;

use crate::decoder::SerialDecoderWorkspace;
use crate::error::JsspError;
use crate::feasibility::validate_schedule;
use crate::genetic::PoxWorkspace;
use crate::initialization::qmc::u01_from_u64;
use crate::initialization::{InitializerId, generate_initial_population};
use crate::local_search::descent::{
    SearchCounters, SearchTerminationReason, checked_probe_move, compute_stepwise_anytime_auc,
    generate_and_count, run_first_improvement, run_steepest_descent,
};
use crate::local_search::evaluation::FastGraphEvaluator;
use crate::model::{JobId, JsspInstance, Schedule};
use crate::neighborhoods::{GraphMove, NeighborhoodId};
use crate::representation::ObrChromosome;
use crate::rng::{RunRngStreams, SeedDerivationContext};
use crate::telemetry::{EvalCounter, compute_obr_hamming_diversity};

use super::super::baldwinian::apply_baldwinian_preservation;
use super::super::lamarckian::{MemeticEvalOutcome, apply_lamarckian_writeback};
use super::super::{PhaseCConfigId, PhaseCGaConfig, verify_final_best};
use super::{
    AdaptiveGateState, DualViewIndividual, FixedGate, MIGRATION_INTERVAL_GENS,
    normalize_dual_view_scores, plan_cross_view_migration, ranked_dual_view_indices,
    reproduce_dual_view_within_view, select_dual_view_parent,
};

const SA_SHORT_COOLING_RATE: f64 = 0.85;
const SA_SHORT_T_MIN: f64 = 1e-3;
const SA_SHORT_ACCEPT_RATIO: f64 = 0.50;

/// Supported local-search kernels for the exploratory Dual-View LS budget investigation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExploratoryLsStrategy {
    SteepestDescent,
    FirstImprovement,
    BudgetLimitedSa,
}

impl ExploratoryLsStrategy {
    /// Parses canonical strategy code or display string.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidGeneticConfiguration`] on unknown strategy names.
    pub fn from_code(code: &str) -> Result<Self, JsspError> {
        match code {
            "steepest_descent" | "Steepest Descent" => Ok(Self::SteepestDescent),
            "first_improvement" | "First Improvement" => Ok(Self::FirstImprovement),
            "budget_limited_sa" | "Budget-Limited SA" => Ok(Self::BudgetLimitedSa),
            _ => Err(JsspError::InvalidGeneticConfiguration {
                detail: format!(
                    "unknown exploratory LS strategy '{code}', expected steepest_descent | first_improvement | budget_limited_sa"
                ),
            }),
        }
    }

    /// Canonical snake_case identifier.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::SteepestDescent => "steepest_descent",
            Self::FirstImprovement => "first_improvement",
            Self::BudgetLimitedSa => "budget_limited_sa",
        }
    }
}

/// Complete configuration for one exploratory Dual-View LS budget allocation experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct DualViewLsAllocationConfig {
    pub experiment_id: String,
    pub base_config_id: PhaseCConfigId,
    pub initializer: InitializerId,
    pub population_size: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub elitism_count: usize,
    pub evaluation_budget: usize,
    pub neighborhood: NeighborhoodId,
    pub strategy: ExploratoryLsStrategy,
    pub max_ls_evals: usize,
    pub candidate_fraction: f64,
}

impl DualViewLsAllocationConfig {
    /// Standard matched-control configuration preserving `C06` GA parameters and `N5`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn standard(
        experiment_id: String,
        base_config_id: PhaseCConfigId,
        initializer: InitializerId,
        evaluation_budget: usize,
        neighborhood: NeighborhoodId,
        strategy: ExploratoryLsStrategy,
        max_ls_evals: usize,
        candidate_fraction: f64,
    ) -> Self {
        Self {
            experiment_id,
            base_config_id,
            initializer,
            population_size: 64,
            tournament_size: 2,
            crossover_rate: 0.85,
            mutation_rate: 0.15,
            elitism_count: 2,
            evaluation_budget,
            neighborhood,
            strategy,
            max_ls_evals,
            candidate_fraction,
        }
    }

    /// Validates Dual-View pairing, GA hyperparameters, and LS allocation bounds.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidGeneticConfiguration`] on invalid settings.
    pub fn validate(&self) -> Result<(), JsspError> {
        if !self.base_config_id.is_dual_view_paired() {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: format!(
                    "{} is not a paired Model A Dual-View config",
                    self.base_config_id.code()
                ),
            });
        }
        if !self.candidate_fraction.is_finite()
            || self.candidate_fraction <= 0.0
            || self.candidate_fraction > 1.0
        {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: format!(
                    "candidate_fraction must lie in (0.0, 1.0], got {}",
                    self.candidate_fraction
                ),
            });
        }
        if self.max_ls_evals == 0 {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: "max_ls_evals must be positive".to_owned(),
            });
        }
        let dummy = PhaseCGaConfig {
            config_id: self.base_config_id,
            initializer: self.initializer,
            population_size: self.population_size,
            tournament_size: self.tournament_size,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            elitism_count: self.elitism_count,
            evaluation_budget: self.evaluation_budget,
            local_search: super::super::MemeticLsConfig {
                neighborhood: self.neighborhood,
                strategy: crate::metaheuristics::PhaseBStrategy::SteepestDescent,
                max_ls_evals: self.max_ls_evals,
            },
        };
        dummy.validate()
    }
}

/// Branch-level decode, LS evaluation, and move-acceptance counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BranchLsTelemetry {
    pub decode_evals: usize,
    pub ls_evals: usize,
    pub ls_calls: usize,
    pub improving_calls: usize,
    pub evaluated_moves: usize,
    pub feasible_moves: usize,
    pub rejected_cyclic_moves: usize,
    pub improving_moves: usize,
    pub accepted_moves: usize,
}

impl BranchLsTelemetry {
    /// Total evaluated candidate moves not committed/accepted (cyclic + non-accepted).
    #[must_use]
    pub const fn rejected_moves(&self) -> usize {
        self.evaluated_moves.saturating_sub(self.accepted_moves)
    }

    fn record_counters(&mut self, c: &SearchCounters) {
        self.evaluated_moves += c.evaluated_moves;
        self.feasible_moves += c.feasible_moves;
        self.rejected_cyclic_moves += c.rejected_cyclic_moves;
        self.improving_moves += c.improving_moves;
        self.accepted_moves += c.accepted_moves;
    }
}

/// Complete outcome and branch-specific telemetry for one exploratory Dual-View run.
#[derive(Debug, Clone, PartialEq)]
pub struct DualViewExploratoryResult {
    pub experiment_id: String,
    pub base_config_id: &'static str,
    pub initializer_id: &'static str,
    pub ls_neighborhood: &'static str,
    pub ls_strategy: &'static str,
    pub max_ls_evals: usize,
    pub candidate_fraction: f64,
    pub genotype_capacity: usize,
    pub evaluation_budget: usize,
    pub evals_used: usize,
    pub init_evals: usize,
    pub search_evals: usize,
    pub generations_executed: usize,
    pub best_makespan: u64,
    pub best_found_eval: usize,
    pub best_found_generation: usize,
    pub feasible_final: bool,
    pub anytime_cmax_auc: f64,
    pub total_elapsed_ns: u64,
    pub final_obr_diversity: f64,
    pub final_lam_diversity: f64,
    pub final_bald_diversity: f64,
    pub final_gate_p: Option<f64>,
    pub lam_best_makespan: u64,
    pub bald_best_makespan: u64,
    pub crossover_count: usize,
    pub mutation_count: usize,
    pub writeback_count: usize,
    pub migrations_applied: usize,
    pub lam_branch: BranchLsTelemetry,
    pub bald_branch: BranchLsTelemetry,
    pub gate_p_trajectory: Vec<(usize, f64)>,
    pub diversity_trajectory: Vec<(usize, f64, f64, f64)>,
    pub best_milestones: Vec<(usize, u64)>,
    pub best_chromosome: Vec<usize>,
    pub best_machine_sequences: Vec<Vec<(usize, usize)>>,
}

/// Short-horizon memetic Simulated Annealing kernel calibrated for tiny `b_LS <= 50` caps.
///
/// Uses `n_cal = (max_ls_evals / 5).clamp(2, 8).min(remaining)` calibration proposals while
/// retaining any strictly improving move seen during calibration, then applies geometric
/// cooling (`gamma = 0.85`) every `epoch_len = (max_ls_evals / 5).max(2)` evaluations.
#[allow(clippy::too_many_arguments)]
pub fn run_budget_limited_sa(
    evaluator: &mut FastGraphEvaluator,
    neighborhood: NeighborhoodId,
    max_ls_evals: usize,
    counter: &mut EvalCounter,
    counters: &mut SearchCounters,
    streams: &mut RunRngStreams,
    best_makespan: &mut u64,
    best_found_eval: &mut usize,
    best_seqs: &mut Vec<Vec<u32>>,
    best_milestones: &mut Vec<(usize, u64)>,
) -> Result<(u64, SearchTerminationReason), JsspError> {
    let Some(mut curr_state) = evaluator.analyze_current()? else {
        return Err(JsspError::InvalidDisjunctiveGraph {
            detail: "Budget-Limited SA received cyclic starting graph".to_owned(),
        });
    };
    let mut curr_cmax = curr_state.makespan;

    let n_cal = (max_ls_evals / 5).clamp(2, 8).min(counter.remaining());
    let mut uphill_sum = 0.0_f64;
    let mut uphill_count = 0usize;
    let mut best_cal_move: Option<(GraphMove, u64)> = None;

    {
        let gen_init = generate_and_count(evaluator, &curr_state, neighborhood, counters);
        if gen_init.moves.is_empty() {
            return Ok((curr_cmax, SearchTerminationReason::EmptyNeighborhood));
        }
        for _ in 0..n_cal {
            let pick = streams
                .local_search_sample
                .gen_range_usize(gen_init.moves.len())?;
            let mv = gen_init.moves[pick];
            let Some(probe_res) = checked_probe_move(evaluator, &mv, counter, counters)? else {
                return Ok((*best_makespan, SearchTerminationReason::BudgetExhausted));
            };
            if let Some(cand_cmax) = probe_res {
                if cand_cmax > curr_cmax {
                    uphill_sum += (cand_cmax - curr_cmax) as f64;
                    uphill_count += 1;
                } else if cand_cmax < curr_cmax {
                    counters.improving_moves += 1;
                    let is_better = match best_cal_move {
                        None => true,
                        Some((_, bc)) => cand_cmax < bc,
                    };
                    if is_better {
                        best_cal_move = Some((mv, cand_cmax));
                    }
                }
            }
        }
    }

    if let Some((best_mv, new_cmax)) = best_cal_move
        && new_cmax < curr_cmax
    {
        evaluator.commit_move(&best_mv);
        counters.iterations += 1;
        counters.accepted_moves += 1;
        curr_state =
            evaluator
                .analyze_current()?
                .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                    detail: "committed Budget-Limited SA calibration move induced cycle".to_owned(),
                })?;
        curr_cmax = curr_state.makespan;
        if curr_cmax < *best_makespan {
            *best_makespan = curr_cmax;
            *best_found_eval = counter.total();
            *best_seqs = evaluator.snapshot_machine_seqs();
            best_milestones.push((*best_found_eval, curr_cmax));
        }
    }

    let mut temp = if uphill_count > 0 {
        let mean_delta = uphill_sum / (uphill_count as f64);
        (-mean_delta / SA_SHORT_ACCEPT_RATIO.ln()).max(0.5)
    } else {
        (0.01 * (curr_cmax as f64)).max(0.5)
    };

    let epoch_len = (max_ls_evals / 5).max(2);
    let mut epoch_steps = 0usize;

    while counter.remaining() > 0 {
        let gen_n = generate_and_count(evaluator, &curr_state, neighborhood, counters);
        if gen_n.moves.is_empty() {
            return Ok((*best_makespan, SearchTerminationReason::EmptyNeighborhood));
        }
        let pick = streams
            .local_search_sample
            .gen_range_usize(gen_n.moves.len())?;
        let mv = gen_n.moves[pick];
        let Some(probe_res) = checked_probe_move(evaluator, &mv, counter, counters)? else {
            return Ok((*best_makespan, SearchTerminationReason::BudgetExhausted));
        };

        epoch_steps += 1;
        if let Some(cand_cmax) = probe_res {
            let accept = if cand_cmax <= curr_cmax {
                if cand_cmax < curr_cmax {
                    counters.improving_moves += 1;
                }
                true
            } else {
                let delta = (cand_cmax - curr_cmax) as f64;
                let prob = (-delta / temp).exp();
                u01_from_u64(streams.sa_accept.next_u64()) < prob
            };
            if accept {
                evaluator.commit_move(&mv);
                counters.iterations += 1;
                counters.accepted_moves += 1;
                curr_state = evaluator.analyze_current()?.ok_or_else(|| {
                    JsspError::InvalidDisjunctiveGraph {
                        detail: "accepted Budget-Limited SA move induced cycle".to_owned(),
                    }
                })?;
                curr_cmax = curr_state.makespan;
                if curr_cmax < *best_makespan {
                    *best_makespan = curr_cmax;
                    *best_found_eval = counter.total();
                    *best_seqs = evaluator.snapshot_machine_seqs();
                    best_milestones.push((*best_found_eval, curr_cmax));
                }
            }
        }

        if epoch_steps >= epoch_len {
            epoch_steps = 0;
            temp = (temp * SA_SHORT_COOLING_RATE).max(SA_SHORT_T_MIN);
        }
    }

    Ok((*best_makespan, SearchTerminationReason::BudgetExhausted))
}

#[allow(clippy::too_many_arguments)]
fn apply_ls_on_decoded_schedule(
    instance: &JsspInstance,
    chr: &ObrChromosome,
    raw_sched: Schedule,
    strategy: ExploratoryLsStrategy,
    neighborhood: NeighborhoodId,
    max_ls_evals: usize,
    counter: &mut EvalCounter,
    ls_decoder_ws: &mut SerialDecoderWorkspace,
    graph_eval: &mut FastGraphEvaluator,
    streams: &mut RunRngStreams,
) -> Result<(MemeticEvalOutcome, SearchCounters), JsspError> {
    let raw_cmax = raw_sched.makespan();
    let allowed_ls = max_ls_evals.min(counter.remaining());
    if allowed_ls == 0 {
        return Ok((
            MemeticEvalOutcome {
                original_genotype: chr.clone(),
                improved_genotype: chr.clone(),
                effective_genotype: chr.clone(),
                raw_makespan: raw_cmax,
                post_ls_makespan: raw_cmax,
                ls_evals_consumed: 0,
                best_schedule: raw_sched,
            },
            SearchCounters::default(),
        ));
    }

    graph_eval.load_from_schedule(&raw_sched)?;
    let mut ls_counter = EvalCounter::new(allowed_ls)?;
    let mut ls_counters = SearchCounters::default();
    let mut ls_best_cmax = raw_cmax;
    let mut dummy_best_eval = 0usize;
    let mut ls_best_seqs = graph_eval.snapshot_machine_seqs();
    let mut dummy_milestones = vec![(0usize, raw_cmax)];

    match strategy {
        ExploratoryLsStrategy::SteepestDescent => {
            let _ = run_steepest_descent(
                graph_eval,
                neighborhood,
                &mut ls_counter,
                &mut ls_counters,
                &mut ls_best_cmax,
                &mut dummy_best_eval,
                &mut ls_best_seqs,
                &mut dummy_milestones,
            )?;
        }
        ExploratoryLsStrategy::FirstImprovement => {
            let _ = run_first_improvement(
                graph_eval,
                neighborhood,
                &mut ls_counter,
                &mut ls_counters,
                &mut ls_best_cmax,
                &mut dummy_best_eval,
                &mut ls_best_seqs,
                &mut dummy_milestones,
            )?;
        }
        ExploratoryLsStrategy::BudgetLimitedSa => {
            let _ = run_budget_limited_sa(
                graph_eval,
                neighborhood,
                max_ls_evals,
                &mut ls_counter,
                &mut ls_counters,
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
        Ok((
            MemeticEvalOutcome {
                original_genotype: chr.clone(),
                improved_genotype: improved_obr.clone(),
                effective_genotype: improved_obr,
                raw_makespan: raw_cmax,
                post_ls_makespan: post_cmax,
                ls_evals_consumed: ls_used,
                best_schedule: ls_sched,
            },
            ls_counters,
        ))
    } else {
        Ok((
            MemeticEvalOutcome {
                original_genotype: chr.clone(),
                improved_genotype: chr.clone(),
                effective_genotype: chr.clone(),
                raw_makespan: raw_cmax,
                post_ls_makespan: raw_cmax,
                ls_evals_consumed: ls_used,
                best_schedule: raw_sched,
            },
            ls_counters,
        ))
    }
}

enum ExploratoryGate {
    Fixed(FixedGate),
    Adaptive(AdaptiveGateState),
}

impl ExploratoryGate {
    fn p(&self) -> f64 {
        match self {
            Self::Fixed(f) => f.p_lamarckian,
            Self::Adaptive(a) => a.p,
        }
    }
}

fn build_gate(config_id: PhaseCConfigId) -> Result<ExploratoryGate, JsspError> {
    match config_id {
        PhaseCConfigId::C05 => Ok(ExploratoryGate::Fixed(FixedGate::new(0.50)?)),
        PhaseCConfigId::C07 => Ok(ExploratoryGate::Fixed(FixedGate::new(0.25)?)),
        PhaseCConfigId::C08 => Ok(ExploratoryGate::Fixed(FixedGate::new(0.75)?)),
        PhaseCConfigId::C06 | PhaseCConfigId::C11 => Ok(ExploratoryGate::Adaptive(
            AdaptiveGateState::new(0.70, 0.30)?,
        )),
        PhaseCConfigId::C09 => Ok(ExploratoryGate::Adaptive(AdaptiveGateState::new(
            1.00, 0.00,
        )?)),
        other => Err(JsspError::InvalidGeneticConfiguration {
            detail: format!("{} is not a Model A Dual-View config", other.code()),
        }),
    }
}

struct SharedEvalState<'a> {
    instance: &'a JsspInstance,
    config: &'a DualViewLsAllocationConfig,
    counter: &'a mut EvalCounter,
    decoder_ws: &'a mut SerialDecoderWorkspace,
    ls_decoder_ws: &'a mut SerialDecoderWorkspace,
    graph_eval: &'a mut FastGraphEvaluator,
    streams: &'a mut RunRngStreams,
    lam_branch: BranchLsTelemetry,
    bald_branch: BranchLsTelemetry,
    writeback_count: usize,
    best_makespan: u64,
    best_chr: ObrChromosome,
    best_found_eval: usize,
    best_found_gen: usize,
    milestones: Vec<(usize, u64)>,
}

impl SharedEvalState<'_> {
    fn maybe_update_best(&mut self, cmax: u64, chr: &ObrChromosome, curr_gen: usize) {
        if cmax < self.best_makespan {
            self.best_makespan = cmax;
            self.best_chr = chr.clone();
            self.best_found_eval = self.counter.total();
            self.best_found_gen = curr_gen;
            self.milestones
                .push((self.best_found_eval, self.best_makespan));
        }
    }

    fn consume_decode(&mut self, is_init: bool) -> bool {
        if is_init {
            self.counter.try_consume_init()
        } else {
            self.counter.try_consume_search()
        }
    }
}

fn finish_view_ls(
    is_lam: bool,
    outcome: &mut MemeticEvalOutcome,
    ls_counters: &SearchCounters,
    state: &mut SharedEvalState<'_>,
    curr_gen: usize,
) -> Result<(), JsspError> {
    let tel = if is_lam {
        &mut state.lam_branch
    } else {
        &mut state.bald_branch
    };
    if outcome.ls_evals_consumed > 0 || ls_counters.evaluated_moves > 0 {
        tel.ls_calls += 1;
        tel.ls_evals += outcome.ls_evals_consumed;
        tel.record_counters(ls_counters);
        if outcome.post_ls_makespan < outcome.raw_makespan {
            tel.improving_calls += 1;
        }
    }

    if is_lam {
        apply_lamarckian_writeback(state.instance, outcome)?;
        if outcome.post_ls_makespan < outcome.raw_makespan {
            state.writeback_count += 1;
        }
        state.maybe_update_best(
            outcome.post_ls_makespan,
            &outcome.effective_genotype,
            curr_gen,
        );
    } else {
        apply_baldwinian_preservation(state.instance, outcome)?;
        state.maybe_update_best(
            outcome.post_ls_makespan,
            &outcome.improved_genotype,
            curr_gen,
        );
    }
    Ok(())
}

/// Exact `100%` control path matching `eval_dual_pair_views` in `dual_view/mod.rs` bit-for-bit.
fn eval_full_pair_sequential(
    pair_id: usize,
    lam_raw: &ObrChromosome,
    bald_raw: &ObrChromosome,
    is_init: bool,
    curr_gen: usize,
    state: &mut SharedEvalState<'_>,
) -> Result<Option<DualViewIndividual>, JsspError> {
    if !state.consume_decode(is_init) {
        return Ok(None);
    }
    state.lam_branch.decode_evals += 1;
    let lam_sched = state.decoder_ws.decode(state.instance, lam_raw, true)?;
    validate_schedule(state.instance, &lam_sched)?;
    let (mut lam_out, lam_cnt) = apply_ls_on_decoded_schedule(
        state.instance,
        lam_raw,
        lam_sched,
        state.config.strategy,
        state.config.neighborhood,
        state.config.max_ls_evals,
        state.counter,
        state.ls_decoder_ws,
        state.graph_eval,
        state.streams,
    )?;
    finish_view_ls(true, &mut lam_out, &lam_cnt, state, curr_gen)?;

    if !state.consume_decode(is_init) {
        return Ok(None);
    }
    state.bald_branch.decode_evals += 1;
    let bald_sched = state.decoder_ws.decode(state.instance, bald_raw, true)?;
    validate_schedule(state.instance, &bald_sched)?;
    let (mut bald_out, bald_cnt) = apply_ls_on_decoded_schedule(
        state.instance,
        bald_raw,
        bald_sched,
        state.config.strategy,
        state.config.neighborhood,
        state.config.max_ls_evals,
        state.counter,
        state.ls_decoder_ws,
        state.graph_eval,
        state.streams,
    )?;
    finish_view_ls(false, &mut bald_out, &bald_cnt, state, curr_gen)?;

    Ok(Some(DualViewIndividual {
        id: pair_id,
        lam_genotype: lam_out.effective_genotype,
        lam_fitness: lam_out.post_ls_makespan,
        lam_ls_delta: lam_out
            .raw_makespan
            .saturating_sub(lam_out.post_ls_makespan),
        lam_ls_evals: lam_out.ls_evals_consumed,
        bald_genotype: bald_out.effective_genotype,
        bald_fitness: bald_out.post_ls_makespan,
        bald_ls_delta: bald_out
            .raw_makespan
            .saturating_sub(bald_out.post_ls_makespan),
        bald_ls_evals: bald_out.ls_evals_consumed,
    }))
}

struct DecodedPairCandidate {
    pair_id: usize,
    lam_raw: ObrChromosome,
    lam_sched: Schedule,
    lam_cmax: u64,
    bald_raw: ObrChromosome,
    bald_sched: Schedule,
    bald_cmax: u64,
}

/// Evaluates a cohort of pairs when `candidate_fraction < 1.0` by first decoding all pairs,
/// deterministically ranking them by `(min(raw_L, raw_B), max(raw_L, raw_B), pair_id)`,
/// and applying bounded LS only to the top `K = clamp(ceil(f * M), 1, M)` pairs.
fn eval_cohort_with_candidate_selection(
    raw_pairs: &[(usize, ObrChromosome, ObrChromosome)],
    is_init: bool,
    curr_gen: usize,
    state: &mut SharedEvalState<'_>,
) -> Result<Vec<DualViewIndividual>, JsspError> {
    let mut cohort: Vec<DecodedPairCandidate> = Vec::with_capacity(raw_pairs.len());
    for (pair_id, lam_raw, bald_raw) in raw_pairs {
        if !state.consume_decode(is_init) {
            break;
        }
        state.lam_branch.decode_evals += 1;
        let lam_sched = state.decoder_ws.decode(state.instance, lam_raw, true)?;
        validate_schedule(state.instance, &lam_sched)?;
        let lam_cmax = lam_sched.makespan();
        state.maybe_update_best(lam_cmax, lam_raw, curr_gen);

        if !state.consume_decode(is_init) {
            break;
        }
        state.bald_branch.decode_evals += 1;
        let bald_sched = state.decoder_ws.decode(state.instance, bald_raw, true)?;
        validate_schedule(state.instance, &bald_sched)?;
        let bald_cmax = bald_sched.makespan();
        state.maybe_update_best(bald_cmax, bald_raw, curr_gen);

        cohort.push(DecodedPairCandidate {
            pair_id: *pair_id,
            lam_raw: lam_raw.clone(),
            lam_sched,
            lam_cmax,
            bald_raw: bald_raw.clone(),
            bald_sched,
            bald_cmax,
        });
    }

    if cohort.is_empty() {
        return Ok(Vec::new());
    }

    let m = cohort.len();
    let k = ((state.config.candidate_fraction * (m as f64)).ceil() as usize).clamp(1, m);
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by_key(|&i| {
        let lo = cohort[i].lam_cmax.min(cohort[i].bald_cmax);
        let hi = cohort[i].lam_cmax.max(cohort[i].bald_cmax);
        (lo, hi, cohort[i].pair_id)
    });

    let mut out_opts: Vec<Option<DualViewIndividual>> = vec![None; m];
    for &idx in order.iter().take(k) {
        let c = &cohort[idx];
        let mut lam_out = if state.counter.remaining() > 0 {
            let (out, cnt) = apply_ls_on_decoded_schedule(
                state.instance,
                &c.lam_raw,
                c.lam_sched.clone(),
                state.config.strategy,
                state.config.neighborhood,
                state.config.max_ls_evals,
                state.counter,
                state.ls_decoder_ws,
                state.graph_eval,
                state.streams,
            )?;
            let mut o = out;
            finish_view_ls(true, &mut o, &cnt, state, curr_gen)?;
            o
        } else {
            MemeticEvalOutcome {
                original_genotype: c.lam_raw.clone(),
                improved_genotype: c.lam_raw.clone(),
                effective_genotype: c.lam_raw.clone(),
                raw_makespan: c.lam_cmax,
                post_ls_makespan: c.lam_cmax,
                ls_evals_consumed: 0,
                best_schedule: c.lam_sched.clone(),
            }
        };

        let mut bald_out = if state.counter.remaining() > 0 {
            let (out, cnt) = apply_ls_on_decoded_schedule(
                state.instance,
                &c.bald_raw,
                c.bald_sched.clone(),
                state.config.strategy,
                state.config.neighborhood,
                state.config.max_ls_evals,
                state.counter,
                state.ls_decoder_ws,
                state.graph_eval,
                state.streams,
            )?;
            let mut o = out;
            finish_view_ls(false, &mut o, &cnt, state, curr_gen)?;
            o
        } else {
            MemeticEvalOutcome {
                original_genotype: c.bald_raw.clone(),
                improved_genotype: c.bald_raw.clone(),
                effective_genotype: c.bald_raw.clone(),
                raw_makespan: c.bald_cmax,
                post_ls_makespan: c.bald_cmax,
                ls_evals_consumed: 0,
                best_schedule: c.bald_sched.clone(),
            }
        };

        // Re-assert outcome consistency without extra LS
        let _ = (&mut lam_out, &mut bald_out);
        out_opts[idx] = Some(DualViewIndividual {
            id: c.pair_id,
            lam_genotype: lam_out.effective_genotype,
            lam_fitness: lam_out.post_ls_makespan,
            lam_ls_delta: lam_out
                .raw_makespan
                .saturating_sub(lam_out.post_ls_makespan),
            lam_ls_evals: lam_out.ls_evals_consumed,
            bald_genotype: bald_out.effective_genotype,
            bald_fitness: bald_out.post_ls_makespan,
            bald_ls_delta: bald_out
                .raw_makespan
                .saturating_sub(bald_out.post_ls_makespan),
            bald_ls_evals: bald_out.ls_evals_consumed,
        });
    }

    for idx in order.into_iter().skip(k) {
        let c = cohort[idx].pair_id;
        out_opts[idx] = Some(DualViewIndividual {
            id: c,
            lam_genotype: cohort[idx].lam_raw.clone(),
            lam_fitness: cohort[idx].lam_cmax,
            lam_ls_delta: 0,
            lam_ls_evals: 0,
            bald_genotype: cohort[idx].bald_raw.clone(),
            bald_fitness: cohort[idx].bald_cmax,
            bald_ls_delta: 0,
            bald_ls_evals: 0,
        });
    }

    Ok(out_opts.into_iter().flatten().collect())
}

fn record_generation_diversity_and_gate(
    pop: &[DualViewIndividual],
    num_jobs: usize,
    generation: usize,
    gate: &mut ExploratoryGate,
    gate_p_trajectory: &mut Vec<(usize, f64)>,
    diversity_trajectory: &mut Vec<(usize, f64, f64, f64)>,
) {
    let lam_chrs: Vec<ObrChromosome> = pop.iter().map(|i| i.lam_genotype.clone()).collect();
    let bald_chrs: Vec<ObrChromosome> = pop.iter().map(|i| i.bald_genotype.clone()).collect();
    let mut all_chrs = Vec::with_capacity(lam_chrs.len() + bald_chrs.len());
    all_chrs.extend(lam_chrs.iter().cloned());
    all_chrs.extend(bald_chrs.iter().cloned());

    let obr_div = compute_obr_hamming_diversity(&all_chrs, num_jobs);
    let d_ham_l = compute_obr_hamming_diversity(&lam_chrs, num_jobs);
    let d_ham_b = compute_obr_hamming_diversity(&bald_chrs, num_jobs);
    diversity_trajectory.push((generation, obr_div, d_ham_l, d_ham_b));

    if let ExploratoryGate::Adaptive(adapt) = gate
        && pop.len() >= 2
    {
        let (_, mean_sl, mean_sb) = normalize_dual_view_scores(pop, adapt.p);
        let lam_ds: u64 = pop.iter().map(|i| i.lam_ls_delta).sum();
        let lam_ev: usize = pop.iter().map(|i| i.lam_ls_evals).sum();
        let bald_ds: u64 = pop.iter().map(|i| i.bald_ls_delta).sum();
        let bald_ev: usize = pop.iter().map(|i| i.bald_ls_evals).sum();
        let dp = AdaptiveGateState::compute_delta_perf(
            mean_sl, mean_sb, lam_ds, lam_ev, bald_ds, bald_ev,
        );
        adapt.update(dp, d_ham_l, d_ham_b);
        if generation == 0 {
            gate_p_trajectory[0] = (0, adapt.p);
        } else {
            gate_p_trajectory.push((generation, adapt.p));
        }
    } else if generation > 0 {
        gate_p_trajectory.push((generation, gate.p()));
    }
}

/// Executes one exploratory Dual-View LS budget allocation run under exact `EvalCounter` parity.
///
/// # Errors
/// Returns [`JsspError`] on invalid configuration, decode failure, or M2 validation failure.
#[allow(clippy::too_many_lines)]
pub fn run_dual_view_ls_allocation(
    instance: &JsspInstance,
    config: &DualViewLsAllocationConfig,
    streams: &mut RunRngStreams,
) -> Result<DualViewExploratoryResult, JsspError> {
    config.validate()?;
    let run_start = Instant::now();

    let n_pairs = config.population_size / 2;
    let elite_pairs = (config.elitism_count / 2).max(1).min(n_pairs - 1);
    let num_jobs = instance.num_jobs();
    let chrom_len = instance.num_operations();
    let use_full_seq = config.candidate_fraction >= 1.0 - 1e-12;
    let use_migration = matches!(config.base_config_id, PhaseCConfigId::C11);

    let mut gate = build_gate(config.base_config_id)?;
    let mut counter = EvalCounter::new(config.evaluation_budget)?;
    let mut decoder_ws = SerialDecoderWorkspace::new(instance);
    let mut ls_decoder_ws = SerialDecoderWorkspace::new(instance);
    let mut graph_eval = FastGraphEvaluator::new(instance)?;

    let generated = generate_initial_population(config.initializer, instance, n_pairs, streams)?;
    let init_chr0 = generated.chromosomes[0].clone();

    let mut state = SharedEvalState {
        instance,
        config,
        counter: &mut counter,
        decoder_ws: &mut decoder_ws,
        ls_decoder_ws: &mut ls_decoder_ws,
        graph_eval: &mut graph_eval,
        streams,
        lam_branch: BranchLsTelemetry::default(),
        bald_branch: BranchLsTelemetry::default(),
        writeback_count: 0,
        best_makespan: u64::MAX,
        best_chr: init_chr0,
        best_found_eval: 0,
        best_found_gen: 0,
        milestones: Vec::new(),
    };

    let mut next_pair_id = 0usize;
    let mut pop: Vec<DualViewIndividual> = if use_full_seq {
        let mut init_pop = Vec::with_capacity(n_pairs);
        for init_chr in &generated.chromosomes {
            let Some(ind) =
                eval_full_pair_sequential(next_pair_id, init_chr, init_chr, true, 0, &mut state)?
            else {
                break;
            };
            next_pair_id += 1;
            init_pop.push(ind);
        }
        init_pop
    } else {
        let raw_pairs: Vec<(usize, ObrChromosome, ObrChromosome)> = generated
            .chromosomes
            .iter()
            .map(|c| {
                let id = next_pair_id;
                next_pair_id += 1;
                (id, c.clone(), c.clone())
            })
            .collect();
        eval_cohort_with_candidate_selection(&raw_pairs, true, 0, &mut state)?
    };

    let mut gate_p_trajectory: Vec<(usize, f64)> = vec![(0, gate.p())];
    let mut diversity_trajectory: Vec<(usize, f64, f64, f64)> = Vec::new();
    record_generation_diversity_and_gate(
        &pop,
        num_jobs,
        0,
        &mut gate,
        &mut gate_p_trajectory,
        &mut diversity_trajectory,
    );

    let mut pox_ws = PoxWorkspace::new(num_jobs, chrom_len);
    let mut buf1: Vec<JobId> = Vec::with_capacity(chrom_len);
    let mut buf2: Vec<JobId> = Vec::with_capacity(chrom_len);
    let mut generation = 0usize;
    let mut crossover_count = 0usize;
    let mut mutation_count = 0usize;
    let mut migrations_applied = 0usize;

    while state.counter.remaining() > 0 && pop.len() >= 2 {
        let p_cur = gate.p();
        let (scores, _, _) = normalize_dual_view_scores(&pop, p_cur);
        let ranked = ranked_dual_view_indices(&pop, &scores);

        let mut next_pop: Vec<DualViewIndividual> = Vec::with_capacity(n_pairs);
        for &e_idx in ranked.iter().take(elite_pairs) {
            let mut elite = pop[e_idx].clone();
            elite.lam_ls_delta = 0;
            elite.lam_ls_evals = 0;
            elite.bald_ls_delta = 0;
            elite.bald_ls_evals = 0;
            next_pop.push(elite);
        }

        let gen_next = generation + 1;
        if use_full_seq {
            while next_pop.len() < n_pairs && state.counter.remaining() > 0 {
                let pa_idx = select_dual_view_parent(
                    &pop,
                    &scores,
                    config.tournament_size,
                    &mut state.streams.selection,
                )?;
                let pb_idx = select_dual_view_parent(
                    &pop,
                    &scores,
                    config.tournament_size,
                    &mut state.streams.selection,
                )?;
                let ((lam_c1, bald_c1), (lam_c2, bald_c2), cx, mt) =
                    reproduce_dual_view_within_view(
                        &pop[pa_idx],
                        &pop[pb_idx],
                        num_jobs,
                        config.crossover_rate,
                        config.mutation_rate,
                        &mut pox_ws,
                        (&mut buf1, &mut buf2),
                        state.streams,
                    )?;
                crossover_count += cx;
                mutation_count += mt;

                let Some(child1) = eval_full_pair_sequential(
                    next_pair_id,
                    &lam_c1,
                    &bald_c1,
                    false,
                    gen_next,
                    &mut state,
                )?
                else {
                    break;
                };
                next_pair_id += 1;
                next_pop.push(child1);

                if next_pop.len() < n_pairs && state.counter.remaining() > 0 {
                    let Some(child2) = eval_full_pair_sequential(
                        next_pair_id,
                        &lam_c2,
                        &bald_c2,
                        false,
                        gen_next,
                        &mut state,
                    )?
                    else {
                        break;
                    };
                    next_pair_id += 1;
                    next_pop.push(child2);
                }
            }
        } else {
            let needed = n_pairs.saturating_sub(next_pop.len());
            let mut raw_pairs: Vec<(usize, ObrChromosome, ObrChromosome)> =
                Vec::with_capacity(needed);
            while raw_pairs.len() < needed {
                let pa_idx = select_dual_view_parent(
                    &pop,
                    &scores,
                    config.tournament_size,
                    &mut state.streams.selection,
                )?;
                let pb_idx = select_dual_view_parent(
                    &pop,
                    &scores,
                    config.tournament_size,
                    &mut state.streams.selection,
                )?;
                let ((lam_c1, bald_c1), (lam_c2, bald_c2), cx, mt) =
                    reproduce_dual_view_within_view(
                        &pop[pa_idx],
                        &pop[pb_idx],
                        num_jobs,
                        config.crossover_rate,
                        config.mutation_rate,
                        &mut pox_ws,
                        (&mut buf1, &mut buf2),
                        state.streams,
                    )?;
                crossover_count += cx;
                mutation_count += mt;

                let id1 = next_pair_id;
                next_pair_id += 1;
                raw_pairs.push((id1, lam_c1, bald_c1));

                if raw_pairs.len() < needed {
                    let id2 = next_pair_id;
                    next_pair_id += 1;
                    raw_pairs.push((id2, lam_c2, bald_c2));
                }
            }
            let evaluated =
                eval_cohort_with_candidate_selection(&raw_pairs, false, gen_next, &mut state)?;
            next_pop.extend(evaluated);
        }

        if next_pop.len() < 2 {
            break;
        }
        generation = gen_next;
        pop = next_pop;

        if use_migration
            && generation.is_multiple_of(MIGRATION_INTERVAL_GENS)
            && state.counter.remaining() > 0
        {
            let plan = plan_cross_view_migration(&pop);
            for (slot, donor_chr) in plan.targets {
                if !state.consume_decode(false) {
                    break;
                }
                if plan.migrate_into_baldwinian {
                    state.bald_branch.decode_evals += 1;
                } else {
                    state.lam_branch.decode_evals += 1;
                }
                let donor_sched = state.decoder_ws.decode(state.instance, &donor_chr, true)?;
                validate_schedule(state.instance, &donor_sched)?;
                let (mut out, cnt) = apply_ls_on_decoded_schedule(
                    state.instance,
                    &donor_chr,
                    donor_sched,
                    config.strategy,
                    config.neighborhood,
                    config.max_ls_evals,
                    state.counter,
                    state.ls_decoder_ws,
                    state.graph_eval,
                    state.streams,
                )?;
                finish_view_ls(
                    !plan.migrate_into_baldwinian,
                    &mut out,
                    &cnt,
                    &mut state,
                    generation,
                )?;
                migrations_applied += 1;
                if plan.migrate_into_baldwinian {
                    pop[slot].bald_genotype = out.effective_genotype;
                    pop[slot].bald_fitness = out.post_ls_makespan;
                } else {
                    pop[slot].lam_genotype = out.effective_genotype;
                    pop[slot].lam_fitness = out.post_ls_makespan;
                }
            }
        }

        record_generation_diversity_and_gate(
            &pop,
            num_jobs,
            generation,
            &mut gate,
            &mut gate_p_trajectory,
            &mut diversity_trajectory,
        );
    }

    let lam_chrs: Vec<ObrChromosome> = pop.iter().map(|i| i.lam_genotype.clone()).collect();
    let bald_chrs: Vec<ObrChromosome> = pop.iter().map(|i| i.bald_genotype.clone()).collect();
    let mut all_chrs = Vec::with_capacity(lam_chrs.len() + bald_chrs.len());
    all_chrs.extend(lam_chrs.iter().cloned());
    all_chrs.extend(bald_chrs.iter().cloned());

    let lam_best = pop
        .iter()
        .map(|i| i.lam_fitness)
        .min()
        .unwrap_or(state.best_makespan);
    let bald_best = pop
        .iter()
        .map(|i| i.bald_fitness)
        .min()
        .unwrap_or(state.best_makespan);

    let (ver_cmax, ver_genes, ver_seqs) =
        verify_final_best(state.instance, &state.best_chr, state.decoder_ws)?;
    let final_best = state.best_makespan.min(ver_cmax);
    let anytime_cmax_auc =
        compute_stepwise_anytime_auc(&state.milestones, config.evaluation_budget);
    let total_elapsed_ns = u64::try_from(run_start.elapsed().as_nanos()).unwrap_or(u64::MAX);

    Ok(DualViewExploratoryResult {
        experiment_id: config.experiment_id.clone(),
        base_config_id: config.base_config_id.code(),
        initializer_id: config.initializer.code(),
        ls_neighborhood: config.neighborhood.code(),
        ls_strategy: config.strategy.code(),
        max_ls_evals: config.max_ls_evals,
        candidate_fraction: config.candidate_fraction,
        genotype_capacity: config.population_size,
        evaluation_budget: config.evaluation_budget,
        evals_used: state.counter.total(),
        init_evals: state.counter.init_evals(),
        search_evals: state.counter.search_evals(),
        generations_executed: generation,
        best_makespan: final_best,
        best_found_eval: state.best_found_eval,
        best_found_generation: state.best_found_gen,
        feasible_final: true,
        anytime_cmax_auc,
        total_elapsed_ns,
        final_obr_diversity: compute_obr_hamming_diversity(&all_chrs, num_jobs),
        final_lam_diversity: compute_obr_hamming_diversity(&lam_chrs, num_jobs),
        final_bald_diversity: compute_obr_hamming_diversity(&bald_chrs, num_jobs),
        final_gate_p: Some(gate.p()),
        lam_best_makespan: lam_best,
        bald_best_makespan: bald_best,
        crossover_count,
        mutation_count,
        writeback_count: state.writeback_count,
        migrations_applied,
        lam_branch: state.lam_branch,
        bald_branch: state.bald_branch,
        gate_p_trajectory,
        diversity_trajectory,
        best_milestones: state.milestones,
        best_chromosome: ver_genes,
        best_machine_sequences: ver_seqs,
    })
}

/// Task descriptor for [`run_dual_view_exploratory_batch`].
#[derive(Debug, Clone)]
pub struct DualViewExploratoryTask {
    pub config: DualViewLsAllocationConfig,
    pub seed_context: SeedDerivationContext,
}

/// Executes a batch of exploratory Dual-View LS budget runs across OS threads deterministically.
///
/// # Errors
/// Returns the first encountered [`JsspError`].
pub fn run_dual_view_exploratory_batch(
    instance: &JsspInstance,
    tasks: &[DualViewExploratoryTask],
    num_threads: usize,
) -> Result<Vec<DualViewExploratoryResult>, JsspError> {
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
                run_dual_view_ls_allocation(instance, &t.config, &mut streams)
            })
            .collect();
    }

    let chunk_size = tasks.len().div_ceil(resolved_threads);
    let chunk_results: Vec<Result<Vec<DualViewExploratoryResult>, JsspError>> =
        std::thread::scope(|scope| {
            let handles: Vec<_> = tasks
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut local = Vec::with_capacity(chunk.len());
                        for t in chunk {
                            let mut streams = t.seed_context.all_streams();
                            local.push(run_dual_view_ls_allocation(
                                instance,
                                &t.config,
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
                        detail: "worker thread panicked during exploratory Dual-View batch"
                            .to_owned(),
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
