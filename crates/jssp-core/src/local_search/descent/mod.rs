//! Shared local search descent state, termination status, and telemetry accumulators (`D05`).

pub mod first_improvement;
pub mod random_improvement;
pub mod steepest;

pub use first_improvement::run_first_improvement;
pub use random_improvement::run_random_improvement;
pub use steepest::run_steepest_descent;

use crate::error::JsspError;
use crate::local_search::evaluation::{EvaluatedGraphState, FastGraphEvaluator};
use crate::neighborhoods::{NeighborhoodId, generate_neighborhood};
use crate::telemetry::EvalCounter;

/// Strict Phase B termination classification distinguishing proven local optimality
/// from budget exhaustion (`D05`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SearchTerminationReason {
    /// Exhaustive neighborhood evaluation completed with zero strictly improving moves.
    LocalOptimum,
    /// Evaluation budget hit zero before proving a local optimum (or at budget cap).
    BudgetExhausted,
    /// Initial or current critical path had zero valid candidate moves.
    EmptyNeighborhood,
}

impl SearchTerminationReason {
    /// Canonical lowercase identifier persisted into manifests and CSV summaries.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LocalOptimum => "local_optimum",
            Self::BudgetExhausted => "budget_exhausted",
            Self::EmptyNeighborhood => "empty_neighborhood",
        }
    }
}

/// Cumulative Phase B move, evaluation, and feasibility counters updated throughout a run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchCounters {
    pub iterations: usize,
    pub generated_moves: usize,
    pub duplicate_pruned_moves: usize,
    pub infeasible_precond_moves: usize,
    pub clipped_moves: usize,
    pub evaluated_moves: usize,
    pub feasible_moves: usize,
    pub rejected_cyclic_moves: usize,
    pub improving_moves: usize,
    pub accepted_moves: usize,
    pub adapter_left_shift_count: usize,
    pub perturbations_applied: usize,
    pub perturbations_accepted: usize,
    pub restarts_triggered: usize,
}

/// Complete telemetry outcome for one Phase B local-search or metaheuristic run.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseBRunResult {
    pub config_id: &'static str,
    pub neighborhood_id: &'static str,
    pub evaluation_budget: usize,
    pub evals_used: usize,
    pub initial_makespan: u64,
    pub final_makespan: u64,
    pub best_makespan: u64,
    pub best_found_eval: usize,
    pub feasible_final: bool,
    pub reached_local_optimum: bool,
    pub terminated_reason: SearchTerminationReason,
    pub counters: SearchCounters,
    pub anytime_cmax_auc: f64,
    pub best_milestones: Vec<(usize, u64)>,
    pub best_machine_sequences: Vec<Vec<(usize, usize)>>,
    pub total_elapsed_ns: u64,
}

/// Evaluates `mv` under `eval_counter`, updating move counters consistently across all
/// Phase B strategies. Returns `Ok(None)` if `eval_counter` is exhausted, or `Ok(Some(opt))`
/// where `Some(cmax)` means acyclic feasible and `None` means DAG cycle rejected.
pub(crate) fn checked_probe_move(
    evaluator: &mut FastGraphEvaluator,
    mv: &crate::neighborhoods::GraphMove,
    counter: &mut EvalCounter,
    counters: &mut SearchCounters,
) -> Result<Option<Option<u64>>, JsspError> {
    if !counter.try_consume_search() {
        return Ok(None);
    }
    counters.evaluated_moves += 1;
    match evaluator.probe_move_makespan(mv)? {
        Some(cmax) => {
            counters.feasible_moves += 1;
            Ok(Some(Some(cmax)))
        }
        None => {
            counters.rejected_cyclic_moves += 1;
            Ok(Some(None))
        }
    }
}

/// Generates neighborhood `neighborhood` on `state` and accumulates structural counters.
pub(crate) fn generate_and_count(
    evaluator: &FastGraphEvaluator,
    state: &EvaluatedGraphState,
    neighborhood: NeighborhoodId,
    counters: &mut SearchCounters,
) -> crate::neighborhoods::GeneratedNeighborhood {
    let gen_n = generate_neighborhood(neighborhood, evaluator, state);
    counters.generated_moves += gen_n.raw_generated_count;
    counters.duplicate_pruned_moves += gen_n.duplicate_pruned_count;
    counters.infeasible_precond_moves += gen_n.infeasible_precond_count;
    counters.clipped_moves += gen_n.clipped_boundary_count;
    gen_n
}

/// Computes exact stepwise area under best-so-far makespan curve normalized to `[0, B_eval]`.
#[must_use]
pub fn compute_stepwise_anytime_auc(milestones: &[(usize, u64)], eval_budget: usize) -> f64 {
    if milestones.is_empty() || eval_budget == 0 {
        return 0.0;
    }
    let mut area = 0.0_f64;
    let mut prev_e = milestones[0].0.min(eval_budget);
    let mut prev_c = milestones[0].1 as f64;
    // Account any initial flat prefix [0..prev_e] cleanly at prev_c
    if prev_e > 0 {
        area += (prev_e as f64) * prev_c;
    }
    for &(e_raw, c_raw) in milestones.iter().skip(1) {
        let e = e_raw.min(eval_budget);
        if e > prev_e {
            area += ((e - prev_e) as f64) * prev_c;
            prev_e = e;
        }
        prev_c = prev_c.min(c_raw as f64);
    }
    if eval_budget > prev_e {
        area += ((eval_budget - prev_e) as f64) * prev_c;
    }
    area / (eval_budget as f64)
}
