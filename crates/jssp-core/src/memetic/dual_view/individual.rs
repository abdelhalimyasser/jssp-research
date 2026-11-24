//! Model A paired Dual-View individual representation, shared-scale normalization,
//! pair-level selection tournament, and elitist ranking (`D06`).

use crate::error::JsspError;
use crate::representation::ObrChromosome;
use crate::rng::NamedRngStream;

/// Identity-paired Dual-View individual `I_i = (id, G^L, G^B)` (`D06` §2).
#[derive(Debug, Clone)]
pub struct DualViewIndividual {
    pub id: usize,
    pub lam_genotype: ObrChromosome,
    pub lam_fitness: u64,
    pub lam_ls_delta: u64,
    pub lam_ls_evals: usize,
    pub bald_genotype: ObrChromosome,
    pub bald_fitness: u64,
    pub bald_ls_delta: u64,
    pub bald_ls_evals: usize,
}

impl DualViewIndividual {
    /// Minimum phenotype makespan across both views (`min(f^L, f^B)`).
    #[must_use]
    pub const fn min_fitness(&self) -> u64 {
        if self.lam_fitness <= self.bald_fitness {
            self.lam_fitness
        } else {
            self.bald_fitness
        }
    }
}

/// Normalized Model A view scores `(S_i^L, S_i^B)` on commensurate population-wide bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedPairScores {
    pub s_lam: f64,
    pub s_bald: f64,
    pub pair_score: f64,
}

/// Computes shared-scale min-max normalized view scores `S_i^L, S_i^B in [0, 1]` and weighted
/// pair selection score `S_i(t) = p_t * S_i^L + (1 - p_t) * S_i^B` (`D06` §4.1).
#[must_use]
pub fn normalize_dual_view_scores(
    pop: &[DualViewIndividual],
    p_gate: f64,
) -> (Vec<NormalizedPairScores>, f64, f64) {
    if pop.is_empty() {
        return (Vec::new(), 1.0, 1.0);
    }

    let mut f_min = u64::MAX;
    let mut f_max = 0u64;
    for ind in pop {
        f_min = f_min.min(ind.min_fitness());
        f_max = f_max.max(ind.lam_fitness.max(ind.bald_fitness));
    }

    let span = f_max.saturating_sub(f_min);
    let mut out = Vec::with_capacity(pop.len());
    let mut sum_s_lam = 0.0_f64;
    let mut sum_s_bald = 0.0_f64;

    for ind in pop {
        let (s_lam, s_bald) = if span == 0 {
            (1.0, 1.0)
        } else {
            let denom = span as f64;
            (
                ((f_max - ind.lam_fitness) as f64) / denom,
                ((f_max - ind.bald_fitness) as f64) / denom,
            )
        };
        let pair_score = p_gate * s_lam + (1.0 - p_gate) * s_bald;
        sum_s_lam += s_lam;
        sum_s_bald += s_bald;
        out.push(NormalizedPairScores {
            s_lam,
            s_bald,
            pair_score,
        });
    }

    let n = pop.len() as f64;
    (out, sum_s_lam / n, sum_s_bald / n)
}

/// Returns pair indices ordered for elitism: smaller `min(f^L, f^B)`, tie-break larger `S_i`,
/// then smaller `individual.id` (`D06` §4.1).
#[must_use]
pub fn ranked_dual_view_indices(
    pop: &[DualViewIndividual],
    scores: &[NormalizedPairScores],
) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..pop.len()).collect();
    indices.sort_by(|&a, &b| {
        pop[a]
            .min_fitness()
            .cmp(&pop[b].min_fitness())
            .then_with(|| scores[b].pair_score.total_cmp(&scores[a].pair_score))
            .then_with(|| pop[a].id.cmp(&pop[b].id))
    });
    indices
}

/// Deterministic binary tournament over paired Dual-View individuals using Model A pair score
/// `S_i(t)` (higher wins), tie-broken by smaller `min(f^L, f^B)`, then smaller `id` (`D06` §4.1).
///
/// # Errors
/// Returns [`JsspError::InvalidGeneticConfiguration`] if `pop` is empty or `tournament_size == 0`.
pub fn select_dual_view_parent(
    pop: &[DualViewIndividual],
    scores: &[NormalizedPairScores],
    tournament_size: usize,
    rng: &mut NamedRngStream,
) -> Result<usize, JsspError> {
    if pop.is_empty() || pop.len() != scores.len() {
        return Err(JsspError::InvalidGeneticConfiguration {
            detail: "dual-view selection requires non-empty aligned population & scores".to_owned(),
        });
    }
    if tournament_size == 0 {
        return Err(JsspError::InvalidGeneticConfiguration {
            detail: "tournament_size must be >= 1".to_owned(),
        });
    }

    let n = pop.len();
    let mut best_idx = rng.gen_range_usize(n)?;
    for _ in 1..tournament_size {
        let cand_idx = rng.gen_range_usize(n)?;
        let cmp = scores[cand_idx]
            .pair_score
            .total_cmp(&scores[best_idx].pair_score)
            .then_with(|| {
                pop[best_idx]
                    .min_fitness()
                    .cmp(&pop[cand_idx].min_fitness())
            })
            .then_with(|| pop[best_idx].id.cmp(&pop[cand_idx].id));
        if cmp.is_gt() {
            best_idx = cand_idx;
        }
    }
    Ok(best_idx)
}
