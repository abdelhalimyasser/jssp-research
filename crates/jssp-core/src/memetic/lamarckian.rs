//! Lamarckian write-back policy (`C02`) & Genotype/Phenotype evaluation helper (`D07`).

use crate::error::JsspError;
use crate::model::{JsspInstance, Schedule};
use crate::representation::ObrChromosome;

/// Result of applying bounded local search to one decoded offspring chromosome.
#[derive(Debug, Clone)]
pub struct MemeticEvalOutcome {
    pub original_genotype: ObrChromosome,
    pub improved_genotype: ObrChromosome,
    pub effective_genotype: ObrChromosome,
    pub raw_makespan: u64,
    pub post_ls_makespan: u64,
    pub ls_evals_consumed: usize,
    pub best_schedule: Schedule,
}

impl MemeticEvalOutcome {
    /// Positive makespan reduction achieved by local search (`max(0, raw - post_ls)`).
    #[must_use]
    pub const fn ls_delta(&self) -> u64 {
        self.raw_makespan.saturating_sub(self.post_ls_makespan)
    }
}

/// Applies Lamarckian write-back: sets inherited genotype to the post-LS re-encoded OBR
/// iff local search found a schedule strictly better than or equal to initial decode.
pub fn apply_lamarckian_writeback(
    instance: &JsspInstance,
    outcome: &mut MemeticEvalOutcome,
) -> Result<(), JsspError> {
    if outcome.post_ls_makespan < outcome.raw_makespan {
        outcome.improved_genotype.validate(instance)?;
        outcome.effective_genotype = outcome.improved_genotype.clone();
    } else {
        outcome.effective_genotype = outcome.original_genotype.clone();
    }
    Ok(())
}
