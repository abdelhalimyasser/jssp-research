//! Baldwinian evaluation policy (`C03`, `D07`): uses locally improved phenotype makespan
//! for selection fitness while keeping the original genotype strictly unaltered.

use crate::error::JsspError;
use crate::model::JsspInstance;

use super::lamarckian::MemeticEvalOutcome;

/// Enforces Baldwinian genotype preservation: `effective_genotype == original_genotype`,
/// verified against `instance` while fitness uses `post_ls_makespan`.
///
/// # Errors
/// Returns [`JsspError`] if `original_genotype` fails validation.
pub fn apply_baldwinian_preservation(
    instance: &JsspInstance,
    outcome: &mut MemeticEvalOutcome,
) -> Result<(), JsspError> {
    outcome.original_genotype.validate(instance)?;
    outcome.effective_genotype = outcome.original_genotype.clone();
    Ok(())
}
