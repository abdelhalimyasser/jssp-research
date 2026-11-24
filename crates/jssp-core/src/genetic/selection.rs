//! Deterministic tournament parent selection for evolutionary JSSP baselines (`D07-A`).

use crate::error::JsspError;
use crate::rng::NamedRngStream;

/// Binary (or general `k >= 1`) tournament parent selection on minimizing makespan.
///
/// Exact makespan ties break deterministically toward the **smaller population index**.
///
/// # Errors
/// Returns [`JsspError::InvalidGeneticConfiguration`] if `makespans` is empty or `tournament_size == 0`.
pub fn tournament_select(
    makespans: &[u64],
    tournament_size: usize,
    rng: &mut NamedRngStream,
) -> Result<usize, JsspError> {
    if makespans.is_empty() {
        return Err(JsspError::InvalidGeneticConfiguration {
            detail: "cannot run tournament selection on an empty population".to_owned(),
        });
    }
    if tournament_size == 0 {
        return Err(JsspError::InvalidGeneticConfiguration {
            detail: "tournament_size must be >= 1".to_owned(),
        });
    }
    let pop_len = makespans.len();
    let mut best_idx = rng.gen_range_usize(pop_len)?;
    let mut best_fit = makespans[best_idx];

    for _ in 1..tournament_size {
        let cand_idx = rng.gen_range_usize(pop_len)?;
        let cand_fit = makespans[cand_idx];
        if cand_fit < best_fit || (cand_fit == best_fit && cand_idx < best_idx) {
            best_idx = cand_idx;
            best_fit = cand_fit;
        }
    }

    Ok(best_idx)
}
