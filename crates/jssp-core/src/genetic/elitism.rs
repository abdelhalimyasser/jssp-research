//! Deterministic rank ordering and elitist preservation (`D07-A`).

/// Returns population indices sorted ascending by `(makespans[idx], idx)` for deterministic elitism.
#[must_use]
pub fn ranked_population_indices(makespans: &[u64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..makespans.len()).collect();
    indices.sort_by_key(|&idx| (makespans[idx], idx));
    indices
}
