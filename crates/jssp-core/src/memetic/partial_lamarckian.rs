//! Partial Lamarckian GA (`C04`) deterministic even-rank 50% write-back policy (`D07`).

use crate::error::JsspError;
use crate::model::JsspInstance;
use crate::representation::ObrChromosome;

/// Applies deterministic `C04` 50% write-back: sorts population indices by `(fitness, id)`;
/// **even sorted ranks (`0, 2, 4, ...`)** receive Lamarckian genotype write-back while
/// **odd sorted ranks (`1, 3, 5, ...`)** retain their original pre-LS Baldwinian genotype.
///
/// # Errors
/// Returns [`JsspError`] if `genotypes`, `improved_genotypes`, `fitnesses` lengths differ or fail validation.
pub fn apply_partial_lamarckian_even_ranks(
    instance: &JsspInstance,
    genotypes: &mut [ObrChromosome],
    improved_genotypes: &[Option<ObrChromosome>],
    fitnesses: &[u64],
) -> Result<usize, JsspError> {
    let n = genotypes.len();
    if improved_genotypes.len() != n || fitnesses.len() != n {
        return Err(JsspError::InvalidGeneticConfiguration {
            detail: "PLGA population vectors must match in length".to_owned(),
        });
    }

    let mut ranked: Vec<usize> = (0..n).collect();
    ranked.sort_by_key(|&idx| (fitnesses[idx], idx));

    let mut writebacks = 0usize;
    for (rank, &idx) in ranked.iter().enumerate() {
        if rank % 2 == 0
            && let Some(imp) = &improved_genotypes[idx]
        {
            imp.validate(instance)?;
            genotypes[idx] = imp.clone();
            writebacks += 1;
        }
    }
    Ok(writebacks)
}
