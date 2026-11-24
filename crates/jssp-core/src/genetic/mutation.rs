//! Distinct-job OBR Swap and Insertion mutation operators (`D07-A`).

use crate::error::JsspError;
use crate::initialization::qmc::u01_from_u64;
use crate::representation::ObrChromosome;
use crate::rng::NamedRngStream;

/// Applies OBR mutation to `chromosome` with probability `mutation_rate`.
///
/// Selects 50% **OBR Swap** or 50% **OBR Insertion** on two chromosome positions with
/// strictly distinct job IDs (`genes[i] != genes[j]`), preserving exact OBR multiplicity.
/// Returns `Ok(true)` iff a mutation was executed.
///
/// # Errors
/// Returns [`JsspError::InvalidGeneticConfiguration`] if `mutation_rate` is outside `[0, 1]`.
pub fn mutate_obr(
    chromosome: &mut ObrChromosome,
    mutation_rate: f64,
    rng: &mut NamedRngStream,
) -> Result<bool, JsspError> {
    if !(0.0..=1.0).contains(&mutation_rate) {
        return Err(JsspError::InvalidGeneticConfiguration {
            detail: format!("mutation_rate must lie in [0, 1], got {mutation_rate}"),
        });
    }

    let u = u01_from_u64(rng.next_u64());
    let genes = chromosome.genes_mut();
    let len = genes.len();
    if u >= mutation_rate || len < 2 {
        return Ok(false);
    }

    // Verify at least two distinct jobs exist so loop always terminates cleanly.
    let first_job = genes[0];
    if genes.iter().all(|&g| g == first_job) {
        return Ok(false);
    }

    let use_swap = (rng.next_u64() & 1) == 0;
    let i = rng.gen_range_usize(len)?;
    let mut j = rng.gen_range_usize(len)?;
    while genes[j] == genes[i] {
        j = rng.gen_range_usize(len)?;
    }

    if use_swap {
        genes.swap(i, j);
    } else if i < j {
        genes[i..=j].rotate_left(1);
    } else {
        genes[j..=i].rotate_right(1);
    }

    Ok(true)
}
