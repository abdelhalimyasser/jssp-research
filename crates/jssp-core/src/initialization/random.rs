//! Uniform Random (`A01`, alpha = 0.0) OBR population generator.

use crate::error::JsspError;
use crate::initialization::monte_carlo::generate_monte_carlo_mwr_population;
use crate::model::JsspInstance;
use crate::representation::ObrChromosome;
use crate::rng::NamedRngStream;

/// Generates `pop_size` uniformly random OBR chromosomes via the sequential discrete mapping
/// at `alpha = 0.0` using `init_rng`, returning both chromosomes and raw coordinate vectors.
///
/// # Errors
/// Returns [`JsspError`] if mapping fails.
pub fn generate_uniform_random_population(
    instance: &JsspInstance,
    pop_size: usize,
    init_rng: &mut NamedRngStream,
) -> Result<(Vec<ObrChromosome>, Vec<Vec<f64>>), JsspError> {
    generate_monte_carlo_mwr_population(instance, pop_size, 0.0, init_rng)
}
