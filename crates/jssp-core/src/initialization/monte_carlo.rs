//! Monte Carlo + Dynamic MWR (`A03`, `A15`, `A16`) OBR population generator.

use crate::error::JsspError;
use crate::initialization::qmc::discrete_mapping::{map_unit_coordinates_to_obr, u01_from_u64};
use crate::model::JsspInstance;
use crate::representation::ObrChromosome;
use crate::rng::NamedRngStream;

/// Generates `pop_size` MWR-biased Monte Carlo OBR chromosomes using the `init` stream
/// and exponent `alpha` (`1.0`, `0.5`, or `2.0`).
///
/// # Errors
/// Returns [`JsspError`] if `alpha` is invalid or mapping fails.
pub fn generate_monte_carlo_mwr_population(
    instance: &JsspInstance,
    pop_size: usize,
    alpha: f64,
    init_rng: &mut NamedRngStream,
) -> Result<(Vec<ObrChromosome>, Vec<Vec<f64>>), JsspError> {
    let dim = instance.num_operations();
    let mut chromosomes = Vec::with_capacity(pop_size);
    let mut coords = Vec::with_capacity(pop_size);
    for _ in 0..pop_size {
        let point: Vec<f64> = (0..dim)
            .map(|_| u01_from_u64(init_rng.next_u64()))
            .collect();
        chromosomes.push(map_unit_coordinates_to_obr(instance, &point, alpha)?);
        coords.push(point);
    }
    Ok((chromosomes, coords))
}
