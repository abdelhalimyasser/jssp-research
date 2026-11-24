//! Quasi-Monte Carlo generators, scrambling, and sequential discrete OBR mapping (`D02`).

pub mod discrete_mapping;
pub mod halton;
pub mod scrambling;
pub mod sobol;

pub use discrete_mapping::{map_unit_coordinates_to_obr, u01_from_u64};
pub use halton::{first_primes, generate_halton_points, radical_inverse};
pub use scrambling::{HaltonDigitPermutations, MatousekMatrix32};
pub use sobol::{MAX_SOBOL_DIMENSION, generate_sobol_points};
