//! Phase A (`A01`–`A24`) OBR initial population generation module (`D02`).

pub mod monte_carlo;
pub mod mwr;
pub mod qmc;
pub mod random;

use std::time::Instant;

use crate::error::JsspError;
use crate::model::JsspInstance;
use crate::representation::ObrChromosome;
use crate::rng::RunRngStreams;

use self::monte_carlo::generate_monte_carlo_mwr_population;
use self::mwr::generate_deterministic_mwr_population;
use self::qmc::{generate_halton_points, generate_sobol_points, map_unit_coordinates_to_obr};
use self::random::generate_uniform_random_population;

/// Canonical identifiers for all 24 Phase A initial population configurations (`A01`–`A24`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InitializerId {
    A01,
    A02,
    A03,
    A04,
    A05,
    A06,
    A07,
    A08,
    A09,
    A10,
    A11,
    A12,
    A13,
    A14,
    A15,
    A16,
    A17,
    A18,
    A19,
    A20,
    A21,
    A22,
    A23,
    A24,
}

impl InitializerId {
    /// Complete ordered list of all 24 Phase A initializers (`A01`–`A24`).
    pub const ALL: [Self; 24] = [
        Self::A01,
        Self::A02,
        Self::A03,
        Self::A04,
        Self::A05,
        Self::A06,
        Self::A07,
        Self::A08,
        Self::A09,
        Self::A10,
        Self::A11,
        Self::A12,
        Self::A13,
        Self::A14,
        Self::A15,
        Self::A16,
        Self::A17,
        Self::A18,
        Self::A19,
        Self::A20,
        Self::A21,
        Self::A22,
        Self::A23,
        Self::A24,
    ];

    /// Parses canonical uppercase identifier `"A01"`–`"A24"`.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidInitializerConfiguration`] if unknown.
    pub fn from_code(code: &str) -> Result<Self, JsspError> {
        Self::ALL
            .iter()
            .copied()
            .find(|id| id.code() == code)
            .ok_or_else(|| JsspError::InvalidInitializerConfiguration {
                detail: format!("unknown Phase A initializer ID '{code}', expected A01..=A24"),
            })
    }

    /// Canonical identifier string (`"A01"`–`"A24"`).
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::A01 => "A01",
            Self::A02 => "A02",
            Self::A03 => "A03",
            Self::A04 => "A04",
            Self::A05 => "A05",
            Self::A06 => "A06",
            Self::A07 => "A07",
            Self::A08 => "A08",
            Self::A09 => "A09",
            Self::A10 => "A10",
            Self::A11 => "A11",
            Self::A12 => "A12",
            Self::A13 => "A13",
            Self::A14 => "A14",
            Self::A15 => "A15",
            Self::A16 => "A16",
            Self::A17 => "A17",
            Self::A18 => "A18",
            Self::A19 => "A19",
            Self::A20 => "A20",
            Self::A21 => "A21",
            Self::A22 => "A22",
            Self::A23 => "A23",
            Self::A24 => "A24",
        }
    }

    /// Human-readable initializer method name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::A01 => "Uniform Random (alpha=0.0)",
            Self::A02 => "Deterministic MWR",
            Self::A03 => "Monte Carlo + MWR (alpha=1.0)",
            Self::A04 => "Unscrambled Halton (alpha=0.0)",
            Self::A05 => "Scrambled Halton (alpha=0.0)",
            Self::A06 => "Unscrambled Sobol (alpha=0.0)",
            Self::A07 => "Scrambled Sobol (alpha=0.0)",
            Self::A08 => "Halton + MWR (alpha=1.0)",
            Self::A09 => "Scrambled Halton + MWR (alpha=1.0)",
            Self::A10 => "Sobol + MWR (alpha=1.0)",
            Self::A11 => "Scrambled Sobol + MWR (alpha=1.0)",
            Self::A12 => "Mixed Scrambled Halton/Sobol (50:50)",
            Self::A13 => "Mixed Scrambled Halton/Sobol (25:75)",
            Self::A14 => "Mixed Scrambled Halton/Sobol (75:25)",
            Self::A15 => "Monte Carlo + MWR (alpha=0.5)",
            Self::A16 => "Monte Carlo + MWR (alpha=2.0)",
            Self::A17 => "Halton + MWR (alpha=0.5)",
            Self::A18 => "Halton + MWR (alpha=2.0)",
            Self::A19 => "Scrambled Halton + MWR (alpha=0.5)",
            Self::A20 => "Scrambled Halton + MWR (alpha=2.0)",
            Self::A21 => "Sobol + MWR (alpha=0.5)",
            Self::A22 => "Sobol + MWR (alpha=2.0)",
            Self::A23 => "Scrambled Sobol + MWR (alpha=0.5)",
            Self::A24 => "Scrambled Sobol + MWR (alpha=2.0)",
        }
    }

    /// MWR weighting exponent `alpha` (`None` for deterministic `A02`).
    #[must_use]
    pub const fn alpha(self) -> Option<f64> {
        match self {
            Self::A02 => None,
            Self::A01
            | Self::A04
            | Self::A05
            | Self::A06
            | Self::A07
            | Self::A12
            | Self::A13
            | Self::A14 => Some(0.0),
            Self::A15 | Self::A17 | Self::A19 | Self::A21 | Self::A23 => Some(0.5),
            Self::A03 | Self::A08 | Self::A09 | Self::A10 | Self::A11 => Some(1.0),
            Self::A16 | Self::A18 | Self::A20 | Self::A22 | Self::A24 => Some(2.0),
        }
    }

    /// Returns `true` iff initial population construction consumes zero RNG draws.
    #[must_use]
    pub const fn is_deterministic(self) -> bool {
        matches!(
            self,
            Self::A02
                | Self::A04
                | Self::A06
                | Self::A08
                | Self::A10
                | Self::A17
                | Self::A18
                | Self::A21
                | Self::A22
        )
    }

    /// Returns `true` iff this configuration invokes Joe–Kuo Sobol generation.
    #[must_use]
    pub const fn uses_sobol(self) -> bool {
        matches!(
            self,
            Self::A06
                | Self::A07
                | Self::A10
                | Self::A11
                | Self::A12
                | Self::A13
                | Self::A14
                | Self::A21
                | Self::A22
                | Self::A23
                | Self::A24
        )
    }

    /// Computes `(halton_count, sobol_count)` for mixed configurations `A12`–`A14`.
    #[must_use]
    pub fn mix_counts(self, pop_size: usize) -> Option<(usize, usize)> {
        let ratio_h = match self {
            Self::A12 => 0.50,
            Self::A13 => 0.25,
            Self::A14 => 0.75,
            _ => return None,
        };
        let n_h = ((pop_size as f64) * ratio_h).round() as usize;
        let n_h_clamped = n_h.min(pop_size);
        Some((n_h_clamped, pop_size - n_h_clamped))
    }

    /// Returns `true` iff the configuration's Sobol subset realizes a strict 0-based
    /// dyadic power-of-two digital net (`N_S.is_power_of_two()` and zero-origin included).
    #[must_use]
    pub fn has_dyadic_sobol_net_balance(self, pop_size: usize) -> bool {
        match self {
            Self::A07 | Self::A11 | Self::A23 | Self::A24 => pop_size.is_power_of_two(),
            Self::A12 | Self::A13 | Self::A14 => {
                let (_, n_s) = self.mix_counts(pop_size).unwrap_or((0, 0));
                n_s > 0 && n_s.is_power_of_two()
            }
            _ => false,
        }
    }
}

/// Generated OBR initial population together with optional first-dimension `[0, 1)` samples
/// for empirical 1D discrepancy checks and generation wall-clock duration.
#[derive(Debug, Clone)]
pub struct GeneratedPopulation {
    pub initializer: InitializerId,
    pub chromosomes: Vec<ObrChromosome>,
    pub first_dim_coords: Option<Vec<f64>>,
    pub generation_elapsed_ns: u64,
}

/// Generates an OBR initial population of size `pop_size` for `initializer` using isolated
/// `streams` (`init` for PRNG/MC-MWR; `qmc_scramble` for Halton/Sobol scrambling).
///
/// # Errors
/// Returns [`JsspError`] if `pop_size == 0`, dimension exceeds `21,201` for Sobol configs,
/// or chromosome construction fails.
pub fn generate_initial_population(
    initializer: InitializerId,
    instance: &JsspInstance,
    pop_size: usize,
    streams: &mut RunRngStreams,
) -> Result<GeneratedPopulation, JsspError> {
    if pop_size == 0 {
        return Err(JsspError::InvalidInitializerConfiguration {
            detail: "population size must be >= 1".to_owned(),
        });
    }

    let started = Instant::now();
    let dim = instance.num_operations();

    let (chromosomes, first_dim_coords) = match initializer {
        InitializerId::A01 => {
            let (chrs, pts) =
                generate_uniform_random_population(instance, pop_size, &mut streams.init)?;
            let first = pts.iter().filter_map(|p| p.first().copied()).collect();
            (chrs, Some(first))
        }
        InitializerId::A02 => {
            let chrs = generate_deterministic_mwr_population(instance, pop_size)?;
            (chrs, None)
        }
        InitializerId::A03 | InitializerId::A15 | InitializerId::A16 => {
            let alpha = initializer.alpha().unwrap_or(1.0);
            let (chrs, pts) =
                generate_monte_carlo_mwr_population(instance, pop_size, alpha, &mut streams.init)?;
            let first = pts.iter().filter_map(|p| p.first().copied()).collect();
            (chrs, Some(first))
        }
        InitializerId::A04 | InitializerId::A08 | InitializerId::A17 | InitializerId::A18 => {
            let alpha = initializer.alpha().unwrap_or(0.0);
            let pts = generate_halton_points(dim, pop_size, false, None)?;
            let chrs = map_point_batch(instance, &pts, alpha)?;
            let first = pts.iter().filter_map(|p| p.first().copied()).collect();
            (chrs, Some(first))
        }
        InitializerId::A05 | InitializerId::A09 | InitializerId::A19 | InitializerId::A20 => {
            let alpha = initializer.alpha().unwrap_or(0.0);
            let pts = generate_halton_points(dim, pop_size, true, Some(&mut streams.qmc_scramble))?;
            let chrs = map_point_batch(instance, &pts, alpha)?;
            let first = pts.iter().filter_map(|p| p.first().copied()).collect();
            (chrs, Some(first))
        }
        InitializerId::A06 | InitializerId::A10 | InitializerId::A21 | InitializerId::A22 => {
            let alpha = initializer.alpha().unwrap_or(0.0);
            let pts = generate_sobol_points(dim, pop_size, false, 1, None)?;
            let chrs = map_point_batch(instance, &pts, alpha)?;
            let first = pts.iter().filter_map(|p| p.first().copied()).collect();
            (chrs, Some(first))
        }
        InitializerId::A07 | InitializerId::A11 | InitializerId::A23 | InitializerId::A24 => {
            let alpha = initializer.alpha().unwrap_or(0.0);
            let pts =
                generate_sobol_points(dim, pop_size, true, 0, Some(&mut streams.qmc_scramble))?;
            let chrs = map_point_batch(instance, &pts, alpha)?;
            let first = pts.iter().filter_map(|p| p.first().copied()).collect();
            (chrs, Some(first))
        }
        InitializerId::A12 | InitializerId::A13 | InitializerId::A14 => {
            let (n_h, n_s) = initializer.mix_counts(pop_size).unwrap_or((pop_size, 0));
            let halton_pts = if n_h > 0 {
                generate_halton_points(dim, n_h, true, Some(&mut streams.qmc_scramble))?
            } else {
                Vec::new()
            };
            let sobol_pts = if n_s > 0 {
                generate_sobol_points(dim, n_s, true, 0, Some(&mut streams.qmc_scramble))?
            } else {
                Vec::new()
            };
            let mut combined_pts = Vec::with_capacity(pop_size);
            combined_pts.extend(halton_pts);
            combined_pts.extend(sobol_pts);
            let chrs = map_point_batch(instance, &combined_pts, 0.0)?;
            let first = combined_pts
                .iter()
                .filter_map(|p| p.first().copied())
                .collect();
            (chrs, Some(first))
        }
    };

    let generation_elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    Ok(GeneratedPopulation {
        initializer,
        chromosomes,
        first_dim_coords,
        generation_elapsed_ns,
    })
}

fn map_point_batch(
    instance: &JsspInstance,
    points: &[Vec<f64>],
    alpha: f64,
) -> Result<Vec<ObrChromosome>, JsspError> {
    let mut chromosomes = Vec::with_capacity(points.len());
    for point in points {
        chromosomes.push(map_unit_coordinates_to_obr(instance, point, alpha)?);
    }
    Ok(chromosomes)
}
