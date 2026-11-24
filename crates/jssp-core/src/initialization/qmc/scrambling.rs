//! Scrambling primitives for Halton radical-inverse digits and Matousek Sobol scrambling (`D02-B`).

use crate::error::JsspError;
use crate::rng::NamedRngStream;

/// Pre-generated per-(dimension, digit_level) random digit permutations for Scrambled Halton.
///
/// Each base-`b` permutation maps digits `0..=max_d` encountered across indices `1..=max_index`
/// using an exact `O(max_d)` partial Fisher–Yates prefix equivalent in distribution to a
/// uniformly random permutation in `S_b`.
#[derive(Debug, Clone)]
pub struct HaltonDigitPermutations {
    /// For coordinate `s`, stores one permutation prefix table per digit position `k`.
    per_dim_levels: Vec<Vec<Vec<usize>>>,
}

impl HaltonDigitPermutations {
    /// Samples independent per-dimension, per-digit-level permutations from `qmc_scramble`.
    ///
    /// # Errors
    /// Returns [`JsspError`] if `max_index == 0` or RNG sampling fails.
    pub fn sample(
        primes: &[usize],
        max_index: usize,
        rng: &mut NamedRngStream,
    ) -> Result<Self, JsspError> {
        if max_index == 0 {
            return Err(JsspError::InvalidInitializerConfiguration {
                detail: "Halton scramble max_index must be >= 1".to_owned(),
            });
        }
        let max_prime = primes.iter().copied().max().unwrap_or(2);
        let mut scratch: Vec<usize> = (0..max_prime).collect();
        let mut touched: Vec<usize> = Vec::with_capacity((max_index + 1).min(max_prime));
        let mut per_dim_levels = Vec::with_capacity(primes.len());

        for &base in primes {
            let mut levels = Vec::new();
            let mut pow_k: usize = 1;
            loop {
                let next_pow = pow_k.saturating_mul(base);
                let max_d = if next_pow <= max_index && next_pow > pow_k {
                    base - 1
                } else {
                    (max_index / pow_k).min(base - 1)
                };

                touched.clear();
                let mut map_prefix = Vec::with_capacity(max_d + 1);
                for x in 0..=max_d {
                    let y = x + rng.gen_range_usize(base - x)?;
                    scratch.swap(x, y);
                    touched.push(x);
                    if y > max_d {
                        touched.push(y);
                    }
                    map_prefix.push(scratch[x]);
                }
                for &idx in &touched {
                    scratch[idx] = idx;
                }
                levels.push(map_prefix);

                if next_pow > max_index || next_pow <= pow_k {
                    break;
                }
                pow_k = next_pow;
            }
            per_dim_levels.push(levels);
        }

        Ok(Self { per_dim_levels })
    }

    /// Evaluates the scrambled Halton coordinate for `(dim, index)` with 1-based `index <= max_index`.
    #[inline]
    #[must_use]
    pub fn radical_inverse(&self, dim: usize, base: usize, mut index: usize) -> f64 {
        let levels = &self.per_dim_levels[dim];
        let inv_base = 1.0 / (base as f64);
        let mut weight = inv_base;
        let mut acc = 0.0_f64;

        for perm_prefix in levels {
            let digit = index % base;
            index /= base;
            let mapped = perm_prefix[digit];
            acc += (mapped as f64) * weight;
            weight *= inv_base;
        }
        acc
    }
}

/// Matousek (1998) invertible lower-triangular linear scrambling matrix + 32-bit digital shift.
///
/// Row ordering follows dyadic precision from MSB (`bit 31`, row 0) down to LSB (`bit 0`, row 31)
/// so that nested dyadic cells of every `(t, m, s)` Sobol net remain exact stratified permutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatousekMatrix32 {
    /// For bit position `bit_pos` in `0..32` (`31` = MSB .. `0` = LSB), stores the XOR mask over
    /// input bits `0..32` whose parity produces output bit `bit_pos` prior to digital shift.
    row_masks_by_bit: [u32; 32],
    /// Random 32-bit XOR digital shift vector applied after linear matrix multiplication.
    digital_shift: u32,
}

impl MatousekMatrix32 {
    /// Draws a uniformly random lower-triangular (MSB-to-LSB) binary matrix and 32-bit shift
    /// from `rng` (consuming one 64-bit word per two rows / compactly from ChaCha20).
    pub fn sample(rng: &mut NamedRngStream) -> Self {
        let mut row_masks_by_bit = [0u32; 32];
        // Build rows from r = 0 (MSB = bit 31) to r = 31 (LSB = bit 0).
        // Row r depends on bit (31 - r) (diagonal 1) and arbitrary bits strictly more significant
        // than (31 - r), i.e., bits (31 - q) for 0 <= q < r.
        for r in 0..32 {
            let bit_pos = 31 - r;
            let diag_bit = 1u32 << bit_pos;
            let strict_coarser_mask = if r == 0 {
                0u32
            } else {
                !((1u32 << (bit_pos + 1)) - 1)
            };
            let random_bits = rng.next_u64() as u32;
            row_masks_by_bit[bit_pos] = (random_bits & strict_coarser_mask) | diag_bit;
        }
        let digital_shift = rng.next_u64() as u32;
        Self {
            row_masks_by_bit,
            digital_shift,
        }
    }

    /// Applies `(M_s * y) XOR c_s` over `GF(2)^32`.
    #[inline]
    #[must_use]
    pub fn apply(&self, y: u32) -> u32 {
        let mut out = self.digital_shift;
        for (bit_pos, &mask) in self.row_masks_by_bit.iter().enumerate() {
            let parity = (y & mask).count_ones() & 1;
            out ^= parity << bit_pos;
        }
        out
    }
}
