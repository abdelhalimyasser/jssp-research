//! Halton (1960) prime-base radical-inverse QMC point generator (`D02-B`).

use crate::error::JsspError;
use crate::initialization::qmc::scrambling::HaltonDigitPermutations;
use crate::rng::NamedRngStream;

/// Generates the first `count` prime numbers via Sieve of Eratosthenes.
#[must_use]
pub fn first_primes(count: usize) -> Vec<usize> {
    if count == 0 {
        return Vec::new();
    }
    let bound = if count < 6 {
        15
    } else {
        let df = count as f64;
        let ln_d = df.ln();
        let ln_ln_d = ln_d.ln().max(1.0);
        (df * (ln_d + ln_ln_d + 2.0)).ceil() as usize
    };
    let mut is_prime = vec![true; bound + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    let limit = (bound as f64).sqrt() as usize;
    for p in 2..=limit {
        if is_prime[p] {
            let mut mult = p * p;
            while mult <= bound {
                is_prime[mult] = false;
                mult += p;
            }
        }
    }
    let mut primes = Vec::with_capacity(count);
    for (num, &prime) in is_prime.iter().enumerate() {
        if prime {
            primes.push(num);
            if primes.len() == count {
                break;
            }
        }
    }
    primes
}

/// Evaluates the unscrambled Halton radical inverse of `index` in prime base `base`.
#[inline]
#[must_use]
pub fn radical_inverse(mut index: usize, base: usize) -> f64 {
    let inv_base = 1.0 / (base as f64);
    let mut factor = inv_base;
    let mut result = 0.0_f64;
    while index > 0 {
        let digit = index % base;
        result += (digit as f64) * factor;
        index /= base;
        factor *= inv_base;
    }
    result
}

/// Generates `pop_size` points in `[0, 1)^dimension` from the 1-based contiguous Halton
/// prefix `1..=pop_size` (`scrambled = false` for A04/A08/A17/A18; `true` for A05/A09/A12-14/A19-20).
///
/// # Errors
/// Returns [`JsspError`] if scrambling sampling fails when `scrambled == true`.
pub fn generate_halton_points(
    dimension: usize,
    pop_size: usize,
    scrambled: bool,
    scramble_rng: Option<&mut NamedRngStream>,
) -> Result<Vec<Vec<f64>>, JsspError> {
    if pop_size == 0 || dimension == 0 {
        return Ok(vec![vec![0.0; dimension]; pop_size]);
    }
    let primes = first_primes(dimension);
    if scrambled {
        let rng = scramble_rng.ok_or_else(|| JsspError::InvalidInitializerConfiguration {
            detail: "Scrambled Halton requires a qmc_scramble RNG stream".to_owned(),
        })?;
        let perms = HaltonDigitPermutations::sample(&primes, pop_size, rng)?;
        let mut points = Vec::with_capacity(pop_size);
        for i in 1..=pop_size {
            let mut row = Vec::with_capacity(dimension);
            for (s, &base) in primes.iter().enumerate() {
                row.push(perms.radical_inverse(s, base, i));
            }
            points.push(row);
        }
        Ok(points)
    } else {
        let mut points = Vec::with_capacity(pop_size);
        for i in 1..=pop_size {
            let mut row = Vec::with_capacity(dimension);
            for &base in &primes {
                row.push(radical_inverse(i, base));
            }
            points.push(row);
        }
        Ok(points)
    }
}
