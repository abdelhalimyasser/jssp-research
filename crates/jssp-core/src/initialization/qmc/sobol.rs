//! Joe & Kuo (2008) `new-joe-kuo-6.21201` 32-bit Sobol QMC point generator (`D02-B`).

use std::sync::OnceLock;

use crate::error::JsspError;
use crate::initialization::qmc::scrambling::MatousekMatrix32;
use crate::rng::NamedRngStream;

/// Maximum coordinate dimension supported by the embedded Joe & Kuo (2008) table.
pub const MAX_SOBOL_DIMENSION: usize = 21_201;

const SCALE_2_NEG_32: f64 = 1.0 / ((1u64 << 32) as f64);

static JOE_KUO_RAW_BYTES: &[u8] = include_bytes!("joe_kuo_21201.bin");
static JOE_KUO_PARSED: OnceLock<Result<Vec<JoeKuoEntry>, String>> = OnceLock::new();

#[derive(Debug, Clone, Copy)]
struct JoeKuoEntry {
    poly: u32,
    s: u8,
    v_init: [u32; 18],
}

fn load_joe_kuo_table() -> Result<&'static [JoeKuoEntry], JsspError> {
    let res = JOE_KUO_PARSED.get_or_init(parse_joe_kuo_table);
    match res {
        Ok(entries) => Ok(entries.as_slice()),
        Err(detail) => Err(JsspError::InvalidInitializerConfiguration {
            detail: detail.clone(),
        }),
    }
}

fn parse_joe_kuo_table() -> Result<Vec<JoeKuoEntry>, String> {
    if JOE_KUO_RAW_BYTES.len() < 8 || &JOE_KUO_RAW_BYTES[..4] != b"JK6\x01" {
        return Err("corrupt embedded joe_kuo_21201.bin header".to_owned());
    }
    let count = u32::from_le_bytes([
        JOE_KUO_RAW_BYTES[4],
        JOE_KUO_RAW_BYTES[5],
        JOE_KUO_RAW_BYTES[6],
        JOE_KUO_RAW_BYTES[7],
    ]) as usize;
    if count != MAX_SOBOL_DIMENSION {
        return Err(format!(
            "expected {MAX_SOBOL_DIMENSION} Sobol dimensions in binary table, got {count}"
        ));
    }

    let mut entries = Vec::with_capacity(MAX_SOBOL_DIMENSION - 1);
    let mut offset = 8usize;
    for _dim_idx in 1..MAX_SOBOL_DIMENSION {
        if offset + 5 > JOE_KUO_RAW_BYTES.len() {
            return Err("truncated joe_kuo_21201.bin record header".to_owned());
        }
        let poly = u32::from_le_bytes([
            JOE_KUO_RAW_BYTES[offset],
            JOE_KUO_RAW_BYTES[offset + 1],
            JOE_KUO_RAW_BYTES[offset + 2],
            JOE_KUO_RAW_BYTES[offset + 3],
        ]);
        let s = JOE_KUO_RAW_BYTES[offset + 4];
        offset += 5;
        let s_usize = s as usize;
        if s_usize == 0 || s_usize > 18 || offset + 4 * s_usize > JOE_KUO_RAW_BYTES.len() {
            return Err("invalid degree or truncated vinit in joe_kuo_21201.bin".to_owned());
        }
        let mut v_init = [0u32; 18];
        for slot in v_init.iter_mut().take(s_usize) {
            *slot = u32::from_le_bytes([
                JOE_KUO_RAW_BYTES[offset],
                JOE_KUO_RAW_BYTES[offset + 1],
                JOE_KUO_RAW_BYTES[offset + 2],
                JOE_KUO_RAW_BYTES[offset + 3],
            ]);
            offset += 4;
        }
        entries.push(JoeKuoEntry { poly, s, v_init });
    }

    if offset != JOE_KUO_RAW_BYTES.len() {
        return Err("unexpected trailing bytes in joe_kuo_21201.bin".to_owned());
    }
    Ok(entries)
}

fn build_direction_numbers(dimension: usize) -> Result<Vec<[u32; 32]>, JsspError> {
    if dimension > MAX_SOBOL_DIMENSION {
        return Err(JsspError::UnsupportedQmcDimension {
            requested_dimension: dimension,
            max_supported_dimension: MAX_SOBOL_DIMENSION,
        });
    }
    let table = load_joe_kuo_table()?;
    let mut dirs = Vec::with_capacity(dimension);

    if dimension == 0 {
        return Ok(dirs);
    }

    // Dimension 0: classical van der Corput base 2
    let mut d0 = [0u32; 32];
    for (k, slot) in d0.iter_mut().enumerate() {
        *slot = 1u32 << (31 - k);
    }
    dirs.push(d0);

    for entry in table.iter().take(dimension - 1) {
        let mut v = [0u32; 32];
        let s = entry.s as usize;
        for k in 0..s {
            v[k] = entry.v_init[k] << (31 - k);
        }
        for k in s..32 {
            let mut val = v[k - s] ^ (v[k - s] >> s);
            for r in 1..s {
                if ((entry.poly >> (s - r)) & 1) == 1 {
                    val ^= v[k - r];
                }
            }
            v[k] = val;
        }
        dirs.push(v);
    }

    Ok(dirs)
}

/// Evaluates unscrambled Sobol integer states `y_{i, s}` for Gray code `i ^ (i >> 1)`
/// across `i in start_index..(start_index + pop_size)`, applying Matousek linear scrambling
/// when `scrambled == true`.
///
/// # Errors
/// Returns [`JsspError::UnsupportedQmcDimension`] if `dimension > 21_201`, or
/// [`JsspError::InvalidInitializerConfiguration`] for index overflow or missing RNG stream.
pub fn generate_sobol_points(
    dimension: usize,
    pop_size: usize,
    scrambled: bool,
    start_index: u32,
    scramble_rng: Option<&mut NamedRngStream>,
) -> Result<Vec<Vec<f64>>, JsspError> {
    if dimension > MAX_SOBOL_DIMENSION {
        return Err(JsspError::UnsupportedQmcDimension {
            requested_dimension: dimension,
            max_supported_dimension: MAX_SOBOL_DIMENSION,
        });
    }
    if pop_size == 0 || dimension == 0 {
        return Ok(vec![vec![0.0; dimension]; pop_size]);
    }
    let count_u32 =
        u32::try_from(pop_size).map_err(|_| JsspError::InvalidInitializerConfiguration {
            detail: format!("Sobol population size {pop_size} exceeds 32-bit sequence range"),
        })?;
    let end_exclusive = start_index.checked_add(count_u32).ok_or_else(|| {
        JsspError::InvalidInitializerConfiguration {
            detail: "Sobol index range overflows u32".to_owned(),
        }
    })?;

    let dirs = build_direction_numbers(dimension)?;
    let matrices = if scrambled {
        let rng = scramble_rng.ok_or_else(|| JsspError::InvalidInitializerConfiguration {
            detail: "Scrambled Sobol requires a qmc_scramble RNG stream".to_owned(),
        })?;
        let mut mats = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            mats.push(MatousekMatrix32::sample(rng));
        }
        Some(mats)
    } else {
        None
    };

    // Initialize Gray-code state cleanly at start_index, then step using Antonov-Saleev ctz rule.
    let start_gray = start_index ^ (start_index >> 1);
    let mut state = vec![0u32; dimension];
    for s in 0..dimension {
        let mut y = 0u32;
        let mut g = start_gray;
        while g != 0 {
            let bit = g.trailing_zeros() as usize;
            y ^= dirs[s][bit];
            g &= g - 1;
        }
        state[s] = y;
    }

    let mut points = Vec::with_capacity(pop_size);
    for idx in start_index..end_exclusive {
        let mut row = Vec::with_capacity(dimension);
        if let Some(ref mats) = matrices {
            for s in 0..dimension {
                let y_scrambled = mats[s].apply(state[s]);
                row.push((y_scrambled as f64) * SCALE_2_NEG_32);
            }
        } else {
            for &y in &state {
                row.push((y as f64) * SCALE_2_NEG_32);
            }
        }
        points.push(row);

        if idx + 1 < end_exclusive {
            let next_i = idx + 1;
            let bit = next_i.trailing_zeros() as usize;
            for s in 0..dimension {
                state[s] ^= dirs[s][bit];
            }
        }
    }

    Ok(points)
}
