//! Sequential discrete inverse-CDF OBR mapping for coordinate vectors in `[0, 1)^L` (`D02-A`).

use crate::error::JsspError;
use crate::model::{JobId, JsspInstance};
use crate::representation::ObrChromosome;

const SCALE_2_NEG_53: f64 = 1.0 / ((1u64 << 53) as f64);

/// Converts a 64-bit unsigned uniform word into an IEEE-754 `f64` strictly in `[0.0, 1.0)`.
#[inline]
#[must_use]
pub fn u01_from_u64(raw: u64) -> f64 {
    ((raw >> 11) as f64) * SCALE_2_NEG_53
}

/// Maps a continuous coordinate vector `coords` in `[0, 1)^L` sequentially onto a
/// guaranteed valid OBR chromosome using dynamic remaining-work MWR weights `W_j(t)^alpha`.
///
/// # Errors
/// Returns [`JsspError::InvalidInitializerConfiguration`] if `coords.len() != L`,
/// `alpha` is negative/non-finite, or any coordinate is non-finite.
pub fn map_unit_coordinates_to_obr(
    instance: &JsspInstance,
    coords: &[f64],
    alpha: f64,
) -> Result<ObrChromosome, JsspError> {
    let expected_len = instance.num_operations();
    if coords.len() != expected_len {
        return Err(JsspError::InvalidInitializerConfiguration {
            detail: format!(
                "coordinate vector length {} does not match instance operation count {expected_len}",
                coords.len()
            ),
        });
    }
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(JsspError::InvalidInitializerConfiguration {
            detail: format!("MWR exponent alpha must be finite and >= 0, got {alpha}"),
        });
    }

    let num_jobs = instance.num_jobs();
    let num_machines = instance.num_machines();
    let mut rem_ops = vec![num_machines; num_jobs];
    let mut next_op = vec![0usize; num_jobs];
    let mut rem_work = initial_job_remaining_work(instance);
    let mut eligible: Vec<usize> = Vec::with_capacity(num_jobs);
    let mut weights: Vec<f64> = Vec::with_capacity(num_jobs);
    let mut genes: Vec<JobId> = Vec::with_capacity(expected_len);

    for (pos, &raw_u) in coords.iter().enumerate() {
        if !raw_u.is_finite() {
            return Err(JsspError::InvalidInitializerConfiguration {
                detail: format!("non-finite coordinate at dimension {pos}: {raw_u}"),
            });
        }
        let u = raw_u.clamp(0.0, 1.0 - f64::EPSILON);

        eligible.clear();
        for (job_idx, &rem) in rem_ops.iter().enumerate() {
            if rem > 0 {
                eligible.push(job_idx);
            }
        }

        let chosen = select_eligible_job(&eligible, &rem_work, alpha, u, &mut weights);
        genes.push(JobId(chosen));

        let op_idx = next_op[chosen];
        let op = instance.operation_by_indices(chosen, op_idx)?;
        rem_work[chosen] = rem_work[chosen].saturating_sub(op.processing_time);
        next_op[chosen] += 1;
        rem_ops[chosen] -= 1;
    }

    Ok(ObrChromosome::new(genes))
}

/// Computes initial total remaining processing work `W_j(0)` per job.
#[must_use]
pub(crate) fn initial_job_remaining_work(instance: &JsspInstance) -> Vec<u64> {
    (0..instance.num_jobs())
        .map(|j| {
            instance
                .job_operations(JobId(j))
                .map(|ops| ops.iter().map(|op| op.processing_time).sum())
                .unwrap_or(0)
        })
        .collect()
}

fn select_eligible_job(
    eligible: &[usize],
    rem_work: &[u64],
    alpha: f64,
    u: f64,
    weights_buf: &mut Vec<f64>,
) -> usize {
    debug_assert!(!eligible.is_empty());
    if eligible.len() == 1 || alpha == 0.0 {
        let idx = ((u * (eligible.len() as f64)).floor() as usize).min(eligible.len() - 1);
        return eligible[idx];
    }

    weights_buf.clear();
    let mut total_w = 0.0_f64;
    for &job_idx in eligible {
        let w = compute_mwr_weight(rem_work[job_idx], alpha);
        weights_buf.push(w);
        total_w += w;
    }

    if total_w <= 0.0 || !total_w.is_finite() {
        let idx = ((u * (eligible.len() as f64)).floor() as usize).min(eligible.len() - 1);
        return eligible[idx];
    }

    let target = u * total_w;
    let mut cum = 0.0_f64;
    for (idx, &job_idx) in eligible.iter().enumerate() {
        cum += weights_buf[idx];
        if target < cum {
            return job_idx;
        }
    }
    eligible[eligible.len() - 1]
}

#[inline]
fn compute_mwr_weight(work: u64, alpha: f64) -> f64 {
    if (alpha - 1.0).abs() == 0.0 {
        work as f64
    } else if (alpha - 2.0).abs() == 0.0 {
        let wf = work as f64;
        wf * wf
    } else {
        (work as f64).powf(alpha)
    }
}
