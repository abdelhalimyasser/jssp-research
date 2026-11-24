//! Strictly within-view POX crossover & distinct-job mutation for Model A (`D06` §2–3).

use crate::error::JsspError;
use crate::genetic::{PoxWorkspace, mutate_obr};
use crate::model::JobId;
use crate::representation::ObrChromosome;
use crate::rng::RunRngStreams;

use super::individual::DualViewIndividual;

/// Result of one within-view reproduction step: `((lam1, bald1), (lam2, bald2), cx_count, mut_count)`.
pub type WithinViewPairBatch = (
    (ObrChromosome, ObrChromosome),
    (ObrChromosome, ObrChromosome),
    usize,
    usize,
);

/// Produces two paired offspring `(G_c^L, G_c^B)` from selected parents `(pa, pb)`
/// executing POX crossover and mutation **strictly within view** (`G^L x G^L`, `G^B x G^B`).
///
/// Cross-view crossover is never performed (`D06` Lineage Separation Invariant).
///
/// # Errors
/// Returns [`JsspError`] if crossover/mutation configuration or chromosome lengths are invalid.
#[allow(clippy::too_many_arguments)]
pub fn reproduce_dual_view_within_view(
    pa: &DualViewIndividual,
    pb: &DualViewIndividual,
    num_jobs: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    pox_ws: &mut PoxWorkspace,
    child_bufs: (&mut Vec<JobId>, &mut Vec<JobId>),
    streams: &mut RunRngStreams,
) -> Result<WithinViewPairBatch, JsspError> {
    let (buf1, buf2) = child_bufs;
    let mut cx_count = 0usize;
    let mut mut_count = 0usize;

    // 1. Lamarckian view lineage: G_a^L x G_b^L
    if pox_ws.crossover_into(
        (pa.lam_genotype.genes(), pb.lam_genotype.genes()),
        num_jobs,
        crossover_rate,
        &mut streams.crossover,
        (buf1, buf2),
    )? {
        cx_count += 1;
    }
    let mut lam_c1 = ObrChromosome::new(buf1.clone());
    let mut lam_c2 = ObrChromosome::new(buf2.clone());
    if mutate_obr(&mut lam_c1, mutation_rate, &mut streams.mutation)? {
        mut_count += 1;
    }
    if mutate_obr(&mut lam_c2, mutation_rate, &mut streams.mutation)? {
        mut_count += 1;
    }

    // 2. Baldwinian view lineage: G_a^B x G_b^B
    if pox_ws.crossover_into(
        (pa.bald_genotype.genes(), pb.bald_genotype.genes()),
        num_jobs,
        crossover_rate,
        &mut streams.crossover,
        (buf1, buf2),
    )? {
        cx_count += 1;
    }
    let mut bald_c1 = ObrChromosome::new(buf1.clone());
    let mut bald_c2 = ObrChromosome::new(buf2.clone());
    if mutate_obr(&mut bald_c1, mutation_rate, &mut streams.mutation)? {
        mut_count += 1;
    }
    if mutate_obr(&mut bald_c2, mutation_rate, &mut streams.mutation)? {
        mut_count += 1;
    }

    Ok(((lam_c1, bald_c1), (lam_c2, bald_c2), cx_count, mut_count))
}
