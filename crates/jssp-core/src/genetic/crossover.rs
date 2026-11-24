//! Precedence Operation Crossover (POX) preserving exact OBR job multiplicities (`D07-A`).

use crate::error::JsspError;
use crate::initialization::qmc::u01_from_u64;
use crate::model::JobId;
use crate::representation::ObrChromosome;
use crate::rng::NamedRngStream;

/// Applies Precedence Operation Crossover (POX) to parent pair `(p1, p2)` with probability
/// `crossover_rate`, returning `(child1, child2, did_crossover)`.
///
/// Partition `J1` includes each job independently with probability `0.5`, rejecting trivial
/// empty or full subsets (`1 <= |J1| <= num_jobs - 1` whenever `num_jobs >= 2`).
///
/// # Errors
/// Returns [`JsspError`] if chromosome lengths mismatch or RNG sampling fails.
pub fn pox_crossover(
    p1: &ObrChromosome,
    p2: &ObrChromosome,
    num_jobs: usize,
    crossover_rate: f64,
    rng: &mut NamedRngStream,
) -> Result<(ObrChromosome, ObrChromosome, bool), JsspError> {
    let g1 = p1.genes();
    let g2 = p2.genes();
    if g1.len() != g2.len() {
        return Err(JsspError::InvalidChromosomeLength {
            expected: g1.len(),
            actual: g2.len(),
        });
    }
    if !(0.0..=1.0).contains(&crossover_rate) {
        return Err(JsspError::InvalidGeneticConfiguration {
            detail: format!("crossover_rate must lie in [0, 1], got {crossover_rate}"),
        });
    }

    let u = u01_from_u64(rng.next_u64());
    if u >= crossover_rate || num_jobs < 2 {
        return Ok((p1.clone(), p2.clone(), false));
    }

    let mut in_j1 = vec![false; num_jobs];
    for _attempt in 0..64 {
        let mut count_j1 = 0usize;
        for flag in &mut in_j1 {
            let pick = (rng.next_u64() & 1) == 1;
            *flag = pick;
            if pick {
                count_j1 += 1;
            }
        }
        if count_j1 > 0 && count_j1 < num_jobs {
            let c1 = assemble_pox_child(g1, g2, &in_j1);
            let c2 = assemble_pox_child(g2, g1, &in_j1);
            return Ok((ObrChromosome::new(c1), ObrChromosome::new(c2), true));
        }
    }

    // Deterministic fallback only if 64 consecutive ChaCha20 coin-flip vectors were trivial.
    let flip = rng.gen_range_usize(num_jobs)?;
    in_j1[flip] = !in_j1[flip];
    let c1 = assemble_pox_child(g1, g2, &in_j1);
    let c2 = assemble_pox_child(g2, g1, &in_j1);
    Ok((ObrChromosome::new(c1), ObrChromosome::new(c2), true))
}

fn assemble_pox_child(primary: &[JobId], secondary: &[JobId], in_j1: &[bool]) -> Vec<JobId> {
    let len = primary.len();
    let mut child = vec![JobId(0); len];
    let mut fixed = vec![false; len];

    for (idx, &gene) in primary.iter().enumerate() {
        if in_j1[gene.0] {
            child[idx] = gene;
            fixed[idx] = true;
        }
    }

    let mut fill_pos = 0usize;
    for &gene in secondary {
        if !in_j1[gene.0] {
            while fixed[fill_pos] {
                fill_pos += 1;
            }
            child[fill_pos] = gene;
            fill_pos += 1;
        }
    }

    child
}
