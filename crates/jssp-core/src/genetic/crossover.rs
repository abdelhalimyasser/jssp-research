//! Precedence Operation Crossover (POX) preserving exact OBR job multiplicities (`D07-A`).

use crate::error::JsspError;
use crate::initialization::qmc::u01_from_u64;
use crate::model::JobId;
use crate::representation::ObrChromosome;
use crate::rng::NamedRngStream;

/// Pre-allocated scratch buffers eliminating repeated heap allocations inside POX loops.
#[derive(Debug, Clone)]
pub struct PoxWorkspace {
    in_j1: Vec<bool>,
    fixed: Vec<bool>,
}

impl PoxWorkspace {
    /// Constructs a scratch workspace sized for `num_jobs` jobs and chromosome length `chrom_len`.
    #[must_use]
    pub fn new(num_jobs: usize, chrom_len: usize) -> Self {
        Self {
            in_j1: vec![false; num_jobs],
            fixed: vec![false; chrom_len],
        }
    }

    /// Executes POX crossover writing children into reusable buffers `out1` and `out2`.
    ///
    /// # Errors
    /// Returns [`JsspError`] if chromosome lengths mismatch or `crossover_rate` is invalid.
    pub fn crossover_into(
        &mut self,
        parents: (&[JobId], &[JobId]),
        num_jobs: usize,
        crossover_rate: f64,
        rng: &mut NamedRngStream,
        children: (&mut Vec<JobId>, &mut Vec<JobId>),
    ) -> Result<bool, JsspError> {
        let (p1, p2) = parents;
        let (out1, out2) = children;
        if p1.len() != p2.len() {
            return Err(JsspError::InvalidChromosomeLength {
                expected: p1.len(),
                actual: p2.len(),
            });
        }
        if !(0.0..=1.0).contains(&crossover_rate) {
            return Err(JsspError::InvalidGeneticConfiguration {
                detail: format!("crossover_rate must lie in [0, 1], got {crossover_rate}"),
            });
        }

        let u = u01_from_u64(rng.next_u64());
        if u >= crossover_rate || num_jobs < 2 {
            out1.clear();
            out1.extend_from_slice(p1);
            out2.clear();
            out2.extend_from_slice(p2);
            return Ok(false);
        }

        self.in_j1.resize(num_jobs, false);
        self.fixed.resize(p1.len(), false);

        let mut split_ok = false;
        for _attempt in 0..64 {
            let mut count_j1 = 0usize;
            for flag in &mut self.in_j1 {
                let pick = (rng.next_u64() & 1) == 1;
                *flag = pick;
                if pick {
                    count_j1 += 1;
                }
            }
            if count_j1 > 0 && count_j1 < num_jobs {
                split_ok = true;
                break;
            }
        }
        if !split_ok {
            let flip = rng.gen_range_usize(num_jobs)?;
            self.in_j1[flip] = !self.in_j1[flip];
        }

        assemble_pox_child_into(p1, p2, &self.in_j1, &mut self.fixed, out1);
        assemble_pox_child_into(p2, p1, &self.in_j1, &mut self.fixed, out2);
        Ok(true)
    }
}

/// Applies Precedence Operation Crossover (POX) to parent pair `(p1, p2)` with probability
/// `crossover_rate`, returning `(child1, child2, did_crossover)`.
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
    let mut ws = PoxWorkspace::new(num_jobs, p1.len());
    let mut c1 = Vec::with_capacity(p1.len());
    let mut c2 = Vec::with_capacity(p1.len());
    let did_cx = ws.crossover_into(
        (p1.genes(), p2.genes()),
        num_jobs,
        crossover_rate,
        rng,
        (&mut c1, &mut c2),
    )?;
    Ok((ObrChromosome::new(c1), ObrChromosome::new(c2), did_cx))
}

fn assemble_pox_child_into(
    primary: &[JobId],
    secondary: &[JobId],
    in_j1: &[bool],
    fixed: &mut [bool],
    child: &mut Vec<JobId>,
) {
    let len = primary.len();
    child.resize(len, JobId(0));
    fixed.fill(false);

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
}
