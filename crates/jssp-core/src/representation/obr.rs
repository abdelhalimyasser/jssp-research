//! Operation-Based Representation (OBR) chromosome and validation helpers.

use crate::error::JsspError;
use crate::model::{JobId, JsspInstance, OperationId};

/// Operation-Based Representation (OBR) chromosome encoding job ID permutations with multiplicity.
///
/// For an instance with `J` jobs and `M` machines, a valid OBR chromosome has length
/// `L = J * M` where each job ID `j in 0..J` appears exactly `M` times.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObrChromosome {
    genes: Vec<JobId>,
}

impl ObrChromosome {
    /// Wraps a sequence of [`JobId`] genes into an `ObrChromosome`.
    #[must_use]
    pub fn new(genes: Vec<JobId>) -> Self {
        Self { genes }
    }

    /// Constructs an `ObrChromosome` from zero-based job index values.
    #[must_use]
    pub fn from_job_indices(indices: &[usize]) -> Self {
        let genes = indices.iter().copied().map(JobId).collect();
        Self { genes }
    }

    /// Constructs an `ObrChromosome` and validates it immediately against `instance`.
    ///
    /// # Errors
    /// Returns [`JsspError`] if length, job IDs, or job multiplicities are invalid.
    pub fn try_new(genes: Vec<JobId>, instance: &JsspInstance) -> Result<Self, JsspError> {
        let chromosome = Self::new(genes);
        chromosome.validate(instance)?;
        Ok(chromosome)
    }

    /// Constructs a deterministic canonical valid OBR chromosome for `instance`
    /// by repeating `0..num_jobs` for `num_machines` rounds.
    #[must_use]
    pub fn canonical(instance: &JsspInstance) -> Self {
        let mut genes = Vec::with_capacity(instance.num_operations());
        for _ in 0..instance.num_machines() {
            for job_idx in 0..instance.num_jobs() {
                genes.push(JobId(job_idx));
            }
        }
        Self { genes }
    }

    /// Number of genes in the chromosome.
    #[must_use]
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    /// Returns `true` if the chromosome has length zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.genes.is_empty()
    }

    /// Borrowed slice of typed job IDs in chromosome scan order.
    #[must_use]
    pub fn genes(&self) -> &[JobId] {
        &self.genes
    }

    /// Mutable slice of typed job IDs for in-place validity-preserving genetic operators.
    pub fn genes_mut(&mut self) -> &mut [JobId] {
        &mut self.genes
    }

    /// Returns the chromosome as plain `usize` job indices.
    #[must_use]
    pub fn to_job_indices(&self) -> Vec<usize> {
        self.genes.iter().map(|job| job.0).collect()
    }

    /// Counts occurrences of each job ID in `0..num_jobs` while checking index bounds.
    ///
    /// # Errors
    /// Returns [`JsspError::EmptyJobs`] if `num_jobs == 0`, or
    /// [`JsspError::InvalidJobId`] if any gene equals or exceeds `num_jobs`.
    pub fn job_counts(&self, num_jobs: usize) -> Result<Vec<usize>, JsspError> {
        if num_jobs == 0 {
            return Err(JsspError::EmptyJobs);
        }

        let mut counts = vec![0usize; num_jobs];
        for (position, gene) in self.genes.iter().enumerate() {
            if gene.0 >= num_jobs {
                return Err(JsspError::InvalidJobId {
                    position,
                    job: gene.0,
                    num_jobs,
                });
            }
            counts[gene.0] += 1;
        }
        Ok(counts)
    }

    /// Validates chromosome length, job ID bounds, and exact job multiplicities against `instance`.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidChromosomeLength`], [`JsspError::InvalidJobId`], or
    /// [`JsspError::InvalidJobMultiplicity`].
    pub fn validate(&self, instance: &JsspInstance) -> Result<(), JsspError> {
        let expected_len = instance.num_operations();
        if self.genes.len() != expected_len {
            return Err(JsspError::InvalidChromosomeLength {
                expected: expected_len,
                actual: self.genes.len(),
            });
        }

        let counts = self.job_counts(instance.num_jobs())?;
        let expected_mult = instance.num_machines();
        for (job, &actual) in counts.iter().enumerate() {
            if actual != expected_mult {
                return Err(JsspError::InvalidJobMultiplicity {
                    job,
                    expected: expected_mult,
                    actual,
                });
            }
        }
        Ok(())
    }

    /// Validates this chromosome against `instance` and resolves each position `t` to its
    /// unique [`OperationId`] `(job, k)` where `k` is the zero-based occurrence count of `job`.
    ///
    /// # Errors
    /// Propagates any validation error from [`Self::validate`].
    pub fn resolve_operations(
        &self,
        instance: &JsspInstance,
    ) -> Result<Vec<OperationId>, JsspError> {
        self.validate(instance)?;

        let mut next_op_for_job = vec![0usize; instance.num_jobs()];
        let mut resolved = Vec::with_capacity(self.genes.len());

        for gene in &self.genes {
            let k = next_op_for_job[gene.0];
            resolved.push(OperationId::from_indices(gene.0, k));
            next_op_for_job[gene.0] = k + 1;
        }

        Ok(resolved)
    }
}
