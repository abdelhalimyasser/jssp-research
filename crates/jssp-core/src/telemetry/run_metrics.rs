//! Layer A population diagnostics and Layer B Phase A GA trajectory telemetry.

use std::collections::HashSet;

use sha2::{Digest, Sha256};

use crate::error::JsspError;
use crate::model::{JobId, JsspInstance, Schedule};
use crate::representation::ObrChromosome;

/// Diagnostic summary computed over a complete decoded OBR population (Layer A / final pop).
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationDiagnostics {
    pub population_size: usize,
    pub feasible_count: usize,
    pub best_makespan: u64,
    pub mean_makespan: f64,
    pub median_makespan: f64,
    pub worst_makespan: u64,
    pub std_makespan: f64,
    pub unique_chromosomes: usize,
    pub unique_schedules: usize,
    pub obr_duplicate_rate: f64,
    pub schedule_duplicate_rate: f64,
    pub obr_hamming_diversity: f64,
    pub schedule_diversity: f64,
    pub qmc_1d_ks_discrepancy: Option<f64>,
    pub generation_elapsed_ns: u64,
    pub decode_elapsed_ns: u64,
}

/// Per-generation progress record inside a Classical GA run.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationSnapshot {
    pub generation: usize,
    pub evals_used: usize,
    pub best_makespan: u64,
    pub mean_makespan: f64,
    pub obr_hamming_diversity: f64,
    pub unique_chromosomes: usize,
}

/// Complete outcome and telemetry bundle from a Phase A `(initializer, C01)` experiment run.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseARunResult {
    pub initializer_id: &'static str,
    pub pop_size: usize,
    pub evaluation_budget: usize,
    pub evals_used: usize,
    pub generations_executed: usize,
    pub initial: PopulationDiagnostics,
    pub final_pop: PopulationDiagnostics,
    pub best_makespan: u64,
    pub best_found_eval: usize,
    pub best_found_generation: usize,
    pub best_chromosome: Vec<usize>,
    pub best_schedule_hash: String,
    pub anytime_cmax_auc: f64,
    pub crossover_count: usize,
    pub mutation_count: usize,
    pub total_elapsed_ns: u64,
    pub trajectory: Vec<GenerationSnapshot>,
}

/// Computes exact normalized pairwise OBR Hamming diversity in `O(N * L)` operations via
/// `(1 / L) * sum_l (N^2 - sum_j c_{l,j}^2) / (N * (N - 1))`.
#[must_use]
pub fn compute_obr_hamming_diversity(chromosomes: &[ObrChromosome], num_jobs: usize) -> f64 {
    let pop_size = chromosomes.len();
    if pop_size < 2 || num_jobs == 0 {
        return 0.0;
    }
    let chrom_len = chromosomes[0].len();
    if chrom_len == 0 {
        return 0.0;
    }

    let denom_pairs = (pop_size as f64) * ((pop_size - 1) as f64);
    let n_sq = (pop_size as u64) * (pop_size as u64);
    let mut counts = vec![0u64; num_jobs];
    let mut sum_locus_div = 0.0_f64;

    for locus in 0..chrom_len {
        counts.fill(0);
        for chr in chromosomes {
            let job = chr.genes()[locus].0;
            if job < num_jobs {
                counts[job] += 1;
            }
        }
        let sum_sq: u64 = counts.iter().map(|&c| c * c).sum();
        let diff_pairs_x2 = n_sq.saturating_sub(sum_sq);
        sum_locus_div += (diff_pairs_x2 as f64) / denom_pairs;
    }

    sum_locus_div / (chrom_len as f64)
}

/// Counts unique OBR chromosomes in a population.
#[must_use]
pub fn count_unique_chromosomes(chromosomes: &[ObrChromosome]) -> usize {
    let unique: HashSet<&[JobId]> = chromosomes.iter().map(ObrChromosome::genes).collect();
    unique.len()
}

/// Computes canonical SHA-256 hex digest of a decoded schedule's ordered machine sequences.
///
/// Two distinct OBR chromosomes that map under Serial SGS to identical machine processing
/// orders on every machine yield the identical schedule digest.
#[must_use]
pub fn schedule_machine_sequence_hash(schedule: &Schedule) -> String {
    let mut hasher = Sha256::new();
    let seqs = schedule.machine_sequences();
    hasher.update((seqs.len() as u64).to_le_bytes());
    for m_seq in seqs {
        hasher.update((m_seq.operations.len() as u64).to_le_bytes());
        for op in m_seq.operations {
            hasher.update((op.job.0 as u64).to_le_bytes());
            hasher.update((op.op_index.0 as u64).to_le_bytes());
        }
    }
    let digest = hasher.finalize();
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(32);
    // 128-bit prefix gives collision probability < 2^-64 with compact JSON footprint
    for &b in &digest[..16] {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
}

/// Computes unique schedule count (by canonical machine sequences) and normalized pairwise
/// machine-slot operation diversity across `schedules`.
///
/// # Errors
/// Returns [`JsspError`] if operation flat-index mapping fails.
pub fn compute_schedule_diversity(
    instance: &JsspInstance,
    schedules: &[Schedule],
) -> Result<(usize, f64), JsspError> {
    let pop_size = schedules.len();
    if pop_size == 0 {
        return Ok((0, 0.0));
    }
    let num_jobs = instance.num_jobs();
    let num_machines = instance.num_machines();
    let num_ops = instance.num_operations();

    let mut unique_hashes: HashSet<String> = HashSet::with_capacity(pop_size);
    let mut slot_matrix: Vec<usize> = Vec::with_capacity(pop_size * num_ops);

    for sched in schedules {
        unique_hashes.insert(schedule_machine_sequence_hash(sched));
        for m_seq in sched.machine_sequences() {
            for op in m_seq.operations {
                slot_matrix.push(op.flat_index(num_jobs, num_machines)?);
            }
        }
    }

    if pop_size < 2 || num_ops == 0 {
        return Ok((unique_hashes.len(), 0.0));
    }

    let denom_pairs = (pop_size as f64) * ((pop_size - 1) as f64);
    let n_sq = (pop_size as u64) * (pop_size as u64);
    let mut counts = vec![0u64; num_ops];
    let mut sum_slot_div = 0.0_f64;

    for slot in 0..num_ops {
        counts.fill(0);
        for ind in 0..pop_size {
            let flat_op = slot_matrix[ind * num_ops + slot];
            counts[flat_op] += 1;
        }
        let sum_sq: u64 = counts.iter().map(|&c| c * c).sum();
        let diff_pairs_x2 = n_sq.saturating_sub(sum_sq);
        sum_slot_div += (diff_pairs_x2 as f64) / denom_pairs;
    }

    Ok((unique_hashes.len(), sum_slot_div / (num_ops as f64)))
}

/// Computes exact 1D Kolmogorov–Smirnov star discrepancy `D_N` for samples in `[0, 1)`.
#[must_use]
pub fn compute_1d_ks_discrepancy(coords: &[f64]) -> Option<f64> {
    if coords.is_empty() {
        return None;
    }
    let mut sorted = coords.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n_f = sorted.len() as f64;
    let mut d_max = 0.0_f64;
    for (idx, &x) in sorted.iter().enumerate() {
        let upper = ((idx + 1) as f64) / n_f - x;
        let lower = x - (idx as f64) / n_f;
        d_max = d_max.max(upper).max(lower);
    }
    Some(d_max)
}

/// Summarizes full Layer A or final-population diagnostics from chromosomes and verified schedules.
///
/// # Errors
/// Returns [`JsspError`] if population is empty or diversity calculation fails.
pub fn summarize_population_diagnostics(
    instance: &JsspInstance,
    chromosomes: &[ObrChromosome],
    schedules: &[Schedule],
    feasible_count: usize,
    first_dim_coords: Option<&[f64]>,
    generation_elapsed_ns: u64,
    decode_elapsed_ns: u64,
) -> Result<PopulationDiagnostics, JsspError> {
    let pop_size = chromosomes.len();
    if pop_size == 0 || schedules.len() != pop_size {
        return Err(JsspError::InvalidInitializerConfiguration {
            detail: "chromosomes and schedules must be non-empty with equal length".to_owned(),
        });
    }

    let mut makespans: Vec<u64> = schedules.iter().map(Schedule::makespan).collect();
    let best_makespan = makespans.iter().copied().min().unwrap_or(0);
    let worst_makespan = makespans.iter().copied().max().unwrap_or(0);
    let mean_makespan = makespans.iter().map(|&v| v as f64).sum::<f64>() / (makespans.len() as f64);

    makespans.sort_unstable();
    let median_makespan = if pop_size % 2 == 1 {
        makespans[pop_size / 2] as f64
    } else {
        0.5 * ((makespans[pop_size / 2 - 1] as f64) + (makespans[pop_size / 2] as f64))
    };

    let std_makespan = if pop_size < 2 {
        0.0
    } else {
        let var = makespans
            .iter()
            .map(|&v| {
                let d = (v as f64) - mean_makespan;
                d * d
            })
            .sum::<f64>()
            / ((pop_size - 1) as f64);
        var.sqrt()
    };

    let unique_chromosomes = count_unique_chromosomes(chromosomes);
    let (unique_schedules, schedule_diversity) = compute_schedule_diversity(instance, schedules)?;
    let n_f = pop_size as f64;
    let obr_duplicate_rate = 1.0 - (unique_chromosomes as f64) / n_f;
    let schedule_duplicate_rate = 1.0 - (unique_schedules as f64) / n_f;
    let obr_hamming_diversity = compute_obr_hamming_diversity(chromosomes, instance.num_jobs());
    let qmc_1d_ks_discrepancy = first_dim_coords.and_then(compute_1d_ks_discrepancy);

    Ok(PopulationDiagnostics {
        population_size: pop_size,
        feasible_count,
        best_makespan,
        mean_makespan,
        median_makespan,
        worst_makespan,
        std_makespan,
        unique_chromosomes,
        unique_schedules,
        obr_duplicate_rate,
        schedule_duplicate_rate,
        obr_hamming_diversity,
        schedule_diversity,
        qmc_1d_ks_discrepancy,
        generation_elapsed_ns,
        decode_elapsed_ns,
    })
}

/// Computes normalized trapezoidal area under the best-so-far `C_max(e)` curve over
/// normalized evaluation fraction `[e_0 / budget, e_last / budget]`, returned in `C_max` units.
#[must_use]
pub fn compute_anytime_cmax_auc(trajectory: &[GenerationSnapshot], eval_budget: usize) -> f64 {
    if trajectory.is_empty() || eval_budget == 0 {
        return 0.0;
    }
    if trajectory.len() == 1 {
        return trajectory[0].best_makespan as f64;
    }
    let budget_f = eval_budget as f64;
    let mut area = 0.0_f64;
    let mut running_best = trajectory[0].best_makespan;
    let mut prev_x = (trajectory[0].evals_used as f64) / budget_f;
    let mut prev_y = running_best as f64;

    for snap in trajectory.iter().skip(1) {
        running_best = running_best.min(snap.best_makespan);
        let x = (snap.evals_used as f64) / budget_f;
        let y = running_best as f64;
        if x > prev_x {
            area += 0.5 * (prev_y + y) * (x - prev_x);
            prev_x = x;
        }
        prev_y = y;
    }

    let start_x = (trajectory[0].evals_used as f64) / budget_f;
    let width = prev_x - start_x;
    if width <= 0.0 { prev_y } else { area / width }
}
