//! Deterministic Serial Schedule Generation Scheme (Serial SGS) decoders for OBR.
//!
//! The canonical decoder is [`decode_obr`] (`SerialInsertionDecoder`), which scans the
//! validated OBR chromosome left-to-right and inserts each operation into the earliest
//! feasible machine idle gap (generating active schedules). [`decode_obr_append_only`]
//! is retained strictly as a testing reference to contrast semi-active tail-append decoding
//! against earliest-gap insertion.

use crate::error::JsspError;
use crate::model::{JsspInstance, OperationId, Schedule, ScheduledOperation};
use crate::representation::ObrChromosome;

/// Stateful handle for the canonical deterministic Serial SGS earliest-gap insertion decoder.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SerialInsertionDecoder;

impl SerialInsertionDecoder {
    /// Creates a new `SerialInsertionDecoder` instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Decodes a validated OBR chromosome into an active schedule for `instance`.
    ///
    /// # Errors
    /// Returns [`JsspError`] if `chromosome` is invalid or checked time arithmetic overflows.
    pub fn decode(
        &self,
        instance: &JsspInstance,
        chromosome: &ObrChromosome,
    ) -> Result<Schedule, JsspError> {
        decode_obr(instance, chromosome)
    }
}

/// Decodes `chromosome` for `instance` using the canonical deterministic Serial SGS
/// with earliest feasible idle-gap insertion.
///
/// # Errors
/// Returns [`JsspError`] if OBR validation fails or `u64` time addition overflows.
pub fn decode_obr(
    instance: &JsspInstance,
    chromosome: &ObrChromosome,
) -> Result<Schedule, JsspError> {
    decode_obr_internal(instance, chromosome, true)
}

/// Decodes `chromosome` for `instance` using semi-active append-only placement (appending
/// each operation after the last scheduled operation on its machine without scanning interior
/// idle gaps).
///
/// Retained as a deterministic reference for comparative unit/integration tests against
/// [`decode_obr`].
///
/// # Errors
/// Returns [`JsspError`] if OBR validation fails or `u64` time addition overflows.
pub fn decode_obr_append_only(
    instance: &JsspInstance,
    chromosome: &ObrChromosome,
) -> Result<Schedule, JsspError> {
    decode_obr_internal(instance, chromosome, false)
}

fn find_earliest_gap_slot(
    timeline: &[ScheduledOperation],
    job_ready: u64,
    processing_time: u64,
) -> Result<(u64, u64, usize), JsspError> {
    let mut prev_end: u64 = 0;

    for (idx, existing) in timeline.iter().enumerate() {
        let cand_start = job_ready.max(prev_end);
        let cand_end = cand_start
            .checked_add(processing_time)
            .ok_or(JsspError::TimeOverflow)?;
        if cand_end <= existing.start_time {
            return Ok((cand_start, cand_end, idx));
        }
        prev_end = prev_end.max(existing.completion_time);
    }

    let tail_start = job_ready.max(prev_end);
    let tail_end = tail_start
        .checked_add(processing_time)
        .ok_or(JsspError::TimeOverflow)?;
    Ok((tail_start, tail_end, timeline.len()))
}

fn find_append_slot(
    timeline: &[ScheduledOperation],
    job_ready: u64,
    processing_time: u64,
) -> Result<(u64, u64, usize), JsspError> {
    let machine_ready = timeline.last().map_or(0, |op| op.completion_time);
    let start_time = job_ready.max(machine_ready);
    let completion_time = start_time
        .checked_add(processing_time)
        .ok_or(JsspError::TimeOverflow)?;
    Ok((start_time, completion_time, timeline.len()))
}

fn decode_obr_internal(
    instance: &JsspInstance,
    chromosome: &ObrChromosome,
    allow_gap_insertion: bool,
) -> Result<Schedule, JsspError> {
    let mut workspace = SerialDecoderWorkspace::new(instance);
    workspace.decode(instance, chromosome, allow_gap_insertion)
}

/// Reusable scratch buffers for zero-allocation repeated calls to the canonical
/// M2 Serial SGS decoder inside inner search loops.
///
/// Produces bit-for-bit identical machine placements, start/completion times, and
/// `u64` makespans to [`decode_obr`].
#[derive(Debug, Clone)]
pub struct SerialDecoderWorkspace {
    cached_num_jobs: usize,
    cached_num_machines: usize,
    flat_specs: Vec<(usize, u64)>,
    machine_timelines: Vec<Vec<ScheduledOperation>>,
    interval_timelines: Vec<Vec<(u64, u64)>>,
    job_next_op: Vec<usize>,
    job_ready_time: Vec<u64>,
    scheduled_by_flat_id: Vec<Option<ScheduledOperation>>,
}

impl SerialDecoderWorkspace {
    /// Allocates reusable decoder scratch storage sized for `instance`.
    #[must_use]
    pub fn new(instance: &JsspInstance) -> Self {
        let num_jobs = instance.num_jobs();
        let num_machines = instance.num_machines();
        let num_ops = instance.num_operations();
        let mut flat_specs = Vec::with_capacity(num_ops);
        for j in 0..num_jobs {
            if let Some(ops) = instance.job_operations(crate::model::JobId(j)) {
                for op in ops {
                    flat_specs.push((op.machine.0, op.processing_time));
                }
            }
        }
        Self {
            cached_num_jobs: num_jobs,
            cached_num_machines: num_machines,
            flat_specs,
            machine_timelines: (0..num_machines)
                .map(|_| Vec::with_capacity(num_jobs))
                .collect(),
            interval_timelines: (0..num_machines)
                .map(|_| Vec::with_capacity(num_jobs))
                .collect(),
            job_next_op: vec![0; num_jobs],
            job_ready_time: vec![0; num_jobs],
            scheduled_by_flat_id: vec![None; num_ops],
        }
    }

    fn reset_for(&mut self, instance: &JsspInstance, with_full_schedule: bool) {
        let num_jobs = instance.num_jobs();
        let num_machines = instance.num_machines();
        let num_ops = instance.num_operations();

        if self.cached_num_jobs != num_jobs
            || self.cached_num_machines != num_machines
            || self.flat_specs.len() != num_ops
        {
            self.cached_num_jobs = num_jobs;
            self.cached_num_machines = num_machines;
            self.flat_specs.clear();
            self.flat_specs.reserve(num_ops);
            for j in 0..num_jobs {
                if let Some(ops) = instance.job_operations(crate::model::JobId(j)) {
                    for op in ops {
                        self.flat_specs.push((op.machine.0, op.processing_time));
                    }
                }
            }
        }

        if with_full_schedule {
            if self.machine_timelines.len() != num_machines {
                self.machine_timelines
                    .resize_with(num_machines, || Vec::with_capacity(num_jobs));
            }
            for tl in &mut self.machine_timelines {
                tl.clear();
            }
            self.scheduled_by_flat_id.resize(num_ops, None);
            self.scheduled_by_flat_id.fill(None);
        } else {
            if self.interval_timelines.len() != num_machines {
                self.interval_timelines
                    .resize_with(num_machines, || Vec::with_capacity(num_jobs));
            }
            for tl in &mut self.interval_timelines {
                tl.clear();
            }
        }

        self.job_next_op.resize(num_jobs, 0);
        self.job_next_op.fill(0);
        self.job_ready_time.resize(num_jobs, 0);
        self.job_ready_time.fill(0);
    }

    fn decode_core(
        &mut self,
        instance: &JsspInstance,
        chromosome: &ObrChromosome,
        allow_gap_insertion: bool,
        store_operations: bool,
    ) -> Result<u64, JsspError> {
        let num_jobs = instance.num_jobs();
        let num_machines = instance.num_machines();
        let num_ops = instance.num_operations();
        let genes = chromosome.genes();
        if genes.len() != num_ops {
            return Err(JsspError::InvalidChromosomeLength {
                expected: num_ops,
                actual: genes.len(),
            });
        }

        self.reset_for(instance, store_operations);
        let mut makespan: u64 = 0;

        for (pos, &job_id) in genes.iter().enumerate() {
            if job_id.0 >= num_jobs {
                return Err(JsspError::InvalidJobId {
                    position: pos,
                    job: job_id.0,
                    num_jobs,
                });
            }
            let op_idx = self.job_next_op[job_id.0];
            if op_idx >= num_machines {
                return Err(JsspError::InvalidJobMultiplicity {
                    job: job_id.0,
                    expected: num_machines,
                    actual: op_idx + 1,
                });
            }
            self.job_next_op[job_id.0] = op_idx + 1;

            let flat_idx = job_id.0 * num_machines + op_idx;
            let (machine_idx, proc_time) = self.flat_specs[flat_idx];
            let ready = self.job_ready_time[job_id.0];

            if store_operations {
                let op_id = OperationId::from_indices(job_id.0, op_idx);
                let spec_op = instance.operation_by_indices(job_id.0, op_idx)?;
                let timeline = &mut self.machine_timelines[machine_idx];
                let (start_time, completion_time, insert_idx) = if allow_gap_insertion {
                    find_earliest_gap_slot(timeline, ready, proc_time)?
                } else {
                    find_append_slot(timeline, ready, proc_time)?
                };
                let scheduled = ScheduledOperation {
                    op_id,
                    machine: spec_op.machine,
                    start_time,
                    completion_time,
                    processing_time: proc_time,
                };
                timeline.insert(insert_idx, scheduled);
                self.job_ready_time[job_id.0] = completion_time;
                makespan = makespan.max(completion_time);
                self.scheduled_by_flat_id[flat_idx] = Some(scheduled);
            } else {
                let timeline = &mut self.interval_timelines[machine_idx];
                let (start_time, completion_time, insert_idx) =
                    find_earliest_gap_intervals(timeline, ready, proc_time)?;
                timeline.insert(insert_idx, (start_time, completion_time));
                self.job_ready_time[job_id.0] = completion_time;
                makespan = makespan.max(completion_time);
            }
        }

        // By pigeonhole principle (genes.len() == J * M and each job's count <= M),
        // reaching here guarantees every job occurred exactly M times.
        Ok(makespan)
    }

    /// Decodes `chromosome` into a complete [`Schedule`] using identical M2 semantics.
    ///
    /// # Errors
    /// Returns [`JsspError`] if OBR validation fails or checked `u64` time arithmetic overflows.
    pub fn decode(
        &mut self,
        instance: &JsspInstance,
        chromosome: &ObrChromosome,
        allow_gap_insertion: bool,
    ) -> Result<Schedule, JsspError> {
        let num_jobs = instance.num_jobs();
        let num_machines = instance.num_machines();
        let num_ops = instance.num_operations();
        let makespan = self.decode_core(instance, chromosome, allow_gap_insertion, true)?;

        let mut operations = Vec::with_capacity(num_ops);
        for (flat_idx, slot) in self.scheduled_by_flat_id.iter().enumerate() {
            match slot {
                Some(op) => operations.push(*op),
                None => {
                    let op_id = OperationId::from_flat_index(flat_idx, num_jobs, num_machines)?;
                    return Err(JsspError::MissingOperation {
                        job: op_id.job.0,
                        op_index: op_id.op_index.0,
                    });
                }
            }
        }

        Ok(Schedule::new(num_jobs, num_machines, operations, makespan))
    }

    /// Decodes `chromosome` via canonical earliest-gap Serial SGS and returns its exact
    /// integer makespan `C_max` without constructing a heap-allocated [`Schedule`] vector.
    ///
    /// # Errors
    /// Returns [`JsspError`] if OBR validation fails or `u64` time addition overflows.
    pub fn decode_makespan_only(
        &mut self,
        instance: &JsspInstance,
        chromosome: &ObrChromosome,
    ) -> Result<u64, JsspError> {
        self.decode_core(instance, chromosome, true, false)
    }
}

#[inline]
fn find_earliest_gap_intervals(
    timeline: &[(u64, u64)],
    job_ready: u64,
    processing_time: u64,
) -> Result<(u64, u64, usize), JsspError> {
    let mut prev_end: u64 = 0;

    for (idx, &(s_i, e_i)) in timeline.iter().enumerate() {
        let cand_start = job_ready.max(prev_end);
        let cand_end = cand_start
            .checked_add(processing_time)
            .ok_or(JsspError::TimeOverflow)?;
        if cand_end <= s_i {
            return Ok((cand_start, cand_end, idx));
        }
        prev_end = prev_end.max(e_i);
    }

    let tail_start = job_ready.max(prev_end);
    let tail_end = tail_start
        .checked_add(processing_time)
        .ok_or(JsspError::TimeOverflow)?;
    Ok((tail_start, tail_end, timeline.len()))
}
