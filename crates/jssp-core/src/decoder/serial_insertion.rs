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
    let resolved_ops: Vec<OperationId> = chromosome.resolve_operations(instance)?;

    let num_jobs = instance.num_jobs();
    let num_machines = instance.num_machines();
    let num_ops = instance.num_operations();

    let mut job_ready_time = vec![0u64; num_jobs];
    let mut machine_timelines: Vec<Vec<ScheduledOperation>> =
        vec![Vec::with_capacity(num_jobs); num_machines];
    let mut scheduled_by_flat_id: Vec<Option<ScheduledOperation>> = vec![None; num_ops];
    let mut makespan: u64 = 0;

    for op_id in resolved_ops {
        let spec_op = instance.operation_by_indices(op_id.job.0, op_id.op_index.0)?;
        let ready = job_ready_time[op_id.job.0];
        let timeline = &mut machine_timelines[spec_op.machine.0];

        let (start_time, completion_time, insert_idx) = if allow_gap_insertion {
            find_earliest_gap_slot(timeline, ready, spec_op.processing_time)?
        } else {
            find_append_slot(timeline, ready, spec_op.processing_time)?
        };

        let scheduled = ScheduledOperation {
            op_id,
            machine: spec_op.machine,
            start_time,
            completion_time,
            processing_time: spec_op.processing_time,
        };

        timeline.insert(insert_idx, scheduled);
        job_ready_time[op_id.job.0] = completion_time;
        makespan = makespan.max(completion_time);

        let flat_idx = op_id.flat_index(num_jobs, num_machines)?;
        scheduled_by_flat_id[flat_idx] = Some(scheduled);
    }

    let mut operations = Vec::with_capacity(num_ops);
    for (flat_idx, slot) in scheduled_by_flat_id.into_iter().enumerate() {
        match slot {
            Some(op) => operations.push(op),
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
