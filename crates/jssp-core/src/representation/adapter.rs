//! Canonical OBR <-> Machine-Sequence adapter and round-trip verification (`D01-B`).
//!
//! Converts a validated acyclic [`DisjunctiveGraph`] (with M4 [`CriticalPathAnalysis`] head
//! times `r_o = head[o]`) into a canonical [`ObrChromosome`] via deterministic sorting on
//! `(r_o, r_o + p_o, job_id, operation_index)`, decodes through the authoritative M2 Serial SGS
//! earliest feasible idle-gap insertion decoder ([`decode_obr`]), and quantifies exactness,
//! machine-order inversions, mismatched adjacent arcs, left/right start shifts, and signed
//! makespan changes via [`AdapterAuditReport`].

use sha2::{Digest, Sha256};

use crate::decoder::decode_obr;
use crate::error::JsspError;
use crate::feasibility::validate_schedule;
use crate::graph::{
    CriticalPathAnalysis, DisjunctiveGraph, GraphVertexKind, analyze_disjunctive_graph,
};
use crate::model::{JsspInstance, MachineSequence, OperationId, Schedule, ScheduledOperation};
use crate::objective::evaluate_makespan;
use crate::representation::ObrChromosome;

/// Typed classification of an OBR <-> machine-sequence adapter round-trip audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoundTripStatus {
    /// Valid OBR, identical machine sequences, identical operation start/completion times, and
    /// identical makespan.
    Exact,
    /// Valid OBR and identical machine sequences, while operation start/completion times shifted.
    StartTimesChanged,
    /// Machine sequences changed without altering decoded operation start/completion times.
    MachineOrderChanged,
    /// Both machine sequences and operation start/completion times changed (for example due to
    /// Serial SGS earliest idle-gap insertion left-shifting an operation earlier on a machine).
    MachineOrderAndStartTimesChanged,
    /// Target graph or generated OBR violated required structural consistency invariants.
    InvalidTargetGraph,
}

impl RoundTripStatus {
    /// Canonical lowercase serialization identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::StartTimesChanged => "start_times_changed",
            Self::MachineOrderChanged => "machine_order_changed",
            Self::MachineOrderAndStartTimesChanged => "machine_order_and_start_times_changed",
            Self::InvalidTargetGraph => "invalid_target_graph",
        }
    }

    /// Returns `true` iff every machine sequence, start time, completion time, and makespan
    /// round-tripped identically.
    #[must_use]
    pub const fn is_exact(self) -> bool {
        matches!(self, Self::Exact)
    }
}

/// Canonical `D01-B` operation sorting key `(head, head + processing_time, job_id, op_index)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalOperationKey {
    /// Target graph head time `r_o = head[o]`.
    pub head: u64,
    /// Target semi-active completion time `r_o + p_o`.
    pub completion: u64,
    /// Owning job identifier `j`.
    pub job: usize,
    /// Intra-job operation index `k`.
    pub op_index: usize,
}

/// Validated output of encoding a target disjunctive graph into a canonical OBR chromosome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedObrResult {
    /// Generated OBR chromosome containing sorted job IDs.
    pub chromosome: ObrChromosome,
    /// Operations sorted deterministically by [`CanonicalOperationKey`].
    pub canonical_operations: Vec<OperationId>,
    /// Exact sorting keys aligned position-for-position with `canonical_operations`.
    pub operation_keys: Vec<CanonicalOperationKey>,
    /// Generated chromosome length `L = J * M`.
    pub chromosome_length: usize,
    /// Observed count per job ID in `0..J`.
    pub job_multiplicities: Vec<usize>,
    /// `true` iff each job's operations appear strictly in order `k = 0 .. M - 1`.
    pub technological_order_preserved: bool,
    /// `true` iff M2 [`ObrChromosome::validate`] passed.
    pub obr_valid: bool,
}

/// Quantitative audit comparing a target machine-sequence graph against the M2 Serial SGS
/// decoding of its canonically encoded OBR chromosome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterAuditReport {
    /// Lowercase 64-hex SHA-256 digest of the target graph structure and machine sequences.
    pub target_graph_hash: String,
    /// Lowercase 64-hex SHA-256 digest of the generated OBR chromosome.
    pub generated_obr_hash: String,
    /// Lowercase 64-hex SHA-256 digest of the canonically decoded schedule.
    pub decoded_schedule_hash: String,
    /// Makespan of the target disjunctive graph `C_max(D)`.
    pub target_graph_makespan: u64,
    /// Makespan of the decoded schedule `C_max(S')`.
    pub decoded_schedule_makespan: u64,
    /// Overflow-safe signed difference `decoded_schedule_makespan - target_graph_makespan`.
    pub signed_makespan_diff: i128,
    /// Operations whose decoded start time is strictly earlier than `target_head` (`S'_o < r_o`).
    pub adapter_left_shift_count: usize,
    /// Operations whose decoded start time is strictly later than `target_head` (`S'_o > r_o`).
    pub adapter_right_shift_count: usize,
    /// Operations whose decoded start time equals `target_head` (`S'_o == r_o`).
    pub adapter_unchanged_start_count: usize,
    /// Sum across all machines of pairwise Kendall order inversions relative to target sequences.
    pub machine_order_inversion_count: usize,
    /// Sum across all machines of consecutive target `(u_i, u_{i+1})` arcs absent in decoded order.
    pub mismatched_machine_arc_count: usize,
    /// `true` iff every machine sequence matches the target machine sequence identically.
    pub exact_machine_order_preserved: bool,
    /// `true` iff every operation has `decoded_start == target_head`.
    pub exact_start_times_preserved: bool,
    /// `true` iff every operation has `decoded_completion == target_head + processing_time`.
    pub exact_completion_times_preserved: bool,
    /// `true` iff the generated OBR passed M2 chromosome validation.
    pub obr_valid: bool,
    /// `true` iff all within-job operations appear in technological precedence order.
    pub technological_order_preserved: bool,
    /// Overall typed round-trip status.
    pub status: RoundTripStatus,
    /// Total operation count `L = J * M`.
    pub total_operation_count: usize,
}

/// Complete output bundle of an end-to-end OBR <-> Machine-Sequence adapter round-trip call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterRoundTripBundle {
    pub target_graph: DisjunctiveGraph,
    pub target_analysis: CriticalPathAnalysis,
    pub encoded: EncodedObrResult,
    pub decoded_schedule: Schedule,
    pub audit: AdapterAuditReport,
}

/// Encodes an acyclic [`DisjunctiveGraph`] and its M4 [`CriticalPathAnalysis`] into a canonical
/// [`ObrChromosome`] using the `D01-B` sorting key `(head, head + p, job_id, op_index)`.
///
/// # Errors
/// Returns [`JsspError`] if dimensions mismatch, `head + processing_time` overflows `u64`,
/// technological job order fails, or M2 OBR validation fails.
pub fn encode_graph_to_obr(
    instance: &JsspInstance,
    graph: &DisjunctiveGraph,
    analysis: &CriticalPathAnalysis,
) -> Result<EncodedObrResult, JsspError> {
    let meta = graph.metadata();
    if meta.num_jobs != instance.num_jobs()
        || meta.num_machines != instance.num_machines()
        || meta.num_operations != instance.num_operations()
        || analysis.operations().len() != instance.num_operations()
    {
        return Err(JsspError::InvalidDisjunctiveGraph {
            detail: format!(
                "graph dimensions ({}x{}, {} ops) do not match instance ({}x{}, {} ops)",
                meta.num_jobs,
                meta.num_machines,
                analysis.operations().len(),
                instance.num_jobs(),
                instance.num_machines(),
                instance.num_operations(),
            ),
        });
    }

    let mut keyed: Vec<(CanonicalOperationKey, OperationId)> =
        Vec::with_capacity(analysis.operations().len());

    for op_crit in analysis.operations() {
        let spec = instance.operation_by_indices(op_crit.op_id.job.0, op_crit.op_id.op_index.0)?;
        if spec.machine != op_crit.machine {
            return Err(JsspError::OperationMachineMismatch {
                job: op_crit.op_id.job.0,
                op_index: op_crit.op_id.op_index.0,
                expected_machine: spec.machine.0,
                actual_machine: op_crit.machine.0,
            });
        }
        if spec.processing_time != op_crit.processing_time {
            return Err(JsspError::OperationProcessingTimeMismatch {
                job: op_crit.op_id.job.0,
                op_index: op_crit.op_id.op_index.0,
                expected_duration: spec.processing_time,
                actual_duration: op_crit.processing_time,
            });
        }

        let completion = op_crit
            .head
            .checked_add(op_crit.processing_time)
            .ok_or(JsspError::TimeOverflow)?;
        let key = CanonicalOperationKey {
            head: op_crit.head,
            completion,
            job: op_crit.op_id.job.0,
            op_index: op_crit.op_id.op_index.0,
        };
        keyed.push((key, op_crit.op_id));
    }

    keyed.sort_by_key(|(key, _)| *key);

    let mut canonical_operations = Vec::with_capacity(keyed.len());
    let mut operation_keys = Vec::with_capacity(keyed.len());
    let mut genes = Vec::with_capacity(keyed.len());
    let mut next_op_for_job = vec![0usize; instance.num_jobs()];
    let mut technological_order_preserved = true;

    for (key, op_id) in keyed {
        if next_op_for_job[op_id.job.0] != op_id.op_index.0 {
            technological_order_preserved = false;
        }
        next_op_for_job[op_id.job.0] = op_id.op_index.0 + 1;
        canonical_operations.push(op_id);
        operation_keys.push(key);
        genes.push(op_id.job);
    }

    if !technological_order_preserved {
        return Err(JsspError::InvalidDisjunctiveGraph {
            detail: "canonical sorting did not preserve within-job technological operation order"
                .to_owned(),
        });
    }

    let chromosome = ObrChromosome::new(genes);
    chromosome.validate(instance)?;
    let job_multiplicities = chromosome.job_counts(instance.num_jobs())?;
    let resolved = chromosome.resolve_operations(instance)?;
    if resolved != canonical_operations {
        return Err(JsspError::InvalidDisjunctiveGraph {
            detail: "OBR operation resolution diverged from sorted canonical operation sequence"
                .to_owned(),
        });
    }

    Ok(EncodedObrResult {
        chromosome_length: chromosome.len(),
        chromosome,
        canonical_operations,
        operation_keys,
        job_multiplicities,
        technological_order_preserved,
        obr_valid: true,
    })
}

/// Decodes `encoded.chromosome` through the canonical M2 Serial SGS decoder ([`decode_obr`]) and
/// produces a quantitative [`AdapterAuditReport`] against `target_graph` and `target_analysis`.
///
/// # Errors
/// Returns [`JsspError`] if M2 decoding, independent schedule validation, makespan evaluation,
/// or checked integer arithmetic fails.
pub fn audit_obr_against_target_graph(
    instance: &JsspInstance,
    target_graph: &DisjunctiveGraph,
    target_analysis: &CriticalPathAnalysis,
    encoded: &EncodedObrResult,
) -> Result<(Schedule, AdapterAuditReport), JsspError> {
    let meta = target_graph.metadata();
    let num_jobs = instance.num_jobs();
    let num_machines = instance.num_machines();
    let total_ops = instance.num_operations();

    if meta.num_jobs != num_jobs
        || meta.num_machines != num_machines
        || meta.num_operations != total_ops
        || target_analysis.operations().len() != total_ops
    {
        return Err(JsspError::InvalidDisjunctiveGraph {
            detail: "target graph/analysis dimensions do not match instance".to_owned(),
        });
    }

    let decoded_schedule = decode_obr(instance, &encoded.chromosome)?;
    validate_schedule(instance, &decoded_schedule)?;
    let recomputed_makespan = evaluate_makespan(instance, &decoded_schedule)?;
    if recomputed_makespan != decoded_schedule.makespan() {
        return Err(JsspError::MakespanMismatch {
            reported: decoded_schedule.makespan(),
            recomputed: recomputed_makespan,
        });
    }

    let target_seqs = target_graph.machine_sequences();
    let decoded_seqs = decoded_schedule.machine_sequences();

    let mut mismatched_machine_arc_count = 0usize;
    let mut machine_order_inversion_count = 0usize;
    let mut exact_machine_order_preserved = target_seqs.len() == decoded_seqs.len();

    for (t_seq, d_seq) in target_seqs.iter().zip(decoded_seqs.iter()) {
        if t_seq != d_seq {
            exact_machine_order_preserved = false;
        }
        mismatched_machine_arc_count +=
            count_mismatched_adjacent_arcs(&t_seq.operations, &d_seq.operations);
        machine_order_inversion_count += count_pairwise_inversions(
            &t_seq.operations,
            &d_seq.operations,
            num_jobs,
            num_machines,
        )?;
    }

    let mut decoded_by_flat: Vec<Option<ScheduledOperation>> = vec![None; total_ops];
    for &op in decoded_schedule.operations() {
        let flat = op.op_id.flat_index(num_jobs, num_machines)?;
        decoded_by_flat[flat] = Some(op);
    }

    let mut adapter_left_shift_count = 0usize;
    let mut adapter_right_shift_count = 0usize;
    let mut adapter_unchanged_start_count = 0usize;
    let mut exact_completion_times_preserved = true;

    for target_op in target_analysis.operations() {
        let flat = target_op.op_id.flat_index(num_jobs, num_machines)?;
        let decoded_op = decoded_by_flat[flat].ok_or(JsspError::MissingOperation {
            job: target_op.op_id.job.0,
            op_index: target_op.op_id.op_index.0,
        })?;

        match decoded_op.start_time.cmp(&target_op.head) {
            std::cmp::Ordering::Less => adapter_left_shift_count += 1,
            std::cmp::Ordering::Greater => adapter_right_shift_count += 1,
            std::cmp::Ordering::Equal => adapter_unchanged_start_count += 1,
        }

        let target_completion = target_op
            .head
            .checked_add(target_op.processing_time)
            .ok_or(JsspError::TimeOverflow)?;
        if decoded_op.completion_time != target_completion {
            exact_completion_times_preserved = false;
        }
    }

    let exact_start_times_preserved =
        adapter_left_shift_count == 0 && adapter_right_shift_count == 0;

    let target_graph_makespan = target_analysis.graph_makespan();
    let decoded_schedule_makespan = decoded_schedule.makespan();
    let signed_makespan_diff =
        i128::from(decoded_schedule_makespan) - i128::from(target_graph_makespan);

    let status = if !encoded.obr_valid || !encoded.technological_order_preserved {
        RoundTripStatus::InvalidTargetGraph
    } else if exact_machine_order_preserved
        && exact_start_times_preserved
        && exact_completion_times_preserved
        && decoded_schedule_makespan == target_graph_makespan
    {
        RoundTripStatus::Exact
    } else if exact_machine_order_preserved {
        RoundTripStatus::StartTimesChanged
    } else if exact_start_times_preserved && exact_completion_times_preserved {
        RoundTripStatus::MachineOrderChanged
    } else {
        RoundTripStatus::MachineOrderAndStartTimesChanged
    };

    let target_graph_hash = hash_target_graph(target_graph);
    let generated_obr_hash = hash_obr_chromosome(&encoded.chromosome);
    let decoded_schedule_hash = hash_schedule(&decoded_schedule)?;

    let report = AdapterAuditReport {
        target_graph_hash,
        generated_obr_hash,
        decoded_schedule_hash,
        target_graph_makespan,
        decoded_schedule_makespan,
        signed_makespan_diff,
        adapter_left_shift_count,
        adapter_right_shift_count,
        adapter_unchanged_start_count,
        machine_order_inversion_count,
        mismatched_machine_arc_count,
        exact_machine_order_preserved,
        exact_start_times_preserved,
        exact_completion_times_preserved,
        obr_valid: encoded.obr_valid,
        technological_order_preserved: encoded.technological_order_preserved,
        status,
        total_operation_count: total_ops,
    };

    Ok((decoded_schedule, report))
}

/// Validates target per-machine sequences, builds an acyclic disjunctive graph, runs M4
/// head/tail analysis, encodes to canonical OBR (`D01-B`), decodes via M2 Serial SGS, and
/// returns the full round-trip audit bundle.
///
/// # Errors
/// Returns [`JsspError`] if machine sequences are invalid, cyclic, or any downstream step fails.
pub fn round_trip_machine_sequences(
    instance: &JsspInstance,
    machine_sequences: &[MachineSequence],
) -> Result<AdapterRoundTripBundle, JsspError> {
    let target_graph =
        DisjunctiveGraph::from_target_machine_sequences(instance, machine_sequences)?;
    let target_analysis = analyze_disjunctive_graph(&target_graph)?;
    let encoded = encode_graph_to_obr(instance, &target_graph, &target_analysis)?;
    let (decoded_schedule, audit) =
        audit_obr_against_target_graph(instance, &target_graph, &target_analysis, &encoded)?;

    Ok(AdapterRoundTripBundle {
        target_graph,
        target_analysis,
        encoded,
        decoded_schedule,
        audit,
    })
}

/// Validates a schedule, extracts its oriented disjunctive graph and head/tail times, re-encodes
/// via `D01-B`, decodes through canonical M2 Serial SGS, and returns the audit bundle.
///
/// # Errors
/// Returns [`JsspError`] if `schedule` validation, graph analysis, encoding, or decoding fails.
pub fn round_trip_schedule(
    instance: &JsspInstance,
    schedule: &Schedule,
) -> Result<AdapterRoundTripBundle, JsspError> {
    let target_graph = DisjunctiveGraph::from_schedule(instance, schedule)?;
    let target_analysis = analyze_disjunctive_graph(&target_graph)?;
    let encoded = encode_graph_to_obr(instance, &target_graph, &target_analysis)?;
    let (decoded_schedule, audit) =
        audit_obr_against_target_graph(instance, &target_graph, &target_analysis, &encoded)?;

    Ok(AdapterRoundTripBundle {
        target_graph,
        target_analysis,
        encoded,
        decoded_schedule,
        audit,
    })
}

fn count_mismatched_adjacent_arcs(target: &[OperationId], decoded: &[OperationId]) -> usize {
    target
        .windows(2)
        .filter(|pair| !decoded.windows(2).any(|d_pair| d_pair == *pair))
        .count()
}

fn count_pairwise_inversions(
    target: &[OperationId],
    decoded: &[OperationId],
    num_jobs: usize,
    num_machines: usize,
) -> Result<usize, JsspError> {
    let total_ops = num_jobs
        .checked_mul(num_machines)
        .ok_or(JsspError::DimensionOverflow)?;
    let mut target_rank = vec![usize::MAX; total_ops];
    for (rank, &op_id) in target.iter().enumerate() {
        let flat = op_id.flat_index(num_jobs, num_machines)?;
        target_rank[flat] = rank;
    }

    let mut ranks = Vec::with_capacity(decoded.len());
    for &op_id in decoded {
        let flat = op_id.flat_index(num_jobs, num_machines)?;
        ranks.push(target_rank[flat]);
    }

    let mut inversions = 0usize;
    for i in 0..ranks.len() {
        for j in (i + 1)..ranks.len() {
            if ranks[i] > ranks[j] {
                inversions += 1;
            }
        }
    }
    Ok(inversions)
}

fn update_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_le_bytes());
}

fn update_usize(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn digest_to_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

fn hash_target_graph(graph: &DisjunctiveGraph) -> String {
    let mut hasher = Sha256::new();
    let meta = graph.metadata();
    hasher.update(b"jssp-target-graph-v1");
    update_usize(&mut hasher, meta.num_jobs);
    update_usize(&mut hasher, meta.num_machines);
    update_usize(&mut hasher, meta.num_operations);
    update_u64(&mut hasher, meta.schedule_makespan);

    for v in graph.vertices() {
        update_usize(&mut hasher, v.id.0);
        match v.kind {
            GraphVertexKind::Source => {
                hasher.update([0]);
                update_usize(&mut hasher, 0);
                update_usize(&mut hasher, 0);
            }
            GraphVertexKind::Operation(op) => {
                hasher.update([1]);
                update_usize(&mut hasher, op.job.0);
                update_usize(&mut hasher, op.op_index.0);
            }
            GraphVertexKind::Sink => {
                hasher.update([2]);
                update_usize(&mut hasher, 0);
                update_usize(&mut hasher, 0);
            }
        }
        update_usize(&mut hasher, v.machine.map_or(usize::MAX, |m| m.0));
        update_u64(&mut hasher, v.processing_time);
    }

    for a in graph.arcs() {
        update_usize(&mut hasher, a.id.0);
        update_usize(&mut hasher, a.from.0);
        update_usize(&mut hasher, a.to.0);
        hasher.update(a.arc_type.kind_str().as_bytes());
        update_usize(&mut hasher, a.arc_type.owner_index().unwrap_or(usize::MAX));
    }

    for seq in graph.machine_sequences() {
        update_usize(&mut hasher, seq.machine.0);
        update_usize(&mut hasher, seq.operations.len());
        for &op in &seq.operations {
            update_usize(&mut hasher, op.job.0);
            update_usize(&mut hasher, op.op_index.0);
        }
    }

    digest_to_hex(&hasher.finalize())
}

fn hash_obr_chromosome(chromosome: &ObrChromosome) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"jssp-obr-v1");
    update_usize(&mut hasher, chromosome.len());
    for gene in chromosome.genes() {
        update_usize(&mut hasher, gene.0);
    }
    digest_to_hex(&hasher.finalize())
}

fn hash_schedule(schedule: &Schedule) -> Result<String, JsspError> {
    let mut hasher = Sha256::new();
    let num_jobs = schedule.num_jobs();
    let num_machines = schedule.num_machines();
    let total_ops = num_jobs
        .checked_mul(num_machines)
        .ok_or(JsspError::DimensionOverflow)?;

    let mut canonical: Vec<Option<ScheduledOperation>> = vec![None; total_ops];
    for &op in schedule.operations() {
        let flat = op.op_id.flat_index(num_jobs, num_machines)?;
        canonical[flat] = Some(op);
    }

    hasher.update(b"jssp-schedule-v1");
    update_usize(&mut hasher, num_jobs);
    update_usize(&mut hasher, num_machines);
    update_u64(&mut hasher, schedule.makespan());

    for entry in canonical.into_iter().flatten() {
        update_usize(&mut hasher, entry.op_id.job.0);
        update_usize(&mut hasher, entry.op_id.op_index.0);
        update_usize(&mut hasher, entry.machine.0);
        update_u64(&mut hasher, entry.start_time);
        update_u64(&mut hasher, entry.completion_time);
        update_u64(&mut hasher, entry.processing_time);
    }

    Ok(digest_to_hex(&hasher.finalize()))
}
