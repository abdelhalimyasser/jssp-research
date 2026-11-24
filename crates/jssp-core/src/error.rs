//! Typed domain error definitions for `jssp-core`.

use std::error::Error;
use std::fmt;

/// Typed error returned by JSSP instance validation, OBR representation checks,
/// schedule decoding, feasibility validation, and makespan evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsspError {
    /// Instance must contain at least one job.
    EmptyJobs,
    /// Instance must contain at least one machine.
    EmptyMachines,
    /// Computing total operation count (`num_jobs * num_machines`) overflowed `usize`.
    DimensionOverflow,
    /// A job row does not match the instance's rectangular machine count.
    NonRectangularJobLength {
        job: usize,
        expected: usize,
        actual: usize,
    },
    /// An operation references a machine index outside `0..num_machines`.
    InvalidMachineId {
        job: usize,
        op_index: usize,
        machine: usize,
        num_machines: usize,
    },
    /// A job visits the same machine more than once (violating classical JSSP permutation routing).
    DuplicateMachineInJob { job: usize, machine: usize },
    /// Classical JSSP operations must have strictly positive integer processing times (`p > 0`).
    NonPositiveProcessingTime {
        job: usize,
        op_index: usize,
        duration: u64,
    },
    /// Checked integer time arithmetic (`u64`) overflowed.
    TimeOverflow,
    /// OBR chromosome length does not equal `num_jobs * num_machines`.
    InvalidChromosomeLength { expected: usize, actual: usize },
    /// OBR chromosome contains a job ID outside `0..num_jobs`.
    InvalidJobId {
        position: usize,
        job: usize,
        num_jobs: usize,
    },
    /// A job does not appear exactly `num_machines` times in the OBR chromosome.
    InvalidJobMultiplicity {
        job: usize,
        expected: usize,
        actual: usize,
    },
    /// Schedule contains no operations.
    EmptySchedule,
    /// Required operation `(job, op_index)` is absent from the schedule.
    MissingOperation { job: usize, op_index: usize },
    /// Operation `(job, op_index)` appears more than once in the schedule.
    DuplicateOperation { job: usize, op_index: usize },
    /// Operation identifier `(job, op_index)` is outside instance bounds.
    UnexpectedOperation {
        job: usize,
        op_index: usize,
        num_jobs: usize,
        num_machines: usize,
    },
    /// Scheduled operation machine does not match the instance specification.
    OperationMachineMismatch {
        job: usize,
        op_index: usize,
        expected_machine: usize,
        actual_machine: usize,
    },
    /// Scheduled operation `processing_time` field does not match the instance specification.
    OperationProcessingTimeMismatch {
        job: usize,
        op_index: usize,
        expected_duration: u64,
        actual_duration: u64,
    },
    /// Scheduled `completion_time` does not equal `start_time + processing_time`.
    ProcessingDurationMismatch {
        job: usize,
        op_index: usize,
        start_time: u64,
        completion_time: u64,
        expected_duration: u64,
    },
    /// Job precedence constraint violated (`C_{j,k} > S_{j,k+1}`).
    JobPrecedenceViolation {
        job: usize,
        prev_op_index: usize,
        next_op_index: usize,
        prev_completion: u64,
        next_start: u64,
    },
    /// Two distinct operations overlap on the same machine.
    MachineOverlapViolation {
        machine: usize,
        op_a: (usize, usize),
        interval_a: (u64, u64),
        op_b: (usize, usize),
        interval_b: (u64, u64),
    },
    /// Reported schedule makespan differs from the recomputed maximum completion time.
    MakespanMismatch { reported: u64, recomputed: u64 },
    /// Replicate index is outside the frozen seed schedule bounds `0..max_replicates`.
    InvalidReplicateIndex {
        replicate_index: u32,
        max_replicates: u32,
    },
    /// Named RNG stream identifier does not match one of the nine approved D07 roles.
    UnknownRngStreamRole { name: String },
    /// Half-open uniform random range `[0, upper)` requires `upper > 0`.
    InvalidRandomBound { upper: u64 },
    /// Requested sample-without-replacement count exceeds population size.
    InvalidSampleSize {
        population_len: usize,
        requested: usize,
    },
    /// Invalid pairing policy identifier or hexadecimal seed payload passed to RNG helpers.
    InvalidRngConfiguration { detail: String },
    /// Topological sorting detected a directed cycle in the disjunctive graph.
    GraphCycleDetected {
        unvisited_vertices: usize,
        cycle_operations: Vec<(usize, usize)>,
    },
    /// Structural disjunctive-graph input violated invariants.
    InvalidDisjunctiveGraph { detail: String },
    /// Longest-path makespan on the disjunctive graph disagrees with schedule makespan.
    GraphMakespanMismatch {
        schedule_makespan: u64,
        graph_makespan: u64,
    },
    /// Critical-path consistency failure: `head[u] + p[u] + tail[u] > C_max`.
    NegativeOperationSlack {
        job: usize,
        op_index: usize,
        head_plus_proc_plus_tail: u64,
        graph_makespan: u64,
    },
}

impl fmt::Display for JsspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyJobs => write!(f, "JSSP instance must contain at least one job"),
            Self::EmptyMachines => write!(f, "JSSP instance must contain at least one machine"),
            Self::DimensionOverflow => {
                write!(f, "JSSP instance dimensions overflow usize multiplication")
            }
            Self::NonRectangularJobLength {
                job,
                expected,
                actual,
            } => write!(
                f,
                "job {job} has {actual} operations, expected {expected} for rectangular JSSP"
            ),
            Self::InvalidMachineId {
                job,
                op_index,
                machine,
                num_machines,
            } => write!(
                f,
                "operation ({job}, {op_index}) references machine {machine}, valid range is 0..{num_machines}"
            ),
            Self::DuplicateMachineInJob { job, machine } => {
                write!(
                    f,
                    "job {job} visits machine {machine} more than once; classical JSSP requires a machine permutation"
                )
            }
            Self::NonPositiveProcessingTime {
                job,
                op_index,
                duration,
            } => write!(
                f,
                "operation ({job}, {op_index}) has non-positive processing time {duration}"
            ),
            Self::TimeOverflow => write!(f, "checked u64 time arithmetic overflowed"),
            Self::InvalidChromosomeLength { expected, actual } => {
                write!(f, "OBR chromosome length is {actual}, expected {expected}")
            }
            Self::InvalidJobId {
                position,
                job,
                num_jobs,
            } => write!(
                f,
                "OBR gene at position {position} has invalid job ID {job}, valid range is 0..{num_jobs}"
            ),
            Self::InvalidJobMultiplicity {
                job,
                expected,
                actual,
            } => write!(
                f,
                "job {job} appears {actual} times in OBR chromosome, expected {expected}"
            ),
            Self::EmptySchedule => write!(f, "schedule contains no operations"),
            Self::MissingOperation { job, op_index } => {
                write!(
                    f,
                    "schedule is missing required operation ({job}, {op_index})"
                )
            }
            Self::DuplicateOperation { job, op_index } => {
                write!(
                    f,
                    "schedule contains duplicate operation ({job}, {op_index})"
                )
            }
            Self::UnexpectedOperation {
                job,
                op_index,
                num_jobs,
                num_machines,
            } => write!(
                f,
                "schedule contains out-of-bounds operation ({job}, {op_index}) for {num_jobs}x{num_machines} instance"
            ),
            Self::OperationMachineMismatch {
                job,
                op_index,
                expected_machine,
                actual_machine,
            } => write!(
                f,
                "operation ({job}, {op_index}) assigned to machine {actual_machine}, expected {expected_machine}"
            ),
            Self::OperationProcessingTimeMismatch {
                job,
                op_index,
                expected_duration,
                actual_duration,
            } => write!(
                f,
                "operation ({job}, {op_index}) reports processing time {actual_duration}, expected {expected_duration}"
            ),
            Self::ProcessingDurationMismatch {
                job,
                op_index,
                start_time,
                completion_time,
                expected_duration,
            } => write!(
                f,
                "operation ({job}, {op_index}) interval [{start_time}, {completion_time}) does not match processing time {expected_duration}"
            ),
            Self::JobPrecedenceViolation {
                job,
                prev_op_index,
                next_op_index,
                prev_completion,
                next_start,
            } => write!(
                f,
                "job {job} precedence violation: operation ({job}, {prev_op_index}) completes at {prev_completion} after operation ({job}, {next_op_index}) starts at {next_start}"
            ),
            Self::MachineOverlapViolation {
                machine,
                op_a,
                interval_a,
                op_b,
                interval_b,
            } => write!(
                f,
                "machine {machine} overlap violation between ({}, {}) [{}, {}) and ({}, {}) [{}, {})",
                op_a.0,
                op_a.1,
                interval_a.0,
                interval_a.1,
                op_b.0,
                op_b.1,
                interval_b.0,
                interval_b.1
            ),
            Self::MakespanMismatch {
                reported,
                recomputed,
            } => write!(
                f,
                "schedule reports makespan {reported}, but recomputed maximum completion time is {recomputed}"
            ),
            Self::InvalidReplicateIndex {
                replicate_index,
                max_replicates,
            } => write!(
                f,
                "replicate index {replicate_index} is out of bounds for frozen schedule 0..{max_replicates}"
            ),
            Self::UnknownRngStreamRole { name } => {
                write!(f, "unknown named RNG stream role '{name}'")
            }
            Self::InvalidRandomBound { upper } => {
                write!(
                    f,
                    "uniform integer sampling bound must be positive, got {upper}"
                )
            }
            Self::InvalidSampleSize {
                population_len,
                requested,
            } => write!(
                f,
                "cannot sample {requested} items without replacement from population of size {population_len}"
            ),
            Self::InvalidRngConfiguration { detail } => {
                write!(f, "invalid RNG configuration: {detail}")
            }
            Self::GraphCycleDetected {
                unvisited_vertices,
                cycle_operations,
            } => {
                let cycle_str: Vec<String> = cycle_operations
                    .iter()
                    .map(|(j, k)| format!("O({j},{k})"))
                    .collect();
                write!(
                    f,
                    "disjunctive graph contains a directed cycle ({unvisited_vertices} unvisited vertices, cycle witness: [{}])",
                    cycle_str.join(" -> ")
                )
            }
            Self::InvalidDisjunctiveGraph { detail } => {
                write!(f, "invalid disjunctive graph: {detail}")
            }
            Self::GraphMakespanMismatch {
                schedule_makespan,
                graph_makespan,
            } => write!(
                f,
                "disjunctive graph makespan {graph_makespan} disagrees with schedule makespan {schedule_makespan}"
            ),
            Self::NegativeOperationSlack {
                job,
                op_index,
                head_plus_proc_plus_tail,
                graph_makespan,
            } => write!(
                f,
                "operation ({job}, {op_index}) has negative slack: head+p+tail={head_plus_proc_plus_tail} > C_max={graph_makespan}"
            ),
        }
    }
}

impl Error for JsspError {}
