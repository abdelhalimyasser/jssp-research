//! Independent schedule feasibility verification tests.

use jssp_core::decoder::decode_obr;
use jssp_core::error::JsspError;
use jssp_core::feasibility::validate_schedule;
use jssp_core::model::{
    JobId, JsspInstance, MachineId, OperationId, OperationIndex, Schedule, ScheduledOperation,
};
use jssp_core::representation::ObrChromosome;

fn fixture_2x2() -> JsspInstance {
    // J0: (M0, 3) -> (M1, 2)
    // J1: (M1, 4) -> (M0, 1)
    JsspInstance::new(vec![vec![(0, 3), (1, 2)], vec![(1, 4), (0, 1)]]).expect("valid 2x2")
}

fn valid_2x2_ops() -> Vec<ScheduledOperation> {
    // Back-to-back exact handoff on J0 (t=3), M0 (t=3? let's make M0 [0,3] and [3,4]!), M1 [0,4] and [4,6]
    vec![
        ScheduledOperation {
            op_id: OperationId::from_indices(0, 0),
            machine: MachineId(0),
            start_time: 0,
            completion_time: 3,
            processing_time: 3,
        },
        ScheduledOperation {
            op_id: OperationId::from_indices(0, 1),
            machine: MachineId(1),
            start_time: 4,
            completion_time: 6,
            processing_time: 2,
        },
        ScheduledOperation {
            op_id: OperationId::from_indices(1, 0),
            machine: MachineId(1),
            start_time: 0,
            completion_time: 4,
            processing_time: 4,
        },
        ScheduledOperation {
            op_id: OperationId::from_indices(1, 1),
            machine: MachineId(0),
            start_time: 4,
            completion_time: 5,
            processing_time: 1,
        },
    ]
}

#[test]
fn valid_and_back_to_back_schedules_pass_feasibility() {
    let instance = fixture_2x2();

    // Standard decoded schedule passes
    let decoded =
        decode_obr(&instance, &ObrChromosome::from_job_indices(&[0, 1, 0, 1])).expect("decode");
    assert!(validate_schedule(&instance, &decoded).is_ok());

    // Hand-constructed back-to-back schedule (C_a == S_b on M1 at t=4) passes
    let manual = Schedule::new(2, 2, valid_2x2_ops(), 6);
    assert!(validate_schedule(&instance, &manual).is_ok());
}

#[test]
fn detects_job_precedence_violation() {
    let instance = fixture_2x2();
    let mut ops = valid_2x2_ops();
    // Move O(0, 1) to start at t=2 before O(0, 0) finishes at t=3 (put O(1, 0) later so M1 stays exclusive)
    ops[1].start_time = 2;
    ops[1].completion_time = 4;
    ops[2].start_time = 4;
    ops[2].completion_time = 8;
    ops[3].start_time = 8;
    ops[3].completion_time = 9;

    let schedule = Schedule::new(2, 2, ops, 9);
    assert_eq!(
        validate_schedule(&instance, &schedule),
        Err(JsspError::JobPrecedenceViolation {
            job: 0,
            prev_op_index: 0,
            next_op_index: 1,
            prev_completion: 3,
            next_start: 2,
        })
    );
}

#[test]
fn detects_machine_overlap_variants() {
    let instance = fixture_2x2();

    // Partial overlap on M1: [0, 4) vs [3, 5)
    let mut partial = valid_2x2_ops();
    partial[1].start_time = 3;
    partial[1].completion_time = 5;
    assert_eq!(
        validate_schedule(&instance, &Schedule::new(2, 2, partial, 5)),
        Err(JsspError::MachineOverlapViolation {
            machine: 1,
            op_a: (1, 0),
            interval_a: (0, 4),
            op_b: (0, 1),
            interval_b: (3, 5),
        })
    );

    // Containment on M1: [0, 4) contains [1, 3) (keeping O(0,0) -> O(0,1) precedence valid with C_{0,0}=1? No, use 3x1 or shift O(0,0)? Keep valid precedence by shifting both M1 ops: [10, 14) and [11, 13))
    let mut containment = valid_2x2_ops();
    containment[2].start_time = 10;
    containment[2].completion_time = 14;
    containment[3].start_time = 14;
    containment[3].completion_time = 15;
    containment[1].start_time = 11;
    containment[1].completion_time = 13;
    assert_eq!(
        validate_schedule(&instance, &Schedule::new(2, 2, containment, 15)),
        Err(JsspError::MachineOverlapViolation {
            machine: 1,
            op_a: (1, 0),
            interval_a: (10, 14),
            op_b: (0, 1),
            interval_b: (11, 13),
        })
    );
}

#[test]
fn detects_missing_duplicate_unexpected_metadata_duration_makespan_and_overflow_errors() {
    let instance = fixture_2x2();

    // Empty schedule
    assert_eq!(
        validate_schedule(&instance, &Schedule::new(2, 2, vec![], 0)),
        Err(JsspError::EmptySchedule)
    );

    // Missing operation (1, 1)
    let mut missing = valid_2x2_ops();
    missing.pop();
    assert_eq!(
        validate_schedule(&instance, &Schedule::new(2, 2, missing, 6)),
        Err(JsspError::MissingOperation {
            job: 1,
            op_index: 1
        })
    );

    // Duplicate operation (0, 0)
    let mut duplicate = valid_2x2_ops();
    duplicate[3].op_id = OperationId::new(JobId(0), OperationIndex(0));
    assert_eq!(
        validate_schedule(&instance, &Schedule::new(2, 2, duplicate, 6)),
        Err(JsspError::DuplicateOperation {
            job: 0,
            op_index: 0
        })
    );

    // Unexpected out-of-bounds operation (2, 0)
    let mut unexpected = valid_2x2_ops();
    unexpected[0].op_id = OperationId::from_indices(2, 0);
    assert_eq!(
        validate_schedule(&instance, &Schedule::new(2, 2, unexpected, 6)),
        Err(JsspError::UnexpectedOperation {
            job: 2,
            op_index: 0,
            num_jobs: 2,
            num_machines: 2,
        })
    );

    // Wrong machine assignment
    let mut wrong_machine = valid_2x2_ops();
    wrong_machine[0].machine = MachineId(1);
    assert_eq!(
        validate_schedule(&instance, &Schedule::new(2, 2, wrong_machine, 6)),
        Err(JsspError::OperationMachineMismatch {
            job: 0,
            op_index: 0,
            expected_machine: 0,
            actual_machine: 1,
        })
    );

    // Wrong processing_time field
    let mut wrong_proc_meta = valid_2x2_ops();
    wrong_proc_meta[0].processing_time = 99;
    assert_eq!(
        validate_schedule(&instance, &Schedule::new(2, 2, wrong_proc_meta, 6)),
        Err(JsspError::OperationProcessingTimeMismatch {
            job: 0,
            op_index: 0,
            expected_duration: 3,
            actual_duration: 99,
        })
    );

    // Wrong interval completion time (C != S + p)
    let mut wrong_end = valid_2x2_ops();
    wrong_end[0].completion_time = 4;
    assert_eq!(
        validate_schedule(&instance, &Schedule::new(2, 2, wrong_end, 6)),
        Err(JsspError::ProcessingDurationMismatch {
            job: 0,
            op_index: 0,
            start_time: 0,
            completion_time: 4,
            expected_duration: 3,
        })
    );

    // Wrong reported makespan
    assert_eq!(
        validate_schedule(&instance, &Schedule::new(2, 2, valid_2x2_ops(), 999)),
        Err(JsspError::MakespanMismatch {
            reported: 999,
            recomputed: 6,
        })
    );

    // Overflow in S + p
    let mut overflow_ops = valid_2x2_ops();
    overflow_ops[3].start_time = u64::MAX;
    overflow_ops[3].completion_time = u64::MAX;
    assert_eq!(
        validate_schedule(&instance, &Schedule::new(2, 2, overflow_ops, u64::MAX)),
        Err(JsspError::TimeOverflow)
    );
}
