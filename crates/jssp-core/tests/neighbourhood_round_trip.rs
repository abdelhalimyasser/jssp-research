//! OBR <-> Machine-Sequence adapter (`D01-B`) round-trip correctness, adversarial gap-insertion,
//! audit telemetry, and seeded property-based tests across Category A (M2 active schedules) and
//! Category B (semi-active append-only orientations) (Milestone M5).

use jssp_core::decoder::{decode_obr, decode_obr_append_only};
use jssp_core::error::JsspError;
use jssp_core::feasibility::validate_schedule;
use jssp_core::graph::{DisjunctiveGraph, analyze_disjunctive_graph};
use jssp_core::model::{JobId, JsspInstance, MachineId, MachineSequence, OperationId};
use jssp_core::representation::{
    CanonicalOperationKey, EncodedObrResult, ObrChromosome, RoundTripStatus,
    audit_obr_against_target_graph, encode_graph_to_obr, round_trip_machine_sequences,
    round_trip_schedule,
};

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.gen_range(i + 1);
            slice.swap(i, j);
        }
    }
}

fn generate_random_instance(
    rng: &mut SplitMix64,
    num_jobs: usize,
    num_machines: usize,
    max_duration: u64,
) -> JsspInstance {
    let mut raw_jobs = Vec::with_capacity(num_jobs);
    for _ in 0..num_jobs {
        let mut machines: Vec<usize> = (0..num_machines).collect();
        rng.shuffle(&mut machines);
        let row = machines
            .into_iter()
            .map(|m| (m, (rng.next_u64() % max_duration) + 1))
            .collect();
        raw_jobs.push(row);
    }
    JsspInstance::new(raw_jobs).expect("valid random instance")
}

fn generate_random_obr(
    rng: &mut SplitMix64,
    num_jobs: usize,
    num_machines: usize,
) -> ObrChromosome {
    let mut genes = Vec::with_capacity(num_jobs * num_machines);
    for j in 0..num_jobs {
        for _ in 0..num_machines {
            genes.push(JobId(j));
        }
    }
    rng.shuffle(&mut genes);
    ObrChromosome::new(genes)
}

fn xie_3x3_instance() -> JsspInstance {
    JsspInstance::new(vec![
        vec![(0, 2), (1, 1), (2, 3)],
        vec![(0, 1), (2, 2), (1, 2)],
        vec![(1, 5), (0, 2), (2, 1)],
    ])
    .expect("valid 3x3 instance")
}

fn multi_path_3x3_instance() -> JsspInstance {
    JsspInstance::new(vec![
        vec![(0, 4), (1, 2), (2, 4)],
        vec![(1, 4), (2, 1), (0, 1)],
        vec![(2, 2), (0, 3), (1, 1)],
    ])
    .expect("valid multi-path 3x3")
}

fn adversarial_2x2_instance() -> JsspInstance {
    // M5 Section 6 illustrative adversarial gap-insertion instance:
    // J0: (M1, 5) -> (M0, 4)
    // J1: (M0, 2) -> (M1, 3)
    JsspInstance::new(vec![vec![(1, 5), (0, 4)], vec![(0, 2), (1, 3)]]).expect("valid 2x2")
}

#[test]
fn m2_generated_xie_3x3_schedule_round_trips_identically_and_deterministically() {
    let instance = xie_3x3_instance();
    let input_obr = ObrChromosome::from_job_indices(&[1, 2, 2, 2, 1, 0, 0, 0, 1]);
    let schedule = decode_obr(&instance, &input_obr).expect("decode 3x3");

    let bundle_sched = round_trip_schedule(&instance, &schedule).expect("round-trip schedule");
    let bundle_seqs = round_trip_machine_sequences(&instance, &schedule.machine_sequences())
        .expect("round-trip machine sequences");

    assert_eq!(bundle_sched, bundle_seqs);
    assert_eq!(
        bundle_sched.encoded.chromosome.to_job_indices(),
        vec![1, 2, 0, 1, 0, 2, 1, 2, 0]
    );
    assert_eq!(bundle_sched.encoded.chromosome_length, 9);
    assert_eq!(bundle_sched.encoded.job_multiplicities, vec![3, 3, 3]);
    assert!(bundle_sched.encoded.technological_order_preserved);
    assert!(bundle_sched.encoded.obr_valid);

    assert_eq!(bundle_sched.decoded_schedule, schedule);
    assert_eq!(bundle_sched.audit.status, RoundTripStatus::Exact);
    assert!(bundle_sched.audit.status.is_exact());
    assert_eq!(bundle_sched.audit.target_graph_makespan, 11);
    assert_eq!(bundle_sched.audit.decoded_schedule_makespan, 11);
    assert_eq!(bundle_sched.audit.signed_makespan_diff, 0);
    assert_eq!(bundle_sched.audit.adapter_left_shift_count, 0);
    assert_eq!(bundle_sched.audit.adapter_right_shift_count, 0);
    assert_eq!(bundle_sched.audit.adapter_unchanged_start_count, 9);
    assert_eq!(bundle_sched.audit.machine_order_inversion_count, 0);
    assert_eq!(bundle_sched.audit.mismatched_machine_arc_count, 0);
    assert!(bundle_sched.audit.exact_machine_order_preserved);
    assert!(bundle_sched.audit.exact_start_times_preserved);
    assert!(bundle_sched.audit.exact_completion_times_preserved);
    assert_eq!(bundle_sched.audit.total_operation_count, 9);
    assert_eq!(bundle_sched.audit.target_graph_hash.len(), 64);
    assert_eq!(bundle_sched.audit.generated_obr_hash.len(), 64);
    assert_eq!(bundle_sched.audit.decoded_schedule_hash.len(), 64);

    // Determinism & idempotency of canonical re-encoding
    let repeat = round_trip_schedule(&instance, &bundle_sched.decoded_schedule).expect("repeat");
    assert_eq!(bundle_sched.audit, repeat.audit);
    assert_eq!(bundle_sched.encoded, repeat.encoded);
}

#[test]
fn multiple_critical_paths_and_equal_head_time_tie_breaking_round_trip() {
    // 1. Symmetric equal (head, completion) tie-breaking by (job_id, op_index)
    let sym_2x2 =
        JsspInstance::new(vec![vec![(0, 3), (1, 2)], vec![(1, 3), (0, 2)]]).expect("sym 2x2");
    let target_seqs = vec![
        MachineSequence::new(
            MachineId(0),
            vec![
                OperationId::from_indices(0, 0),
                OperationId::from_indices(1, 1),
            ],
        ),
        MachineSequence::new(
            MachineId(1),
            vec![
                OperationId::from_indices(1, 0),
                OperationId::from_indices(0, 1),
            ],
        ),
    ];
    let sym_bundle = round_trip_machine_sequences(&sym_2x2, &target_seqs).expect("sym round-trip");
    // O(0,0) and O(1,0) both tie at (0, 3); Job 0 precedes Job 1.
    // O(0,1) and O(1,1) both tie at (3, 5); Job 0 precedes Job 1.
    assert_eq!(
        sym_bundle.encoded.operation_keys,
        vec![
            CanonicalOperationKey {
                head: 0,
                completion: 3,
                job: 0,
                op_index: 0,
            },
            CanonicalOperationKey {
                head: 0,
                completion: 3,
                job: 1,
                op_index: 0,
            },
            CanonicalOperationKey {
                head: 3,
                completion: 5,
                job: 0,
                op_index: 1,
            },
            CanonicalOperationKey {
                head: 3,
                completion: 5,
                job: 1,
                op_index: 1,
            },
        ]
    );
    assert_eq!(
        sym_bundle.encoded.chromosome.to_job_indices(),
        vec![0, 1, 0, 1]
    );
    assert_eq!(sym_bundle.audit.status, RoundTripStatus::Exact);

    // 2. Multiple critical paths + completion tie-breaking on multi-path 3x3
    let mp = multi_path_3x3_instance();
    let mp_sched = decode_obr(
        &mp,
        &ObrChromosome::from_job_indices(&[0, 1, 2, 1, 1, 0, 0, 2, 2]),
    )
    .expect("decode multi-path 3x3");
    let mp_bundle = round_trip_schedule(&mp, &mp_sched).expect("round-trip multi-path");
    assert_eq!(mp_bundle.audit.status, RoundTripStatus::Exact);
    assert_eq!(mp_bundle.audit.target_graph_makespan, 10);
    // At head = 0: O(2,0) has completion 2 < 4 for O(0,0) and O(1,0); O(0,0) before O(1,0)
    assert_eq!(
        &mp_bundle.encoded.canonical_operations[0..3],
        &[
            OperationId::from_indices(2, 0),
            OperationId::from_indices(0, 0),
            OperationId::from_indices(1, 0),
        ]
    );
}

#[test]
fn rejects_invalid_duplicate_missing_wrong_machine_and_cyclic_machine_sequences() {
    let instance = adversarial_2x2_instance();

    // 1. Wrong count of machine sequences
    assert!(matches!(
        DisjunctiveGraph::from_target_machine_sequences(
            &instance,
            &[MachineSequence::new(
                MachineId(0),
                vec![
                    OperationId::from_indices(0, 1),
                    OperationId::from_indices(1, 0),
                ],
            )],
        ),
        Err(JsspError::InvalidDisjunctiveGraph { .. })
    ));

    // 2. Missing operation / truncated machine sequence
    assert!(matches!(
        round_trip_machine_sequences(
            &instance,
            &[
                MachineSequence::new(MachineId(0), vec![OperationId::from_indices(0, 1)]),
                MachineSequence::new(
                    MachineId(1),
                    vec![
                        OperationId::from_indices(0, 0),
                        OperationId::from_indices(1, 1),
                    ],
                ),
            ],
        ),
        Err(JsspError::InvalidDisjunctiveGraph { .. })
    ));

    // 3. Out-of-bounds / unexpected operation
    assert_eq!(
        round_trip_machine_sequences(
            &instance,
            &[
                MachineSequence::new(
                    MachineId(0),
                    vec![
                        OperationId::from_indices(0, 1),
                        OperationId::from_indices(9, 0),
                    ],
                ),
                MachineSequence::new(
                    MachineId(1),
                    vec![
                        OperationId::from_indices(0, 0),
                        OperationId::from_indices(1, 1),
                    ],
                ),
            ],
        ),
        Err(JsspError::UnexpectedOperation {
            job: 9,
            op_index: 0,
            num_jobs: 2,
            num_machines: 2,
        })
    );

    // 4. Operation assigned to wrong machine
    assert_eq!(
        round_trip_machine_sequences(
            &instance,
            &[
                MachineSequence::new(
                    MachineId(0),
                    vec![
                        OperationId::from_indices(0, 0), // actually M1!
                        OperationId::from_indices(1, 0),
                    ],
                ),
                MachineSequence::new(
                    MachineId(1),
                    vec![
                        OperationId::from_indices(0, 1),
                        OperationId::from_indices(1, 1),
                    ],
                ),
            ],
        ),
        Err(JsspError::OperationMachineMismatch {
            job: 0,
            op_index: 0,
            expected_machine: 1,
            actual_machine: 0,
        })
    );

    // 5. Duplicate operation on the same machine
    assert_eq!(
        round_trip_machine_sequences(
            &instance,
            &[
                MachineSequence::new(
                    MachineId(0),
                    vec![
                        OperationId::from_indices(1, 0),
                        OperationId::from_indices(1, 0),
                    ],
                ),
                MachineSequence::new(
                    MachineId(1),
                    vec![
                        OperationId::from_indices(0, 0),
                        OperationId::from_indices(1, 1),
                    ],
                ),
            ],
        ),
        Err(JsspError::DuplicateOperation {
            job: 1,
            op_index: 0,
        })
    );

    // 6. Cyclic machine orientation: M0=[O(1,0), O(0,1)] vs M1=[O(1,1), O(0,0)] creates
    //    O(1,0) -> O(1,1) -> O(0,0) -> O(0,1) -> O(1,0)? Actually wait:
    //    M0=[O(0,1), O(1,0)] and M1=[O(1,1), O(0,0)]:
    //    O(0,0)->O(0,1)->O(1,0)->O(1,1)->O(0,0) is a 4-operation directed cycle!
    let cyclic_err = round_trip_machine_sequences(
        &instance,
        &[
            MachineSequence::new(
                MachineId(0),
                vec![
                    OperationId::from_indices(0, 1),
                    OperationId::from_indices(1, 0),
                ],
            ),
            MachineSequence::new(
                MachineId(1),
                vec![
                    OperationId::from_indices(1, 1),
                    OperationId::from_indices(0, 0),
                ],
            ),
        ],
    )
    .expect_err("cyclic sequences must fail");
    assert_eq!(
        cyclic_err,
        JsspError::GraphCycleDetected {
            unvisited_vertices: 4,
            cycle_operations: vec![(0, 0), (0, 1), (1, 0), (1, 1)],
        }
    );
}

#[test]
fn section_6_adversarial_2x2_gap_insertion_demonstrates_non_active_order_change() {
    // Section 6 Illustrative Adversarial Test Case:
    // J0: (M1, 5) -> (M0, 4)
    // J1: (M0, 2) -> (M1, 3)
    // Target: M0 = [O(0,1), O(1,0)], M1 = [O(0,0), O(1,1)]
    let instance = adversarial_2x2_instance();
    let target_seqs = vec![
        MachineSequence::new(
            MachineId(0),
            vec![
                OperationId::from_indices(0, 1),
                OperationId::from_indices(1, 0),
            ],
        ),
        MachineSequence::new(
            MachineId(1),
            vec![
                OperationId::from_indices(0, 0),
                OperationId::from_indices(1, 1),
            ],
        ),
    ];

    let bundle =
        round_trip_machine_sequences(&instance, &target_seqs).expect("valid acyclic semi-active");

    // 1. Target graph head values & makespan
    assert_eq!(
        bundle
            .target_analysis
            .head(OperationId::from_indices(0, 0))
            .expect("O(0,0)"),
        0
    );
    assert_eq!(
        bundle
            .target_analysis
            .head(OperationId::from_indices(0, 1))
            .expect("O(0,1)"),
        5
    );
    assert_eq!(
        bundle
            .target_analysis
            .head(OperationId::from_indices(1, 0))
            .expect("O(1,0)"),
        9
    );
    assert_eq!(
        bundle
            .target_analysis
            .head(OperationId::from_indices(1, 1))
            .expect("O(1,1)"),
        11
    );
    assert_eq!(bundle.target_analysis.graph_makespan(), 14);

    // 2. Canonical topological OBR encoding is completely valid
    assert_eq!(
        bundle.encoded.canonical_operations,
        vec![
            OperationId::from_indices(0, 0),
            OperationId::from_indices(0, 1),
            OperationId::from_indices(1, 0),
            OperationId::from_indices(1, 1),
        ]
    );
    assert_eq!(bundle.encoded.chromosome.to_job_indices(), vec![0, 0, 1, 1]);
    assert!(bundle.encoded.obr_valid);
    assert!(bundle.encoded.technological_order_preserved);

    // 3. M2 Serial SGS inserts O(1,0) into idle gap [0, 5] before O(0,1) on M0
    let decoded_m0 = &bundle.decoded_schedule.machine_sequences()[0].operations;
    let decoded_m1 = &bundle.decoded_schedule.machine_sequences()[1].operations;
    assert_eq!(
        decoded_m0,
        &vec![
            OperationId::from_indices(1, 0),
            OperationId::from_indices(0, 1),
        ]
    );
    assert_eq!(
        decoded_m1,
        &vec![
            OperationId::from_indices(0, 0),
            OperationId::from_indices(1, 1),
        ]
    );
    assert_eq!(
        bundle
            .decoded_schedule
            .operation(OperationId::from_indices(0, 0))
            .expect("O(0,0)")
            .start_time,
        0
    );
    assert_eq!(
        bundle
            .decoded_schedule
            .operation(OperationId::from_indices(0, 1))
            .expect("O(0,1)")
            .start_time,
        5
    );
    assert_eq!(
        bundle
            .decoded_schedule
            .operation(OperationId::from_indices(1, 0))
            .expect("O(1,0)")
            .start_time,
        0
    );
    assert_eq!(
        bundle
            .decoded_schedule
            .operation(OperationId::from_indices(1, 1))
            .expect("O(1,1)")
            .start_time,
        5
    );
    assert_eq!(bundle.decoded_schedule.makespan(), 9);

    // 4. Audit report accurately quantifies left-shifts, inversion, arc mismatch, and diff
    assert_eq!(
        bundle.audit.status,
        RoundTripStatus::MachineOrderAndStartTimesChanged
    );
    assert!(!bundle.audit.status.is_exact());
    assert_eq!(bundle.audit.target_graph_makespan, 14);
    assert_eq!(bundle.audit.decoded_schedule_makespan, 9);
    assert_eq!(bundle.audit.signed_makespan_diff, -5);
    assert_eq!(bundle.audit.adapter_left_shift_count, 2);
    assert_eq!(bundle.audit.adapter_right_shift_count, 0);
    assert_eq!(bundle.audit.adapter_unchanged_start_count, 2);
    assert_eq!(bundle.audit.machine_order_inversion_count, 1);
    assert_eq!(bundle.audit.mismatched_machine_arc_count, 1);
    assert!(!bundle.audit.exact_machine_order_preserved);
    assert!(!bundle.audit.exact_start_times_preserved);
    assert!(!bundle.audit.exact_completion_times_preserved);

    // 5. Exhaustive check of all C(4, 2) = 6 valid OBR chromosomes on this 2x2 instance:
    // Every valid OBR chromosome decodes to M0 = [O(1,0), O(0,1)] (five yield C_max = 9 with
    // M1 = [O(0,0), O(1,1)], while [1, 1, 0, 0] flips M1 to [O(1,1), O(0,0)] with C_max = 14),
    // proving the semi-active non-active target orientation [O(0,1), O(1,0)] has no OBR
    // pre-image under active Serial SGS.
    let expected_outcomes = [
        ([0, 0, 1, 1], 9u64),
        ([0, 1, 0, 1], 9u64),
        ([0, 1, 1, 0], 9u64),
        ([1, 0, 0, 1], 9u64),
        ([1, 0, 1, 0], 9u64),
        ([1, 1, 0, 0], 14u64),
    ];
    for (genes, expected_cmax) in expected_outcomes {
        let sched =
            decode_obr(&instance, &ObrChromosome::from_job_indices(&genes)).expect("decode");
        assert_eq!(sched.makespan(), expected_cmax);
        assert_eq!(
            sched.machine_sequences()[0].operations,
            vec![
                OperationId::from_indices(1, 0),
                OperationId::from_indices(0, 1),
            ]
        );
    }
}

#[test]
fn right_shift_multi_inversion_and_makespan_increase_audit_telemetry() {
    // 3x1 instance: single machine M0 shared by 3 jobs with lengths [2, 3, 4].
    // Target sequence M0 = [O(0,0), O(1,0), O(2,0)] gives heads [0, 2, 5], C_max = 9.
    // Reversed chromosome [2, 1, 0] puts [O(2,0), O(1,0), O(0,0)] -> starts [7, 4, 0], C_max = 9:
    // - O(0,0): 0 -> 7 (right shift)
    // - O(1,0): 2 -> 4 (right shift)
    // - O(2,0): 5 -> 0 (left shift)
    // - 3 pairwise Kendall inversions ((2,1), (2,0), (1,0)) and 2 mismatched adjacent arcs!
    let inst_3x1 = JsspInstance::new(vec![vec![(0, 2)], vec![(0, 3)], vec![(0, 4)]]).expect("3x1");
    let target_seqs = vec![MachineSequence::new(
        MachineId(0),
        vec![
            OperationId::from_indices(0, 0),
            OperationId::from_indices(1, 0),
            OperationId::from_indices(2, 0),
        ],
    )];
    let target_graph = DisjunctiveGraph::from_target_machine_sequences(&inst_3x1, &target_seqs)
        .expect("target 3x1");
    let target_analysis = analyze_disjunctive_graph(&target_graph).expect("analyze 3x1");

    let rev_obr = ObrChromosome::from_job_indices(&[2, 1, 0]);
    let manual_encoded = EncodedObrResult {
        chromosome: rev_obr,
        canonical_operations: vec![
            OperationId::from_indices(2, 0),
            OperationId::from_indices(1, 0),
            OperationId::from_indices(0, 0),
        ],
        operation_keys: vec![],
        chromosome_length: 3,
        job_multiplicities: vec![1, 1, 1],
        technological_order_preserved: true,
        obr_valid: true,
    };
    let (_, rev_audit) =
        audit_obr_against_target_graph(&inst_3x1, &target_graph, &target_analysis, &manual_encoded)
            .expect("audit reverse 3x1");
    assert_eq!(rev_audit.adapter_right_shift_count, 2);
    assert_eq!(rev_audit.adapter_left_shift_count, 1);
    assert_eq!(rev_audit.adapter_unchanged_start_count, 0);
    assert_eq!(rev_audit.machine_order_inversion_count, 3);
    assert_eq!(rev_audit.mismatched_machine_arc_count, 2);
    assert_eq!(rev_audit.signed_makespan_diff, 0);
    assert_eq!(
        rev_audit.status,
        RoundTripStatus::MachineOrderAndStartTimesChanged
    );

    // Construct explicit positive signed_makespan_diff (+1) using non-canonical OBR on 2x2:
    // J0: (M0,3)->(M1,2), J1: (M0,4)->(M1,1). Target M0=[O(0,0), O(1,0)], M1=[O(0,1), O(1,1)] -> C_max=8.
    // Auditing [1, 1, 0, 0] produces C_max = 9 (+1 positive makespan diff, 2 right shifts).
    let inst_2x2 =
        JsspInstance::new(vec![vec![(0, 3), (1, 2)], vec![(0, 4), (1, 1)]]).expect("2x2");
    let tgt_2x2 = vec![
        MachineSequence::new(
            MachineId(0),
            vec![
                OperationId::from_indices(0, 0),
                OperationId::from_indices(1, 0),
            ],
        ),
        MachineSequence::new(
            MachineId(1),
            vec![
                OperationId::from_indices(0, 1),
                OperationId::from_indices(1, 1),
            ],
        ),
    ];
    let tgt_g2 = DisjunctiveGraph::from_target_machine_sequences(&inst_2x2, &tgt_2x2).expect("g2");
    let tgt_a2 = analyze_disjunctive_graph(&tgt_g2).expect("a2");
    let alt_encoded = EncodedObrResult {
        chromosome: ObrChromosome::from_job_indices(&[1, 1, 0, 0]),
        canonical_operations: vec![
            OperationId::from_indices(1, 0),
            OperationId::from_indices(1, 1),
            OperationId::from_indices(0, 0),
            OperationId::from_indices(0, 1),
        ],
        operation_keys: vec![],
        chromosome_length: 4,
        job_multiplicities: vec![2, 2],
        technological_order_preserved: true,
        obr_valid: true,
    };
    let (_, pos_diff_audit) =
        audit_obr_against_target_graph(&inst_2x2, &tgt_g2, &tgt_a2, &alt_encoded)
            .expect("audit positive diff");
    assert_eq!(pos_diff_audit.target_graph_makespan, 8);
    assert_eq!(pos_diff_audit.decoded_schedule_makespan, 9);
    assert_eq!(pos_diff_audit.signed_makespan_diff, 1);
    assert_eq!(pos_diff_audit.adapter_right_shift_count, 2);
    assert_eq!(pos_diff_audit.adapter_left_shift_count, 2);
    assert_eq!(pos_diff_audit.machine_order_inversion_count, 2);

    // Canonical D01-B encoding of both targets produces exact round trips
    let canon_2x2 = round_trip_machine_sequences(&inst_2x2, &tgt_2x2).expect("canon 2x2");
    assert_eq!(canon_2x2.audit.status, RoundTripStatus::Exact);
}

#[test]
fn property_based_category_a_m2_active_schedules_100_percent_exact() {
    let mut rng = SplitMix64::new(0x2026_1124_0000_0005);

    for case in 0..200 {
        let num_jobs = rng.gen_range(6) + 1;
        let num_machines = rng.gen_range(6) + 1;
        let instance = generate_random_instance(&mut rng, num_jobs, num_machines, 25);
        let obr = generate_random_obr(&mut rng, num_jobs, num_machines);
        let schedule = decode_obr(&instance, &obr).expect("valid M2 active decode");

        let bundle = round_trip_schedule(&instance, &schedule).expect("round-trip active");
        let repeat = round_trip_machine_sequences(&instance, &schedule.machine_sequences())
            .expect("round-trip seqs");

        assert_eq!(bundle.encoded, repeat.encoded);
        assert_eq!(bundle.audit, repeat.audit);
        assert_eq!(bundle.decoded_schedule, schedule);
        assert!(
            bundle.encoded.obr_valid && bundle.encoded.technological_order_preserved,
            "case {case}: OBR validity failed"
        );
        assert_eq!(
            bundle.audit.status,
            RoundTripStatus::Exact,
            "case {case}: active M2 schedule must round-trip identically"
        );
        assert_eq!(bundle.audit.adapter_left_shift_count, 0);
        assert_eq!(bundle.audit.adapter_right_shift_count, 0);
        assert_eq!(
            bundle.audit.adapter_unchanged_start_count,
            instance.num_operations()
        );
        assert_eq!(bundle.audit.machine_order_inversion_count, 0);
        assert_eq!(bundle.audit.mismatched_machine_arc_count, 0);
        assert_eq!(bundle.audit.signed_makespan_diff, 0);
        validate_schedule(&instance, &bundle.decoded_schedule).expect("feasible decoded schedule");
    }
}

#[test]
fn property_based_category_b_semi_active_append_only_measures_left_shifts_and_validity() {
    let mut rng = SplitMix64::new(0x2026_1124_0005_B001);
    let mut exact_count = 0usize;
    let mut left_shifted_count = 0usize;

    for case in 0..200 {
        let num_jobs = rng.gen_range(6) + 1;
        let num_machines = rng.gen_range(6) + 1;
        let instance = generate_random_instance(&mut rng, num_jobs, num_machines, 25);
        let obr = generate_random_obr(&mut rng, num_jobs, num_machines);

        // Append-only decoder yields valid semi-active (not necessarily active) machine sequences.
        let semi_active = decode_obr_append_only(&instance, &obr).expect("append-only decode");
        let bundle = round_trip_machine_sequences(&instance, &semi_active.machine_sequences())
            .expect("round-trip semi-active orientation");

        // OBR validity, technological precedence, and decoded M2 feasibility ALWAYS hold.
        assert!(bundle.encoded.obr_valid, "case {case}: obr invalid");
        assert!(
            bundle.encoded.technological_order_preserved,
            "case {case}: tech order invalid"
        );
        validate_schedule(&instance, &bundle.decoded_schedule).expect("decoded feasible");

        // By strong induction on canonical sorting key (head, head+p, job, op), Serial SGS
        // never right-shifts any operation relative to target_head and never worsens makespan.
        assert_eq!(
            bundle.audit.adapter_right_shift_count, 0,
            "case {case}: unexpected right shift on acyclic semi-active target"
        );
        assert!(
            bundle.audit.signed_makespan_diff <= 0,
            "case {case}: makespan unexpectedly increased"
        );

        let graph = DisjunctiveGraph::from_target_machine_sequences(
            &instance,
            &semi_active.machine_sequences(),
        )
        .expect("target");
        let analysis = analyze_disjunctive_graph(&graph).expect("analysis");
        let encoded = encode_graph_to_obr(&instance, &graph, &analysis).expect("encode");
        assert_eq!(encoded, bundle.encoded);

        if bundle.audit.status == RoundTripStatus::Exact {
            exact_count += 1;
            assert_eq!(bundle.audit.adapter_left_shift_count, 0);
            assert_eq!(bundle.audit.machine_order_inversion_count, 0);
        } else {
            left_shifted_count += 1;
            assert!(bundle.audit.adapter_left_shift_count >= 1);
        }
    }

    assert!(exact_count > 0 && left_shifted_count > 0);
}
