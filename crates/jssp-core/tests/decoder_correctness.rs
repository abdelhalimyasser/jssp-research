//! Serial SGS earliest idle-gap insertion decoder correctness and property tests.

use jssp_core::decoder::{SerialInsertionDecoder, decode_obr, decode_obr_append_only};
use jssp_core::error::JsspError;
use jssp_core::feasibility::validate_schedule;
use jssp_core::model::{JobId, JsspInstance, MachineId, OperationId, OperationIndex};
use jssp_core::objective::evaluate_makespan;
use jssp_core::representation::ObrChromosome;

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
) -> JsspInstance {
    let mut raw_jobs = Vec::with_capacity(num_jobs);
    for _ in 0..num_jobs {
        let mut machines: Vec<usize> = (0..num_machines).collect();
        rng.shuffle(&mut machines);
        let row = machines
            .into_iter()
            .map(|m| (m, (rng.gen_range(25) as u64) + 1))
            .collect();
        raw_jobs.push(row);
    }
    JsspInstance::new(raw_jobs).expect("generated instance must be valid")
}

fn xie_3x3_instance() -> JsspInstance {
    JsspInstance::new(vec![
        vec![(0, 2), (1, 1), (2, 3)],
        vec![(0, 1), (2, 2), (1, 2)],
        vec![(1, 5), (0, 2), (2, 1)],
    ])
    .expect("valid 3x3 instance")
}

#[test]
fn documented_3x3_gap_insertion_benchmark_gives_cmax_11_vs_append_only_13() {
    let instance = xie_3x3_instance();
    let chromosome = ObrChromosome::from_job_indices(&[1, 2, 2, 2, 1, 0, 0, 0, 1]);

    let schedule = decode_obr(&instance, &chromosome).expect("gap-insertion decode succeeds");
    validate_schedule(&instance, &schedule).expect("decoded schedule must pass feasibility");

    assert_eq!(schedule.makespan(), 11);
    assert_eq!(
        evaluate_makespan(&instance, &schedule).expect("makespan evaluation"),
        11
    );

    let expected_intervals = [
        ((1, 0), 0, 0, 1),
        ((2, 0), 1, 0, 5),
        ((2, 1), 0, 5, 7),
        ((2, 2), 2, 7, 8),
        ((1, 1), 2, 1, 3), // earliest idle-gap insertion before (2, 2) [7, 8]
        ((0, 0), 0, 1, 3), // earliest idle-gap insertion before (2, 1) [5, 7]
        ((0, 1), 1, 5, 6),
        ((0, 2), 2, 8, 11), // gap [3, 7] too narrow when ready=6
        ((1, 2), 1, 6, 8),
    ];

    for &((j, k), machine, start, end) in &expected_intervals {
        let op = schedule
            .operation(OperationId::from_indices(j, k))
            .expect("operation present");
        assert_eq!(op.machine, MachineId(machine));
        assert_eq!(op.start_time, start, "wrong start for O({j}, {k})");
        assert_eq!(op.completion_time, end, "wrong completion for O({j}, {k})");
    }

    let machine_seqs = schedule.machine_sequences();
    assert_eq!(
        machine_seqs[0].operations,
        vec![
            OperationId::from_indices(1, 0),
            OperationId::from_indices(0, 0),
            OperationId::from_indices(2, 1),
        ]
    );
    assert_eq!(
        machine_seqs[1].operations,
        vec![
            OperationId::from_indices(2, 0),
            OperationId::from_indices(0, 1),
            OperationId::from_indices(1, 2),
        ]
    );
    assert_eq!(
        machine_seqs[2].operations,
        vec![
            OperationId::from_indices(1, 1),
            OperationId::from_indices(2, 2),
            OperationId::from_indices(0, 2),
        ]
    );

    // Contrast with pure semi-active append-only decoding (C_max = 13).
    let append_schedule =
        decode_obr_append_only(&instance, &chromosome).expect("append-only decode succeeds");
    validate_schedule(&instance, &append_schedule).expect("append-only schedule is still feasible");
    assert_eq!(append_schedule.makespan(), 13);
    assert_eq!(
        append_schedule
            .operation(OperationId::from_indices(1, 1))
            .expect("O(1,1)")
            .start_time,
        8
    );
    assert_eq!(
        append_schedule
            .operation(OperationId::from_indices(1, 2))
            .expect("O(1,2)")
            .completion_time,
        12
    );
    assert_eq!(
        append_schedule
            .operation(OperationId::from_indices(0, 2))
            .expect("O(0,2)")
            .completion_time,
        13
    );
}

#[test]
fn hand_computable_2x2_and_predecessor_waiting() {
    // J0: (M0, 4) -> (M1, 3)
    // J1: (M1, 5) -> (M0, 2)
    // Chromosome [1, 1, 0, 0]:
    //   O(1,0) on M1 -> [0, 5]
    //   O(1,1) on M0 (must wait for job predecessor C_{1,0}=5 even though M0 idle at t=0) -> [5, 7]
    //   O(0,0) on M0 (ready=0; gap [0, 5] before O(1,1) fits duration 4!) -> [0, 4]
    //   O(0,1) on M1 (ready=4; waits for M1 busy [0, 5]) -> [5, 8]
    // Makespan = 8.
    let instance =
        JsspInstance::new(vec![vec![(0, 4), (1, 3)], vec![(1, 5), (0, 2)]]).expect("2x2");
    let obr = ObrChromosome::from_job_indices(&[1, 1, 0, 0]);
    let schedule = SerialInsertionDecoder::new()
        .decode(&instance, &obr)
        .expect("2x2 decode");
    validate_schedule(&instance, &schedule).expect("valid 2x2 schedule");

    assert_eq!(schedule.makespan(), 8);
    assert_eq!(
        schedule
            .operation(OperationId::from_indices(1, 0))
            .expect("O(1,0)")
            .completion_time,
        5
    );
    assert_eq!(
        schedule
            .operation(OperationId::from_indices(1, 1))
            .expect("O(1,1)")
            .start_time,
        5
    );
    assert_eq!(
        schedule
            .operation(OperationId::from_indices(0, 0))
            .expect("O(0,0)")
            .start_time,
        0
    );
    assert_eq!(
        schedule
            .operation(OperationId::from_indices(0, 1))
            .expect("O(0,1)")
            .start_time,
        5
    );
}

#[test]
fn earliest_gap_selection_prefers_first_feasible_idle_gap() {
    // Construct a 3x3 instance where M0 receives:
    //   J1: O(1,0)=(M1,5), O(1,1)=(M0,1), O(1,2)=(M2,1)   -> O(1,1) on M0 at [5, 6]
    //   J2: O(2,0)=(M2,10), O(2,1)=(M0,1), O(2,2)=(M1,1)  -> O(2,1) on M0 at [10, 11]
    // Create two gaps on M0: [0, 5] and [6, 10].
    // Next J0 has O(0,0)=(M0,3) ready at t=0: must pick [0, 3] (first gap) rather than [6, 9] or [11, 14].
    let instance = JsspInstance::new(vec![
        vec![(0, 3), (1, 1), (2, 1)],
        vec![(1, 5), (0, 1), (2, 1)],
        vec![(2, 10), (0, 1), (1, 1)],
    ])
    .expect("valid 3x3");

    let obr = ObrChromosome::from_job_indices(&[1, 1, 2, 2, 0, 0, 0, 1, 2]);
    let schedule = decode_obr(&instance, &obr).expect("decode");
    validate_schedule(&instance, &schedule).expect("feasible");

    let o_00 = schedule
        .operation(OperationId::new(JobId(0), OperationIndex(0)))
        .expect("O(0,0)");
    assert_eq!((o_00.start_time, o_00.completion_time), (0, 3));
}

#[test]
fn decoder_is_deterministic_and_handles_large_u64_durations() {
    let instance = xie_3x3_instance();
    let obr = ObrChromosome::from_job_indices(&[1, 2, 2, 2, 1, 0, 0, 0, 1]);
    let first = decode_obr(&instance, &obr).expect("first decode");
    let second = decode_obr(&instance, &obr).expect("second decode");
    assert_eq!(first, second);

    // Large valid u64 durations summing to <= u64::MAX
    let quarter = u64::MAX / 4;
    let big_instance = JsspInstance::new(vec![
        vec![(0, quarter), (1, quarter)],
        vec![(0, quarter), (1, quarter)],
    ])
    .expect("sum fits within u64::MAX");
    let big_obr = ObrChromosome::from_job_indices(&[0, 1, 0, 1]);
    let big_schedule = decode_obr(&big_instance, &big_obr).expect("large u64 decode succeeds");
    validate_schedule(&big_instance, &big_schedule).expect("large schedule is feasible");
    assert_eq!(big_schedule.makespan(), quarter * 3);

    // Overflowing instance sum rejected deterministically with TimeOverflow
    let overflow_instance =
        JsspInstance::new(vec![vec![(0, u64::MAX - 1), (1, 1)], vec![(0, 2), (1, 1)]]);
    assert_eq!(overflow_instance, Err(JsspError::TimeOverflow));
}

#[test]
fn deterministic_property_based_suite_verifies_all_eight_decoder_invariants() {
    let mut rng = SplitMix64::new(0x2026_1123_0000_0002);

    for _case in 0..200 {
        let num_jobs = rng.gen_range(6) + 1;
        let num_machines = rng.gen_range(6) + 1;
        let instance = generate_random_instance(&mut rng, num_jobs, num_machines);

        let mut genes: Vec<JobId> = Vec::with_capacity(num_jobs * num_machines);
        for j in 0..num_jobs {
            for _ in 0..num_machines {
                genes.push(JobId(j));
            }
        }
        rng.shuffle(&mut genes);
        let obr = ObrChromosome::new(genes);

        // 1. Every valid OBR decodes without error
        let schedule = decode_obr(&instance, &obr).expect("valid OBR must decode");
        let schedule_repeat = decode_obr(&instance, &obr).expect("repeat decode succeeds");
        let append_schedule =
            decode_obr_append_only(&instance, &obr).expect("append-only decode succeeds");

        // 2. Every decoded schedule passes validate_schedule
        validate_schedule(&instance, &schedule).expect("gap-insertion schedule must be valid");
        validate_schedule(&instance, &append_schedule).expect("append-only schedule must be valid");

        // 3. Operation count equals J * M
        assert_eq!(schedule.operations().len(), num_jobs * num_machines);

        // 4. Job precedence holds for every job
        for job_ops in schedule.job_sequences() {
            assert_eq!(job_ops.len(), num_machines);
            for window in job_ops.windows(2) {
                assert!(window[0].completion_time <= window[1].start_time);
            }
        }

        // 5. Machine exclusivity holds for every machine
        for m in 0..num_machines {
            let m_ops = schedule.machine_operations(MachineId(m));
            assert_eq!(m_ops.len(), num_jobs);
            for window in m_ops.windows(2) {
                assert!(window[0].completion_time <= window[1].start_time);
            }
        }

        // 6. Completion time equals start + processing time
        for op in schedule.operations() {
            assert_eq!(op.start_time + op.processing_time, op.completion_time);
        }

        // 7. Recomputed makespan matches decoded makespan and active <= append-only
        let recomputed =
            evaluate_makespan(&instance, &schedule).expect("makespan evaluation succeeds");
        assert_eq!(schedule.makespan(), recomputed);
        assert!(schedule.makespan() <= append_schedule.makespan());

        // 8. Bit-for-bit determinism
        assert_eq!(schedule, schedule_repeat);
    }
}
