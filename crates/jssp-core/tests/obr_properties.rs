//! OBR chromosome representation, instance validation, and property tests.

use jssp_core::error::JsspError;
use jssp_core::model::{JobId, JsspInstance, OperationId};
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
            .map(|m| (m, (rng.gen_range(20) as u64) + 1))
            .collect();
        raw_jobs.push(row);
    }
    JsspInstance::new(raw_jobs).expect("generated instance must be valid")
}

#[test]
fn valid_obr_construction_and_resolution_across_sizes() {
    for &(num_jobs, num_machines) in &[(1, 1), (2, 2), (3, 3), (5, 4)] {
        let mut rng = SplitMix64::new((num_jobs * 100 + num_machines) as u64);
        let instance = generate_random_instance(&mut rng, num_jobs, num_machines);
        let canonical = ObrChromosome::canonical(&instance);

        assert_eq!(canonical.len(), num_jobs * num_machines);
        assert!(canonical.validate(&instance).is_ok());

        let counts = canonical
            .job_counts(num_jobs)
            .expect("valid job counts expected");
        assert!(counts.iter().all(|&c| c == num_machines));

        let resolved = canonical
            .resolve_operations(&instance)
            .expect("resolution should succeed");
        assert_eq!(resolved.len(), num_jobs * num_machines);
    }
}

#[test]
fn obr_rejects_invalid_length_bounds_and_multiplicities() {
    let instance = JsspInstance::new(vec![
        vec![(0, 2), (1, 1), (2, 3)],
        vec![(0, 1), (2, 2), (1, 2)],
        vec![(1, 5), (0, 2), (2, 1)],
    ])
    .expect("valid 3x3 instance");

    // Empty and short/long lengths
    assert_eq!(
        ObrChromosome::from_job_indices(&[]).validate(&instance),
        Err(JsspError::InvalidChromosomeLength {
            expected: 9,
            actual: 0
        })
    );
    assert_eq!(
        ObrChromosome::from_job_indices(&[0, 1, 2, 0]).validate(&instance),
        Err(JsspError::InvalidChromosomeLength {
            expected: 9,
            actual: 4
        })
    );
    assert_eq!(
        ObrChromosome::from_job_indices(&[0, 1, 2, 0, 1, 2, 0, 1, 2, 0]).validate(&instance),
        Err(JsspError::InvalidChromosomeLength {
            expected: 9,
            actual: 10
        })
    );

    // Invalid job ID
    assert_eq!(
        ObrChromosome::from_job_indices(&[0, 1, 3, 0, 1, 2, 0, 1, 2]).validate(&instance),
        Err(JsspError::InvalidJobId {
            position: 2,
            job: 3,
            num_jobs: 3
        })
    );

    // Wrong multiplicity: too many of job 0, too few of job 2
    assert_eq!(
        ObrChromosome::from_job_indices(&[0, 0, 0, 0, 1, 1, 1, 2, 2]).validate(&instance),
        Err(JsspError::InvalidJobMultiplicity {
            job: 0,
            expected: 3,
            actual: 4
        })
    );

    // Missing job (0 occurrences of job 1)
    assert_eq!(
        ObrChromosome::from_job_indices(&[0, 0, 0, 0, 2, 2, 2, 2, 2]).validate(&instance),
        Err(JsspError::InvalidJobMultiplicity {
            job: 0,
            expected: 3,
            actual: 4
        })
    );
    assert_eq!(
        ObrChromosome::from_job_indices(&[0, 0, 0, 2, 2, 2, 2, 2, 2]).validate(&instance),
        Err(JsspError::InvalidJobMultiplicity {
            job: 1,
            expected: 3,
            actual: 0
        })
    );
}

#[test]
fn obr_preserves_within_job_operation_resolution_order() {
    let instance = JsspInstance::new(vec![
        vec![(0, 2), (1, 1), (2, 3)],
        vec![(0, 1), (2, 2), (1, 2)],
        vec![(1, 5), (0, 2), (2, 1)],
    ])
    .expect("valid 3x3 instance");

    let chromosome = ObrChromosome::from_job_indices(&[1, 2, 2, 2, 1, 0, 0, 0, 1]);
    let resolved = chromosome
        .resolve_operations(&instance)
        .expect("OBR should resolve cleanly");

    let expected = vec![
        OperationId::from_indices(1, 0),
        OperationId::from_indices(2, 0),
        OperationId::from_indices(2, 1),
        OperationId::from_indices(2, 2),
        OperationId::from_indices(1, 1),
        OperationId::from_indices(0, 0),
        OperationId::from_indices(0, 1),
        OperationId::from_indices(0, 2),
        OperationId::from_indices(1, 2),
    ];
    assert_eq!(resolved, expected);
}

#[test]
fn operation_id_flat_index_round_trip_and_bounds() {
    let num_jobs = 4;
    let num_machines = 3;
    for flat in 0..(num_jobs * num_machines) {
        let op_id = OperationId::from_flat_index(flat, num_jobs, num_machines)
            .expect("flat index within bounds");
        let round_trip = op_id
            .flat_index(num_jobs, num_machines)
            .expect("round-trip flat index");
        assert_eq!(round_trip, flat);
    }

    assert_eq!(
        OperationId::from_flat_index(12, 4, 3),
        Err(JsspError::UnexpectedOperation {
            job: 4,
            op_index: 0,
            num_jobs: 4,
            num_machines: 3
        })
    );
    assert_eq!(
        OperationId::from_indices(1, 3).flat_index(4, 3),
        Err(JsspError::UnexpectedOperation {
            job: 1,
            op_index: 3,
            num_jobs: 4,
            num_machines: 3
        })
    );
}

#[test]
fn instance_validation_rejects_malformed_inputs_and_overflow() {
    assert_eq!(JsspInstance::new(vec![]), Err(JsspError::EmptyJobs));
    assert_eq!(
        JsspInstance::new(vec![vec![], vec![]]),
        Err(JsspError::EmptyMachines)
    );
    assert_eq!(
        JsspInstance::new(vec![vec![(0, 1), (1, 2)], vec![(0, 1)]]),
        Err(JsspError::NonRectangularJobLength {
            job: 1,
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        JsspInstance::new(vec![vec![(0, 1), (2, 2)], vec![(0, 1), (1, 2)]]),
        Err(JsspError::InvalidMachineId {
            job: 0,
            op_index: 1,
            machine: 2,
            num_machines: 2
        })
    );
    assert_eq!(
        JsspInstance::new(vec![vec![(0, 1), (0, 2)], vec![(0, 1), (1, 2)]]),
        Err(JsspError::DuplicateMachineInJob { job: 0, machine: 0 })
    );
    assert_eq!(
        JsspInstance::new(vec![vec![(0, 0), (1, 0)], vec![(0, 1), (1, 2)]]),
        Err(JsspError::NonPositiveProcessingTime {
            job: 0,
            op_index: 0,
            duration: 0
        })
    );
    let zero_op_inst = JsspInstance::new(vec![vec![(0, 2), (1, 0)], vec![(1, 3), (0, 2)]])
        .expect("zero-duration operation inside positive-work job (like orb07) must validate");
    let zero_chr = ObrChromosome::from_job_indices(&[0, 0, 1, 1]);
    let zero_sched =
        jssp_core::decoder::decode_obr(&zero_op_inst, &zero_chr).expect("decode with p=0 op");
    assert!(jssp_core::feasibility::validate_schedule(&zero_op_inst, &zero_sched).is_ok());
    let zero_graph = jssp_core::graph::DisjunctiveGraph::from_schedule(&zero_op_inst, &zero_sched)
        .expect("graph with p=0 op");
    let zero_cpa =
        jssp_core::graph::analyze_disjunctive_graph(&zero_graph).expect("cpa with p=0 op");
    assert_eq!(zero_cpa.graph_makespan(), zero_sched.makespan());
    let zero_rt = jssp_core::representation::round_trip_schedule(&zero_op_inst, &zero_sched)
        .expect("round-trip with p=0 op");
    assert_eq!(
        zero_rt.audit.decoded_schedule_makespan,
        zero_sched.makespan()
    );
    assert_eq!(
        JsspInstance::new(vec![vec![(0, u64::MAX), (1, 1)], vec![(0, 1), (1, 1)],]),
        Err(JsspError::TimeOverflow)
    );
}

#[test]
fn deterministic_property_test_obr_multiplicity_and_resolution_order() {
    let mut rng = SplitMix64::new(0xA5A5_5A5A_1234_5678);

    for _case in 0..100 {
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

        let obr = ObrChromosome::try_new(genes.clone(), &instance)
            .expect("shuffled valid chromosome must validate");
        let resolved = obr
            .resolve_operations(&instance)
            .expect("valid chromosome must resolve");
        assert_eq!(resolved.len(), num_jobs * num_machines);

        // Verify per-job operation index sequence is strictly 0, 1, ..., num_machines - 1
        let mut per_job_seen = vec![0usize; num_jobs];
        for (pos, op_id) in resolved.iter().enumerate() {
            assert_eq!(op_id.job, genes[pos]);
            assert_eq!(op_id.op_index.0, per_job_seen[op_id.job.0]);
            per_job_seen[op_id.job.0] += 1;
        }
        assert!(per_job_seen.iter().all(|&c| c == num_machines));
    }
}
