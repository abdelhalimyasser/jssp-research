//! Comprehensive tests for disjunctive graph construction, DAG topological sorting,
//! head/tail DP, critical-path / critical-subgraph analysis, critical blocks, and
//! seeded property-based + exhaustive-path-enumeration oracle checks (Milestone M4).

use std::collections::BTreeSet;

use jssp_core::decoder::{decode_obr, decode_obr_append_only};
use jssp_core::error::JsspError;
use jssp_core::feasibility::{check_disjunctive_graph_acyclicity, validate_schedule};
use jssp_core::graph::{
    ArcId, ArcType, DisjunctiveGraph, GraphVertexKind, VertexId, analyze_disjunctive_graph,
    analyze_schedule, extract_critical_blocks,
};
use jssp_core::model::{
    JobId, JsspInstance, MachineId, MachineSequence, OperationId, Schedule, ScheduledOperation,
};
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
    // J0: (M0, 4) -> (M1, 2) -> (M2, 4)
    // J1: (M1, 4) -> (M2, 1) -> (M0, 1)
    // J2: (M2, 2) -> (M0, 3) -> (M1, 1)
    JsspInstance::new(vec![
        vec![(0, 4), (1, 2), (2, 4)],
        vec![(1, 4), (2, 1), (0, 1)],
        vec![(2, 2), (0, 3), (1, 1)],
    ])
    .expect("valid multi-path 3x3")
}

fn two_blocks_one_machine_4x3_instance() -> JsspInstance {
    // J0: (M0, 2) -> (M1, 1) -> (M2, 1)
    // J1: (M0, 3) -> (M1, 6) -> (M2, 2)
    // J2: (M2, 8) -> (M0, 2) -> (M1, 1)
    // J3: (M1, 1) -> (M2, 1) -> (M0, 3)
    JsspInstance::new(vec![
        vec![(0, 2), (1, 1), (2, 1)],
        vec![(0, 3), (1, 6), (2, 2)],
        vec![(2, 8), (0, 2), (1, 1)],
        vec![(1, 1), (2, 1), (0, 3)],
    ])
    .expect("valid 4x3")
}

#[test]
fn graph_construction_single_job_single_machine_and_xie_3x3() {
    // 1x1 single job single machine
    let inst_1x1 = JsspInstance::new(vec![vec![(0, 7)]]).expect("1x1");
    let sched_1x1 =
        decode_obr(&inst_1x1, &ObrChromosome::from_job_indices(&[0])).expect("decode 1x1");
    let (g1, a1) = analyze_schedule(&inst_1x1, &sched_1x1).expect("analyze 1x1");

    assert_eq!(g1.metadata().num_operations, 1);
    assert_eq!(g1.metadata().num_vertices, 3);
    assert_eq!(g1.metadata().num_job_precedence_arcs, 0);
    assert_eq!(g1.metadata().num_machine_order_arcs, 0);
    assert_eq!(g1.metadata().num_source_arcs, 1);
    assert_eq!(g1.metadata().num_sink_arcs, 1);
    assert_eq!(g1.metadata().num_arcs, 2);
    assert_eq!(a1.graph_makespan(), 7);
    assert_eq!(
        a1.topological_order(),
        &[VertexId::SOURCE, VertexId(1), VertexId(2)]
    );
    assert_eq!(a1.critical_operations(), &[OperationId::from_indices(0, 0)]);
    assert!(a1.critical_blocks().is_empty());
    assert_eq!(
        a1.representative_critical_path().operations,
        vec![OperationId::from_indices(0, 0)]
    );

    // Xie 3x3 benchmark (Cmax = 11)
    let inst_3x3 = xie_3x3_instance();
    let sched_3x3 = decode_obr(
        &inst_3x3,
        &ObrChromosome::from_job_indices(&[1, 2, 2, 2, 1, 0, 0, 0, 1]),
    )
    .expect("decode 3x3");
    let (g3, a3) = analyze_schedule(&inst_3x3, &sched_3x3).expect("analyze 3x3");

    assert_eq!(g3.metadata().num_vertices, 11);
    assert_eq!(g3.metadata().num_job_precedence_arcs, 6);
    assert_eq!(g3.metadata().num_machine_order_arcs, 6);
    assert_eq!(g3.metadata().num_source_arcs, 2); // O(1,0) and O(2,0)
    assert_eq!(g3.metadata().num_sink_arcs, 2); // O(0,2) and O(1,2)
    assert_eq!(g3.metadata().num_arcs, 16);

    // Verify no duplicate (from, to, arc_type) arcs
    let arc_keys: BTreeSet<(VertexId, VertexId, ArcType)> = g3
        .arcs()
        .iter()
        .map(|a| (a.from, a.to, a.arc_type))
        .collect();
    assert_eq!(arc_keys.len(), g3.arcs().len());

    // Exact heads & tails match hand calculations
    let expected_ht = [
        ((0, 0), 1, 6, 2, false),
        ((0, 1), 5, 3, 2, false),
        ((0, 2), 8, 0, 0, true),
        ((1, 0), 0, 8, 2, false),
        ((1, 1), 1, 4, 4, false),
        ((1, 2), 6, 0, 3, false),
        ((2, 0), 0, 6, 0, true),
        ((2, 1), 5, 4, 0, true),
        ((2, 2), 7, 3, 0, true),
    ];
    for &((j, k), head, tail, slack, crit) in &expected_ht {
        let rec = a3
            .operation(OperationId::from_indices(j, k))
            .expect("op exists");
        assert_eq!(rec.head, head, "head mismatch for O({j},{k})");
        assert_eq!(rec.tail, tail, "tail mismatch for O({j},{k})");
        assert_eq!(rec.slack, slack, "slack mismatch for O({j},{k})");
        assert_eq!(rec.is_critical, crit, "critical mismatch for O({j},{k})");
    }

    assert_eq!(a3.graph_makespan(), 11);
    assert_eq!(
        a3.representative_critical_path().operations,
        vec![
            OperationId::from_indices(2, 0),
            OperationId::from_indices(2, 1),
            OperationId::from_indices(2, 2),
            OperationId::from_indices(0, 2),
        ]
    );
    assert_eq!(a3.critical_blocks().len(), 1);
    let blk = &a3.critical_blocks()[0];
    assert_eq!(blk.machine, MachineId(2));
    assert_eq!(
        blk.operations,
        vec![
            OperationId::from_indices(2, 2),
            OperationId::from_indices(0, 2),
        ]
    );
}

#[test]
fn invalid_and_non_semi_active_schedules_are_rejected() {
    let inst = xie_3x3_instance();
    let sched = decode_obr(
        &inst,
        &ObrChromosome::from_job_indices(&[1, 2, 2, 2, 1, 0, 0, 0, 1]),
    )
    .expect("decode");

    // Wrong reported makespan triggers M2 validation failure inside DisjunctiveGraph::from_schedule
    let bad_makespan = Schedule::new(3, 3, sched.operations().to_vec(), 99);
    assert_eq!(
        DisjunctiveGraph::from_schedule(&inst, &bad_makespan),
        Err(JsspError::MakespanMismatch {
            reported: 99,
            recomputed: 11,
        })
    );

    // Valid feasible schedule with artificial right-shift delay passes M2 feasibility
    // validation, yet fails GraphMakespanMismatch because Cmax_graph < Cmax_schedule.
    let inst_1x1 = JsspInstance::new(vec![vec![(0, 5)]]).expect("1x1");
    let shifted_sched = Schedule::new(
        1,
        1,
        vec![ScheduledOperation {
            op_id: OperationId::from_indices(0, 0),
            machine: MachineId(0),
            start_time: 2,
            completion_time: 7,
            processing_time: 5,
        }],
        7,
    );
    validate_schedule(&inst_1x1, &shifted_sched).expect("right-shifted schedule is M2 feasible");
    assert_eq!(
        analyze_schedule(&inst_1x1, &shifted_sched),
        Err(JsspError::GraphMakespanMismatch {
            schedule_makespan: 7,
            graph_makespan: 5,
        })
    );
}

#[test]
fn dag_topological_tie_breaking_and_malformed_cycle_detection() {
    // 2x2 parallel jobs: J0: (M0,3)->(M1,2), J1: (M1,4)->(M0,1)
    // Machine sequences: M0=[O(0,0), O(1,1)], M1=[O(1,0), O(0,1)]
    // Both O(0,0) and O(1,0) become ready together; stable VertexId tie-breaker picks O(0,0) before O(1,0).
    let inst = JsspInstance::new(vec![vec![(0, 3), (1, 2)], vec![(1, 4), (0, 1)]]).expect("2x2");
    let sched =
        decode_obr(&inst, &ObrChromosome::from_job_indices(&[1, 0, 0, 1])).expect("decode 2x2");
    let graph = DisjunctiveGraph::from_schedule(&inst, &sched).expect("acyclic");
    let topo = check_disjunctive_graph_acyclicity(&graph).expect("dag valid");

    let op_order: Vec<OperationId> = topo
        .iter()
        .filter_map(|&v| graph.vertex(v).expect("v").operation_id())
        .collect();
    assert_eq!(
        op_order,
        vec![
            OperationId::from_indices(0, 0),
            OperationId::from_indices(1, 0),
            OperationId::from_indices(0, 1),
            OperationId::from_indices(1, 1),
        ]
    );

    // Malformed cyclic graph detection on reversed machine orders:
    // M0: [O(1,1), O(0,0)], M1: [O(0,1), O(1,0)] => cycle O(0,0)->O(0,1)->O(1,0)->O(1,1)->O(0,0)
    let cyclic_seqs = vec![
        MachineSequence::new(
            MachineId(0),
            vec![
                OperationId::from_indices(1, 1),
                OperationId::from_indices(0, 0),
            ],
        ),
        MachineSequence::new(
            MachineId(1),
            vec![
                OperationId::from_indices(0, 1),
                OperationId::from_indices(1, 0),
            ],
        ),
    ];
    let cyclic_graph = DisjunctiveGraph::from_machine_sequences_unchecked(&inst, &cyclic_seqs, 10)
        .expect("unchecked build");
    let err = check_disjunctive_graph_acyclicity(&cyclic_graph).expect_err("must detect cycle");
    assert_eq!(
        err,
        JsspError::GraphCycleDetected {
            unvisited_vertices: 4,
            cycle_operations: vec![(0, 0), (0, 1), (1, 0), (1, 1)],
        }
    );
    assert_eq!(analyze_disjunctive_graph(&cyclic_graph), Err(err));
}

#[test]
fn boundary_u64_max_single_operation_head_and_tail() {
    let inst = JsspInstance::new(vec![vec![(0, u64::MAX)]]).expect("u64::MAX 1x1");
    let sched = decode_obr(&inst, &ObrChromosome::from_job_indices(&[0])).expect("decode");
    let (_, analysis) = analyze_schedule(&inst, &sched).expect("analyze max u64");
    assert_eq!(analysis.graph_makespan(), u64::MAX);
    let op = analysis
        .operation(OperationId::from_indices(0, 0))
        .expect("op");
    assert_eq!(op.head, 0);
    assert_eq!(op.tail, 0);
    assert_eq!(op.slack, 0);
    assert!(op.is_critical);
}

#[test]
fn multiple_critical_paths_ties_and_non_critical_arcs_between_critical_ops() {
    let inst = multi_path_3x3_instance();
    let obr = ObrChromosome::from_job_indices(&[0, 1, 2, 1, 1, 0, 0, 2, 2]);
    let sched = decode_obr(&inst, &obr).expect("decode multi-path 3x3");
    let (graph, analysis) = analyze_schedule(&inst, &sched).expect("analyze multi-path");

    assert_eq!(analysis.graph_makespan(), 10);

    // Representative path follows lowest OperationId backtracking from Sink:
    // Sink max preds = [O(0,2), O(2,2)] -> O(0,2)
    // O(0,2) max pred = [O(0,1)]
    // O(0,1) max preds = [O(0,0), O(1,0)] -> O(0,0)
    assert_eq!(
        analysis.representative_critical_path().operations,
        vec![
            OperationId::from_indices(0, 0),
            OperationId::from_indices(0, 1),
            OperationId::from_indices(0, 2),
        ]
    );
    assert_eq!(
        analysis
            .representative_critical_path()
            .total_processing_time,
        10
    );

    // Verify all 8 critical operations (every op except O(2,0)), including 5 NOT on rep path
    let mut crit_ops = analysis.critical_operations().to_vec();
    crit_ops.sort();
    assert_eq!(
        crit_ops,
        vec![
            OperationId::from_indices(0, 0),
            OperationId::from_indices(0, 1),
            OperationId::from_indices(0, 2),
            OperationId::from_indices(1, 0),
            OperationId::from_indices(1, 1),
            OperationId::from_indices(1, 2),
            OperationId::from_indices(2, 1),
            OperationId::from_indices(2, 2),
        ]
    );
    let o20 = analysis
        .operation(OperationId::from_indices(2, 0))
        .expect("O(2,0)");
    assert!(!o20.is_critical);
    assert_eq!(o20.slack, 2);

    // Maximizing predecessors & successors checks
    assert_eq!(
        analysis
            .maximizing_predecessors(OperationId::from_indices(0, 1))
            .expect("O(0,1)"),
        &[
            OperationId::from_indices(0, 0),
            OperationId::from_indices(1, 0),
        ]
    );
    assert_eq!(
        analysis
            .maximizing_successors(OperationId::from_indices(1, 0))
            .expect("O(1,0)"),
        &[
            OperationId::from_indices(0, 1),
            OperationId::from_indices(1, 1),
        ]
    );
    assert_eq!(
        analysis.sink_maximizing_predecessors(),
        &[
            OperationId::from_indices(0, 2),
            OperationId::from_indices(2, 2),
        ]
    );
    assert_eq!(
        analysis.source_maximizing_successors(),
        &[
            OperationId::from_indices(0, 0),
            OperationId::from_indices(1, 0),
        ]
    );

    // Crucial test (§9): O(0,0) -> O(1,2) on M0, O(1,1) -> O(0,2) on M2, and O(0,1) -> O(2,2)
    // on M1 ALL connect TWO critical operations, yet NONE of those three machine arcs is critical!
    let find_arc = |from: (usize, usize), to: (usize, usize)| {
        let f_op = OperationId::from_indices(from.0, from.1);
        let t_op = OperationId::from_indices(to.0, to.1);
        analysis
            .arcs()
            .iter()
            .find(|a| a.from_op == Some(f_op) && a.to_op == Some(t_op))
            .copied()
            .expect("arc must exist")
    };

    for non_crit_pair in [((0, 0), (1, 2)), ((1, 1), (0, 2)), ((0, 1), (2, 2))] {
        let arc = find_arc(non_crit_pair.0, non_crit_pair.1);
        assert!(
            matches!(arc.arc_type, ArcType::MachineOrder { .. }),
            "expected machine arc for {non_crit_pair:?}"
        );
        assert!(
            !arc.is_head_tight && !arc.is_critical,
            "arc {non_crit_pair:?} between critical ops must NOT be classified critical"
        );
    }

    // Critical blocks extracted on M0 ([O(1,2), O(2,1)]) and M1 ([O(1,0), O(0,1)]) — neither
    // illegally merges adjacent non-tight critical ops O(0,0) or O(2,2).
    let blocks = analysis.critical_blocks();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].machine, MachineId(0));
    assert_eq!(
        blocks[0].operations,
        vec![
            OperationId::from_indices(1, 2),
            OperationId::from_indices(2, 1),
        ]
    );
    assert_eq!(blocks[1].machine, MachineId(1));
    assert_eq!(
        blocks[1].operations,
        vec![
            OperationId::from_indices(1, 0),
            OperationId::from_indices(0, 1),
        ]
    );

    // Determinism check against machine sequences order
    assert_eq!(graph.machine_sequences()[0].operations.len(), 3);
}

#[test]
fn multiple_critical_blocks_on_same_machine_are_maximal() {
    let inst = two_blocks_one_machine_4x3_instance();
    let obr = ObrChromosome::from_job_indices(&[0, 1, 2, 0, 0, 1, 1, 3, 3, 2, 3, 2]);
    let sched = decode_obr(&inst, &obr).expect("decode 4x3");
    let (_, analysis) = analyze_schedule(&inst, &sched).expect("analyze 4x3");

    assert_eq!(analysis.graph_makespan(), 13);
    // Two separate blocks on M0 ([O(0,0), O(1,0)] and [O(2,1), O(3,2)]), plus one 3-op block
    // on M2 ([O(2,0), O(0,2), O(3,1)]) which stops before critical op O(1,2) due to gap [10, 11].
    assert_eq!(analysis.critical_blocks().len(), 3);

    let b0 = &analysis.critical_blocks()[0];
    let b1 = &analysis.critical_blocks()[1];
    let b2 = &analysis.critical_blocks()[2];
    assert_eq!(b0.id.index(), 0);
    assert_eq!(b0.machine, MachineId(0));
    assert_eq!(
        b0.operations,
        vec![
            OperationId::from_indices(0, 0),
            OperationId::from_indices(1, 0),
        ]
    );
    assert_eq!(b0.first_operation, OperationId::from_indices(0, 0));
    assert_eq!(b0.last_operation, OperationId::from_indices(1, 0));
    assert_eq!(b0.len, 2);
    assert_eq!(b0.critical_machine_arcs.len(), 1);

    assert_eq!(b1.id.index(), 1);
    assert_eq!(b1.machine, MachineId(0));
    assert_eq!(
        b1.operations,
        vec![
            OperationId::from_indices(2, 1),
            OperationId::from_indices(3, 2),
        ]
    );
    assert_eq!(b1.first_operation, OperationId::from_indices(2, 1));
    assert_eq!(b1.last_operation, OperationId::from_indices(3, 2));
    assert_eq!(b1.len, 2);
    assert_eq!(b1.critical_machine_arcs.len(), 1);

    assert_eq!(b2.id.index(), 2);
    assert_eq!(b2.machine, MachineId(2));
    assert_eq!(
        b2.operations,
        vec![
            OperationId::from_indices(2, 0),
            OperationId::from_indices(0, 2),
            OperationId::from_indices(3, 1),
        ]
    );
    assert_eq!(b2.first_operation, OperationId::from_indices(2, 0));
    assert_eq!(b2.last_operation, OperationId::from_indices(3, 1));
    assert_eq!(b2.len, 3);
    assert_eq!(b2.critical_machine_arcs.len(), 2);
}

fn enumerate_source_to_sink_paths(
    graph: &DisjunctiveGraph,
) -> Vec<(Vec<VertexId>, Vec<ArcId>, u64)> {
    fn dfs(
        graph: &DisjunctiveGraph,
        curr: VertexId,
        sink: VertexId,
        path_v: &mut Vec<VertexId>,
        path_a: &mut Vec<ArcId>,
        weight: u64,
        out: &mut Vec<(Vec<VertexId>, Vec<ArcId>, u64)>,
    ) {
        if curr == sink {
            out.push((path_v.clone(), path_a.clone(), weight));
            return;
        }
        for &arc_id in graph.outgoing_arcs(curr).expect("outgoing") {
            let next_v = graph.arc(arc_id).expect("arc").to;
            let next_w = weight + graph.vertex(next_v).expect("v").processing_time;
            path_v.push(next_v);
            path_a.push(arc_id);
            dfs(graph, next_v, sink, path_v, path_a, next_w, out);
            path_a.pop();
            path_v.pop();
        }
    }

    let mut paths = Vec::new();
    let mut pv = vec![graph.source_vertex_id()];
    let mut pa = Vec::new();
    dfs(
        graph,
        graph.source_vertex_id(),
        graph.sink_vertex_id(),
        &mut pv,
        &mut pa,
        0,
        &mut paths,
    );
    paths
}

#[test]
fn property_based_and_exhaustive_path_oracle_verify_all_m4_invariants() {
    let mut rng = SplitMix64::new(0x2026_1124_0000_0004);

    for case in 0..200 {
        let num_jobs = rng.gen_range(6) + 1;
        let num_machines = rng.gen_range(6) + 1;
        let instance = generate_random_instance(&mut rng, num_jobs, num_machines, 25);
        let obr = generate_random_obr(&mut rng, num_jobs, num_machines);

        let schedule = decode_obr(&instance, &obr).expect("valid decode");
        validate_schedule(&instance, &schedule).expect("valid schedule");

        let (graph, analysis) = analyze_schedule(&instance, &schedule).expect("graph analysis");
        let (graph_repeat, analysis_repeat) =
            analyze_schedule(&instance, &schedule).expect("deterministic repeat");

        // 1. Determinism
        assert_eq!(graph, graph_repeat);
        assert_eq!(analysis, analysis_repeat);

        // 2. Acyclicity & exact operation representation
        let topo = check_disjunctive_graph_acyclicity(&graph).expect("acyclic");
        assert_eq!(topo.first().copied(), Some(graph.source_vertex_id()));
        assert_eq!(topo.last().copied(), Some(graph.sink_vertex_id()));
        assert_eq!(
            analysis.operations().len(),
            instance.num_operations(),
            "case {case}: every op represented once"
        );

        // 3. Every job precedence arc respects technological order; every machine arc respects sequence
        for arc in graph.arcs() {
            match arc.arc_type {
                ArcType::JobPrecedence { job } => {
                    let from_op = arc.from_op.expect("from_op");
                    let to_op = arc.to_op.expect("to_op");
                    assert_eq!(from_op.job, job);
                    assert_eq!(to_op.job, job);
                    assert_eq!(from_op.op_index.0 + 1, to_op.op_index.0);
                }
                ArcType::MachineOrder { machine } => {
                    let from_op = arc.from_op.expect("from_op");
                    let to_op = arc.to_op.expect("to_op");
                    let m_seq = &schedule.machine_sequences()[machine.0].operations;
                    let pos = m_seq
                        .iter()
                        .position(|&id| id == from_op)
                        .expect("from_op on machine");
                    assert_eq!(m_seq[pos + 1], to_op);
                }
                ArcType::Source | ArcType::Sink => {}
            }
        }

        // 4. Head/tail inequalities, slack >= 0, decoder start agreement, >= 1 critical op
        let cmax = analysis.graph_makespan();
        assert_eq!(cmax, schedule.makespan());
        assert!(!analysis.critical_operations().is_empty());

        for op_crit in analysis.operations() {
            assert!(op_crit.head + op_crit.processing_time + op_crit.tail <= cmax);
            assert_eq!(
                op_crit.slack,
                cmax - (op_crit.head + op_crit.processing_time + op_crit.tail)
            );
            assert_eq!(op_crit.is_critical, op_crit.slack == 0);
            let sched_op = schedule.operation(op_crit.op_id).expect("scheduled");
            assert_eq!(op_crit.head, sched_op.start_time);
        }

        // 5. Critical blocks use consecutive machine ops, every internal arc is critical, maximal
        let block_again = extract_critical_blocks(&graph, analysis.arcs());
        assert_eq!(analysis.critical_blocks(), block_again.as_slice());

        for blk in analysis.critical_blocks() {
            assert!(blk.len >= 2);
            assert_eq!(blk.operations.len(), blk.len);
            assert_eq!(blk.first_operation, blk.operations[0]);
            assert_eq!(blk.last_operation, blk.operations[blk.len - 1]);
            assert_eq!(blk.critical_machine_arcs.len(), blk.len - 1);

            let m_seq = &graph.machine_sequences()[blk.machine.0].operations;
            let start_pos = m_seq
                .iter()
                .position(|&o| o == blk.first_operation)
                .expect("start pos");
            assert_eq!(&m_seq[start_pos..(start_pos + blk.len)], &blk.operations);

            let m_arcs = graph.machine_order_arcs(blk.machine).expect("m arcs");
            for (offset, &arc_id) in blk.critical_machine_arcs.iter().enumerate() {
                assert_eq!(arc_id, m_arcs[start_pos + offset]);
                assert!(analysis.arcs()[arc_id.0].is_critical);
            }
            if start_pos > 0 {
                assert!(
                    !analysis.arcs()[m_arcs[start_pos - 1].0].is_critical,
                    "block can extend left!"
                );
            }
            if start_pos + blk.len - 1 < m_arcs.len() {
                assert!(
                    !analysis.arcs()[m_arcs[start_pos + blk.len - 1].0].is_critical,
                    "block can extend right!"
                );
            }
        }

        // 6. Append-only semi-active decoder also agrees on graph makespan
        let append_sched = decode_obr_append_only(&instance, &obr).expect("append decode");
        let (_, append_analysis) =
            analyze_schedule(&instance, &append_sched).expect("append analysis");
        assert_eq!(append_analysis.graph_makespan(), append_sched.makespan());

        // 7. Exhaustive Source-to-Sink path enumeration oracle on small instances (J<=3, M<=3)
        if num_jobs <= 3 && num_machines <= 3 {
            let paths = enumerate_source_to_sink_paths(&graph);
            assert!(!paths.is_empty());
            let oracle_cmax = paths.iter().map(|p| p.2).max().expect("max");
            assert_eq!(oracle_cmax, analysis.graph_makespan());

            let mut oracle_crit_ops = BTreeSet::new();
            let mut oracle_crit_arcs = BTreeSet::new();
            for (pv, pa, w) in &paths {
                if *w == oracle_cmax {
                    for &v in pv {
                        if let GraphVertexKind::Operation(op_id) = graph.vertex(v).expect("v").kind
                        {
                            oracle_crit_ops.insert(op_id);
                        }
                    }
                    for &a in pa {
                        oracle_crit_arcs.insert(a);
                    }
                }
            }

            let actual_crit_ops: BTreeSet<OperationId> =
                analysis.critical_operations().iter().copied().collect();
            let actual_crit_arcs: BTreeSet<ArcId> =
                analysis.critical_arcs().iter().map(|a| a.id).collect();
            assert_eq!(actual_crit_ops, oracle_crit_ops);
            assert_eq!(actual_crit_arcs, oracle_crit_arcs);
        }
    }
}
