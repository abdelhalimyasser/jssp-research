//! PyO3 disjunctive-graph, critical-path, head/tail, and critical-block entrypoints.

use jssp_core::decoder::decode_obr as core_decode_obr;
use jssp_core::graph::{
    CriticalPathAnalysis, DisjunctiveGraph, GraphVertexKind, analyze_disjunctive_graph,
    analyze_schedule,
};
use jssp_core::model::{MachineId, MachineSequence, OperationId};
use jssp_core::representation::{
    AdapterAuditReport, EncodedObrResult, ObrChromosome, encode_graph_to_obr,
    round_trip_machine_sequences as core_round_trip_machine_sequences,
    round_trip_schedule as core_round_trip_schedule,
};
use pyo3::prelude::*;

use crate::instance_api::{jssp_err_to_py, parse_instance};
use crate::result_api::{
    DecodedSchedulePayload, ScheduledOpTuple, py_ops_to_schedule, schedule_to_py_payload,
};

type MetadataTuple = (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
);
type TopoVertexTuple = (usize, &'static str, Option<(usize, usize)>);
type OpCritTuple = (
    (usize, usize),
    usize,
    usize,
    u64,
    (u64, u64, u64, bool),
    Vec<(usize, usize)>,
    Option<(usize, usize)>,
    Vec<(usize, usize)>,
    Option<(usize, usize)>,
);
type AnalyzedArcTuple = (
    usize,
    usize,
    usize,
    Option<(usize, usize)>,
    Option<(usize, usize)>,
    &'static str,
    Option<usize>,
    bool,
    bool,
);
type CriticalSubgraphTuple = (Vec<(usize, usize)>, Vec<usize>, Vec<usize>, Vec<usize>);
type RepPathTuple = (Vec<(usize, usize)>, Vec<usize>, u64);
type CriticalBlockTuple = (
    usize,
    usize,
    Vec<(usize, usize)>,
    (usize, usize),
    (usize, usize),
    usize,
    Vec<usize>,
);

pub type GraphAnalysisPayload = (
    u64,
    u64,
    MetadataTuple,
    Vec<TopoVertexTuple>,
    Vec<OpCritTuple>,
    Vec<AnalyzedArcTuple>,
    CriticalSubgraphTuple,
    RepPathTuple,
    Vec<CriticalBlockTuple>,
);

fn vertex_tuple(graph: &DisjunctiveGraph, vertex_id: usize) -> PyResult<TopoVertexTuple> {
    let v = graph
        .vertex(jssp_core::graph::VertexId(vertex_id))
        .map_err(jssp_err_to_py)?;
    let (kind, op) = match v.kind {
        GraphVertexKind::Source => ("source", None),
        GraphVertexKind::Operation(op_id) => ("operation", Some((op_id.job.0, op_id.op_index.0))),
        GraphVertexKind::Sink => ("sink", None),
    };
    Ok((v.id.0, kind, op))
}

fn analysis_to_py_payload(
    graph: &DisjunctiveGraph,
    analysis: &CriticalPathAnalysis,
) -> PyResult<GraphAnalysisPayload> {
    let meta = graph.metadata();
    let meta_tuple: MetadataTuple = (
        meta.num_jobs,
        meta.num_machines,
        meta.num_operations,
        meta.num_vertices,
        meta.num_arcs,
        meta.num_job_precedence_arcs,
        meta.num_machine_order_arcs,
        meta.num_source_arcs,
        meta.num_sink_arcs,
    );

    let topo_order = analysis
        .topological_order()
        .iter()
        .map(|v| vertex_tuple(graph, v.0))
        .collect::<PyResult<Vec<_>>>()?;

    let ops = analysis
        .operations()
        .iter()
        .map(|op| {
            (
                (op.op_id.job.0, op.op_id.op_index.0),
                op.vertex_id.0,
                op.machine.0,
                op.processing_time,
                (op.head, op.tail, op.slack, op.is_critical),
                op.maximizing_predecessors
                    .iter()
                    .map(|p| (p.job.0, p.op_index.0))
                    .collect(),
                op.representative_predecessor
                    .map(|p| (p.job.0, p.op_index.0)),
                op.maximizing_successors
                    .iter()
                    .map(|s| (s.job.0, s.op_index.0))
                    .collect(),
                op.representative_successor.map(|s| (s.job.0, s.op_index.0)),
            )
        })
        .collect();

    let arcs = analysis
        .arcs()
        .iter()
        .map(|arc| {
            (
                arc.id.0,
                arc.from.0,
                arc.to.0,
                arc.from_op.map(|o| (o.job.0, o.op_index.0)),
                arc.to_op.map(|o| (o.job.0, o.op_index.0)),
                arc.arc_type.kind_str(),
                arc.arc_type.owner_index(),
                arc.is_head_tight,
                arc.is_critical,
            )
        })
        .collect();

    let sub = analysis.critical_subgraph();
    let subgraph_tuple: CriticalSubgraphTuple = (
        sub.critical_operations
            .iter()
            .map(|o| (o.job.0, o.op_index.0))
            .collect(),
        sub.critical_arcs.iter().map(|a| a.0).collect(),
        sub.critical_machine_arcs.iter().map(|a| a.0).collect(),
        sub.critical_job_arcs.iter().map(|a| a.0).collect(),
    );

    let rep = analysis.representative_critical_path();
    let rep_tuple: RepPathTuple = (
        rep.operations
            .iter()
            .map(|o| (o.job.0, o.op_index.0))
            .collect(),
        rep.arcs.iter().map(|a| a.0).collect(),
        rep.total_processing_time,
    );

    let blocks = analysis
        .critical_blocks()
        .iter()
        .map(|b| {
            (
                b.id.0,
                b.machine.0,
                b.operations
                    .iter()
                    .map(|o| (o.job.0, o.op_index.0))
                    .collect(),
                (b.first_operation.job.0, b.first_operation.op_index.0),
                (b.last_operation.job.0, b.last_operation.op_index.0),
                b.len,
                b.critical_machine_arcs.iter().map(|a| a.0).collect(),
            )
        })
        .collect();

    Ok((
        analysis.graph_makespan(),
        analysis.schedule_makespan(),
        meta_tuple,
        topo_order,
        ops,
        arcs,
        subgraph_tuple,
        rep_tuple,
        blocks,
    ))
}

/// Validates a schedule, builds its compact oriented disjunctive graph, and runs complete
/// head/tail, critical-path, critical-subgraph, and critical-block analysis in Rust.
#[pyfunction]
pub fn analyze_schedule_graph(
    jobs: Vec<Vec<(usize, u64)>>,
    operations: Vec<ScheduledOpTuple>,
    makespan: u64,
) -> PyResult<GraphAnalysisPayload> {
    let instance = parse_instance(jobs)?;
    let schedule = py_ops_to_schedule(
        instance.num_jobs(),
        instance.num_machines(),
        operations,
        makespan,
    );
    let (graph, analysis) = analyze_schedule(&instance, &schedule).map_err(jssp_err_to_py)?;
    analysis_to_py_payload(&graph, &analysis)
}

/// Decodes an OBR chromosome with canonical Serial SGS and runs full disjunctive-graph
/// critical-path and critical-block analysis in one coarse-grained FFI call.
#[pyfunction]
pub fn analyze_obr_graph(
    jobs: Vec<Vec<(usize, u64)>>,
    chromosome: Vec<usize>,
) -> PyResult<GraphAnalysisPayload> {
    let instance = parse_instance(jobs)?;
    let obr = ObrChromosome::from_job_indices(&chromosome);
    let schedule = core_decode_obr(&instance, &obr).map_err(jssp_err_to_py)?;
    let (graph, analysis) = analyze_schedule(&instance, &schedule).map_err(jssp_err_to_py)?;
    analysis_to_py_payload(&graph, &analysis)
}

/// Builds a compact oriented disjunctive graph from raw per-machine operation sequences and
/// verifies DAG acyclicity, returning deterministic topological order or raising `ValueError`.
#[pyfunction]
pub fn validate_machine_sequences_dag(
    jobs: Vec<Vec<(usize, u64)>>,
    machine_sequences: Vec<Vec<(usize, usize)>>,
) -> PyResult<Vec<TopoVertexTuple>> {
    let instance = parse_instance(jobs)?;
    let seqs = parse_machine_sequences(machine_sequences);
    let graph = DisjunctiveGraph::from_machine_sequences_unchecked(&instance, &seqs, 0)
        .map_err(jssp_err_to_py)?;
    let order = graph.topological_sort().map_err(jssp_err_to_py)?;
    order
        .into_iter()
        .map(|v| vertex_tuple(&graph, v.0))
        .collect()
}

type EncodedObrTuple = (
    Vec<usize>,
    Vec<(usize, usize)>,
    Vec<(usize, usize, u64, u64)>,
    usize,
    Vec<usize>,
    bool,
    bool,
);

type AuditHashesTuple = (String, String, String);
type AuditMakespansTuple = (u64, u64, i128);
type AuditShiftsTuple = (usize, usize, usize);
type AuditMachineTuple = (usize, usize, bool);
type AuditValidityTuple = (bool, bool, bool, bool);
type AdapterAuditTuple = (
    AuditHashesTuple,
    AuditMakespansTuple,
    AuditShiftsTuple,
    AuditMachineTuple,
    AuditValidityTuple,
    &'static str,
    usize,
);
pub type RoundTripPayload = (EncodedObrTuple, DecodedSchedulePayload, AdapterAuditTuple);

fn parse_machine_sequences(raw: Vec<Vec<(usize, usize)>>) -> Vec<MachineSequence> {
    raw.into_iter()
        .enumerate()
        .map(|(m, ops)| {
            let op_ids = ops
                .into_iter()
                .map(|(j, k)| OperationId::from_indices(j, k))
                .collect();
            MachineSequence::new(MachineId(m), op_ids)
        })
        .collect()
}

fn encoded_obr_to_py(encoded: &EncodedObrResult) -> EncodedObrTuple {
    (
        encoded.chromosome.to_job_indices(),
        encoded
            .canonical_operations
            .iter()
            .map(|o| (o.job.0, o.op_index.0))
            .collect(),
        encoded
            .operation_keys
            .iter()
            .map(|k| (k.job, k.op_index, k.head, k.completion))
            .collect(),
        encoded.chromosome_length,
        encoded.job_multiplicities.clone(),
        encoded.technological_order_preserved,
        encoded.obr_valid,
    )
}

fn audit_to_py(audit: &AdapterAuditReport) -> AdapterAuditTuple {
    (
        (
            audit.target_graph_hash.clone(),
            audit.generated_obr_hash.clone(),
            audit.decoded_schedule_hash.clone(),
        ),
        (
            audit.target_graph_makespan,
            audit.decoded_schedule_makespan,
            audit.signed_makespan_diff,
        ),
        (
            audit.adapter_left_shift_count,
            audit.adapter_right_shift_count,
            audit.adapter_unchanged_start_count,
        ),
        (
            audit.machine_order_inversion_count,
            audit.mismatched_machine_arc_count,
            audit.exact_machine_order_preserved,
        ),
        (
            audit.exact_start_times_preserved,
            audit.exact_completion_times_preserved,
            audit.obr_valid,
            audit.technological_order_preserved,
        ),
        audit.status.as_str(),
        audit.total_operation_count,
    )
}

/// Validates complete target machine sequences, builds an acyclic disjunctive graph with
/// exact longest-path makespan, and runs M4 critical-path/block analysis in Rust.
#[pyfunction]
pub fn build_machine_sequence_graph(
    jobs: Vec<Vec<(usize, u64)>>,
    machine_sequences: Vec<Vec<(usize, usize)>>,
) -> PyResult<GraphAnalysisPayload> {
    let instance = parse_instance(jobs)?;
    let seqs = parse_machine_sequences(machine_sequences);
    let graph = DisjunctiveGraph::from_target_machine_sequences(&instance, &seqs)
        .map_err(jssp_err_to_py)?;
    let analysis = analyze_disjunctive_graph(&graph).map_err(jssp_err_to_py)?;
    analysis_to_py_payload(&graph, &analysis)
}

/// Validates target machine sequences and encodes the resulting acyclic DAG into a canonical
/// `D01-B` OBR chromosome, returning chromosome genes, canonical operations, and sort keys.
#[pyfunction]
pub fn encode_machine_sequences_to_obr(
    jobs: Vec<Vec<(usize, u64)>>,
    machine_sequences: Vec<Vec<(usize, usize)>>,
) -> PyResult<EncodedObrTuple> {
    let instance = parse_instance(jobs)?;
    let seqs = parse_machine_sequences(machine_sequences);
    let graph = DisjunctiveGraph::from_target_machine_sequences(&instance, &seqs)
        .map_err(jssp_err_to_py)?;
    let analysis = analyze_disjunctive_graph(&graph).map_err(jssp_err_to_py)?;
    let encoded = encode_graph_to_obr(&instance, &graph, &analysis).map_err(jssp_err_to_py)?;
    Ok(encoded_obr_to_py(&encoded))
}

/// Runs the complete validated machine-sequence -> canonical OBR -> M2 Serial SGS decode ->
/// audit report pipeline in Rust.
#[pyfunction]
pub fn round_trip_machine_sequences(
    jobs: Vec<Vec<(usize, u64)>>,
    machine_sequences: Vec<Vec<(usize, usize)>>,
) -> PyResult<RoundTripPayload> {
    let instance = parse_instance(jobs)?;
    let seqs = parse_machine_sequences(machine_sequences);
    let bundle = core_round_trip_machine_sequences(&instance, &seqs).map_err(jssp_err_to_py)?;
    Ok((
        encoded_obr_to_py(&bundle.encoded),
        schedule_to_py_payload(&bundle.decoded_schedule),
        audit_to_py(&bundle.audit),
    ))
}

/// Decodes an OBR chromosome with M2 Serial SGS and audits its canonical D01-B round-trip back
/// through the disjunctive graph and M2 decoder.
#[pyfunction]
pub fn round_trip_obr_schedule(
    jobs: Vec<Vec<(usize, u64)>>,
    chromosome: Vec<usize>,
) -> PyResult<RoundTripPayload> {
    let instance = parse_instance(jobs)?;
    let obr = ObrChromosome::from_job_indices(&chromosome);
    let schedule = core_decode_obr(&instance, &obr).map_err(jssp_err_to_py)?;
    let bundle = core_round_trip_schedule(&instance, &schedule).map_err(jssp_err_to_py)?;
    Ok((
        encoded_obr_to_py(&bundle.encoded),
        schedule_to_py_payload(&bundle.decoded_schedule),
        audit_to_py(&bundle.audit),
    ))
}
