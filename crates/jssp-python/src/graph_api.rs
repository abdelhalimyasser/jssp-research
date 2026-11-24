//! PyO3 disjunctive-graph, critical-path, head/tail, and critical-block entrypoints.

use jssp_core::decoder::decode_obr as core_decode_obr;
use jssp_core::graph::{CriticalPathAnalysis, DisjunctiveGraph, GraphVertexKind, analyze_schedule};
use jssp_core::model::{MachineId, MachineSequence, OperationId};
use jssp_core::representation::ObrChromosome;
use pyo3::prelude::*;

use crate::instance_api::{jssp_err_to_py, parse_instance};
use crate::result_api::{ScheduledOpTuple, py_ops_to_schedule};

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
    let seqs: Vec<MachineSequence> = machine_sequences
        .into_iter()
        .enumerate()
        .map(|(m, ops)| {
            let op_ids = ops
                .into_iter()
                .map(|(j, k)| OperationId::from_indices(j, k))
                .collect();
            MachineSequence::new(MachineId(m), op_ids)
        })
        .collect();
    let graph = DisjunctiveGraph::from_machine_sequences_unchecked(&instance, &seqs, 0)
        .map_err(jssp_err_to_py)?;
    let order = graph.topological_sort().map_err(jssp_err_to_py)?;
    order
        .into_iter()
        .map(|v| vertex_tuple(&graph, v.0))
        .collect()
}
