use std::collections::BTreeMap;

use vyre_foundation::{
    algebraic_reordering::{reordering_class, ReorderingClass},
    ir::Ident,
    ir::Program,
    logical::{LogicalProgramGraph, LogicalRegion},
    numeric::NumericContract,
    optimizer::cost::CostCertificate,
};

use vyre_foundation::ir::ValueLifetime;

use crate::allocation::ValueLiveness;
use crate::{
    value_byte_count, workgroup_scratch_declarations, ArtifactNodeId, ArtifactValueId,
    CompileError, DependencyEdge, DependencyEndpoint, DependencyKind,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DataflowEdge {
    pub(crate) from: ArtifactNodeId,
    pub(crate) to: ArtifactNodeId,
    pub(crate) value: ArtifactValueId,
}

/// Per-node and per-value measurements the cost model and the workgroup search
/// read. Every field is derived from the graph, never from a candidate, so one
/// derivation serves every candidate the search scores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PlanningFacts {
    /// Semantic IR nodes in each node's program.
    pub(crate) node_work: Vec<u64>,
    /// Simultaneously live values in each node's program, from the foundation
    /// register-pressure estimate.
    pub(crate) node_live_values: Vec<u64>,
    /// Workgroup-scoped scratch each node declares, one entry per buffer.
    ///
    /// Held per buffer rather than as a total because fusion unions buffers by
    /// name: two members that declare the same tile share it in the generated
    /// kernel, and a total cannot say which bytes are the same bytes.
    pub(crate) node_workgroup_scratch: Vec<Vec<(Ident, u64)>>,
    /// Invocations per workgroup each node's program declares.
    pub(crate) node_declared_invocations: Vec<u64>,
    /// Workgroup dimensions each node's program declares.
    pub(crate) node_declared_workgroup: Vec<[u32; 3]>,
    /// Whether a node's program stays correct under a different launch width.
    pub(crate) node_accepts_width: Vec<bool>,
    /// Whether a schedule may reorder how each node's invocations combine into
    /// a shared location.
    ///
    /// A rounding accumulation is order-dependent, so a schedule that changes
    /// the order computes a different number. The semantic IR owner classifies
    /// the combines once, from operator laws and element types, so search never
    /// re-derives numerics from a candidate.
    pub(crate) node_reordering: Vec<ReorderingClass>,
    /// Numeric contract each region states, derived by the logical stage.
    ///
    /// A schedule that reorders a combine is legal over a rounding accumulation
    /// only where a stated budget admits the error the new order produces, so
    /// the contract travels with the region rather than being re-derived per
    /// candidate.
    pub(crate) node_numeric: Vec<NumericContract>,
    /// Values each region combines into one output point.
    ///
    /// A reduction over more points rounds more often, so the count decides how
    /// much a reordered schedule costs.
    pub(crate) node_reduction_terms: Vec<u32>,
    /// Instructions each node's program states.
    ///
    /// The count is static: a loop body counts once, because no analysis here
    /// bounds its trip count. It is priced only against a device that reported
    /// an instruction rate, and recorded as evidence otherwise.
    pub(crate) node_instructions: Vec<u64>,
    /// Workgroup-scoped rendezvous each node's program states.
    pub(crate) node_barriers: Vec<u64>,
    /// Whole-grid rendezvous each node's program states.
    pub(crate) node_grid_syncs: Vec<u64>,
    /// Matrix-engine statements each node's program states.
    pub(crate) node_tensor_ops: Vec<u64>,
    /// Lane-gated regions each node's program states.
    ///
    /// A region entered by one lane of a subgroup leaves the rest idle for its
    /// duration, so the count bounds the lanes a candidate wastes from below.
    pub(crate) node_divergent_regions: Vec<u64>,
    /// Bytes of graph values each node produces or consumes.
    ///
    /// A value shared by two nodes counts for both, which is the traffic they
    /// each move when they are not fused. The occupancy term prices a group's
    /// traffic a second time when the group exceeds a device budget, so it reads
    /// this rather than the whole-graph byte total.
    pub(crate) node_touched_bytes: Vec<u64>,
    /// Producer-consumer value edges the search may fuse.
    pub(crate) dataflow: Vec<DataflowEdge>,
    /// Packed byte length of every graph value, keyed by artifact value id.
    pub(crate) value_bytes: BTreeMap<u32, u64>,
    /// What every graph value contributes to the resident byte total, ready for
    /// one candidate's grouping to resolve into stages.
    pub(crate) value_liveness: Vec<ValueLiveness>,
}

/// Everything one node's program contributes to the facts.
///
/// Held as one record so the walk over the graph and a law-derived rewrite of
/// one node are measured by the same code. A second measurement site is how a
/// derived alternative gets priced by a rule the baseline never saw.
pub(crate) struct NodeMeasurement {
    pub(crate) work: u64,
    pub(crate) live_values: u64,
    pub(crate) workgroup_scratch: Vec<(Ident, u64)>,
    pub(crate) declared_invocations: u64,
    pub(crate) declared_workgroup: [u32; 3],
    pub(crate) accepts_width: bool,
    pub(crate) reordering: ReorderingClass,
    pub(crate) numeric: NumericContract,
    pub(crate) reduction_terms: u32,
    pub(crate) instructions: u64,
    pub(crate) barriers: u64,
    pub(crate) grid_syncs: u64,
    pub(crate) tensor_ops: u64,
    pub(crate) divergent_regions: u64,
}

/// Measure one node's program against the logical region it states.
pub(crate) fn measure_node(program: &Program, region: Option<&LogicalRegion>) -> NodeMeasurement {
    let stats = program.stats();
    let declared = program.workgroup_size;
    let invocations = u64::from(declared[0])
        .saturating_mul(u64::from(declared[1]))
        .saturating_mul(u64::from(declared[2]));
    NodeMeasurement {
        work: u64::try_from(stats.node_count).unwrap_or(u64::MAX),
        live_values: u64::from(stats.register_pressure_estimate),
        workgroup_scratch: workgroup_scratch_declarations(program).collect(),
        declared_invocations: invocations.max(1),
        declared_workgroup: declared,
        // Only a schedule-only width may vary. The semantic IR owner classifies
        // geometry observability once so search and logical identity cannot
        // disagree about whether the declaration affects behavior.
        accepts_width: program.workgroup_size_is_schedule_only(),
        reordering: reordering_class(program),
        numeric: region.map_or(NumericContract::EXACT, |region| region.numeric),
        reduction_terms: region.map_or(1, |region| {
            u32::try_from(region.max_points).unwrap_or(u32::MAX)
        }),
        instructions: stats.instruction_count,
        barriers: stats.barrier_count,
        grid_syncs: stats.grid_sync_count,
        tensor_ops: stats.tensor_op_count,
        divergent_regions: CostCertificate::for_program(program).divergence_score,
    }
}

impl PlanningFacts {
    /// These facts with one node measured from a rewritten program.
    ///
    /// A law-derived alternative differs from the baseline in one node's
    /// program, so every other measurement is the one already derived and the
    /// graph-level totals do not move: a rewrite inside a node changes neither
    /// the connected values nor their packed bytes.
    pub(crate) fn with_node_measurement(&self, node: usize, measured: NodeMeasurement) -> Self {
        let mut facts = self.clone();
        facts.node_work[node] = measured.work;
        facts.node_live_values[node] = measured.live_values;
        facts.node_workgroup_scratch[node] = measured.workgroup_scratch;
        facts.node_declared_invocations[node] = measured.declared_invocations;
        facts.node_declared_workgroup[node] = measured.declared_workgroup;
        facts.node_accepts_width[node] = measured.accepts_width;
        facts.node_reordering[node] = measured.reordering;
        facts.node_numeric[node] = measured.numeric;
        facts.node_reduction_terms[node] = measured.reduction_terms;
        facts.node_instructions[node] = measured.instructions;
        facts.node_barriers[node] = measured.barriers;
        facts.node_grid_syncs[node] = measured.grid_syncs;
        facts.node_tensor_ops[node] = measured.tensor_ops;
        facts.node_divergent_regions[node] = measured.divergent_regions;
        facts
    }
}

pub(crate) fn derive(
    logical: &LogicalProgramGraph<'_>,
    dependencies: &[DependencyEdge],
    bindings: &BTreeMap<String, u64>,
) -> Result<PlanningFacts, CompileError> {
    let graph = logical.graph();
    let node_count = graph.nodes().len();
    let mut node_work = Vec::with_capacity(node_count);
    let mut node_live_values = Vec::with_capacity(node_count);
    let mut node_workgroup_scratch = Vec::with_capacity(node_count);
    let mut node_declared_invocations = Vec::with_capacity(node_count);
    let mut node_declared_workgroup = Vec::with_capacity(node_count);
    let mut node_accepts_width = Vec::with_capacity(node_count);
    let mut node_reordering = Vec::with_capacity(node_count);
    let mut node_numeric = Vec::with_capacity(node_count);
    let mut node_reduction_terms = Vec::with_capacity(node_count);
    let mut node_instructions = Vec::with_capacity(node_count);
    let mut node_barriers = Vec::with_capacity(node_count);
    let mut node_grid_syncs = Vec::with_capacity(node_count);
    let mut node_tensor_ops = Vec::with_capacity(node_count);
    let mut node_divergent_regions = Vec::with_capacity(node_count);
    for node in graph.nodes() {
        let measured = measure_node(&node.program, logical.region(node.id));
        node_work.push(measured.work);
        node_live_values.push(measured.live_values);
        node_workgroup_scratch.push(measured.workgroup_scratch);
        node_declared_invocations.push(measured.declared_invocations);
        node_declared_workgroup.push(measured.declared_workgroup);
        node_accepts_width.push(measured.accepts_width);
        node_reordering.push(measured.reordering);
        node_numeric.push(measured.numeric);
        node_reduction_terms.push(measured.reduction_terms);
        node_instructions.push(measured.instructions);
        node_barriers.push(measured.barriers);
        node_grid_syncs.push(measured.grid_syncs);
        node_tensor_ops.push(measured.tensor_ops);
        node_divergent_regions.push(measured.divergent_regions);
    }
    let dataflow = dependencies
        .iter()
        .filter_map(|edge| match (edge.from, edge.to, edge.kind, edge.value) {
            (
                DependencyEndpoint::Node(from),
                DependencyEndpoint::Node(to),
                DependencyKind::Data,
                Some(value),
            ) => Some(DataflowEdge { from, to, value }),
            _ => None,
        })
        .collect();
    let mut value_bytes = BTreeMap::new();
    let mut value_liveness = Vec::with_capacity(graph.values().len());
    let mut node_touched_bytes = vec![0_u64; node_count];
    for value in graph.values() {
        let bytes = value_byte_count(value, bindings)?;
        value_bytes.insert(value.id.0, bytes);
        value_liveness.push(ValueLiveness {
            value: ArtifactValueId(value.id.0),
            bytes,
            producer: value.producer.map(|producer| ArtifactNodeId(producer.0)),
            consumers: value
                .consumers
                .iter()
                .map(|consumer| ArtifactNodeId(consumer.0))
                .collect(),
            survives_to_end: matches!(
                value.contract.lifetime,
                ValueLifetime::Output | ValueLifetime::Retained
            ),
        });
        for node in value.producer.iter().chain(value.consumers.iter()) {
            if let Some(total) = node_touched_bytes.get_mut(node.0 as usize) {
                *total = total.saturating_add(bytes);
            }
        }
    }
    Ok(PlanningFacts {
        node_work,
        node_live_values,
        node_workgroup_scratch,
        node_declared_invocations,
        node_declared_workgroup,
        node_accepts_width,
        node_reordering,
        node_numeric,
        node_reduction_terms,
        node_instructions,
        node_barriers,
        node_grid_syncs,
        node_tensor_ops,
        node_divergent_regions,
        node_touched_bytes,
        dataflow,
        value_bytes,
        value_liveness,
    })
}
