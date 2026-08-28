use std::collections::BTreeMap;

use vyre_foundation::{
    algebraic_reordering::{reordering_class, ReorderingClass},
    ir::Ident,
    logical::LogicalProgramGraph,
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
#[derive(Debug)]
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
        let program = &node.program;
        let stats = program.stats();
        node_work.push(u64::try_from(stats.node_count).unwrap_or(u64::MAX));
        node_live_values.push(u64::from(stats.register_pressure_estimate));
        let scratch: Vec<(Ident, u64)> = workgroup_scratch_declarations(program).collect();
        let declared = program.workgroup_size;
        let invocations = u64::from(declared[0])
            .saturating_mul(u64::from(declared[1]))
            .saturating_mul(u64::from(declared[2]));
        // Only a schedule-only width may vary. The semantic IR owner classifies
        // geometry observability once so search and logical identity cannot
        // disagree about whether the declaration affects behavior.
        let accepts_width = program.workgroup_size_is_schedule_only();
        node_workgroup_scratch.push(scratch);
        node_declared_invocations.push(invocations.max(1));
        node_declared_workgroup.push(declared);
        node_accepts_width.push(accepts_width);
        node_reordering.push(reordering_class(program));
        let region = logical.region(node.id);
        node_numeric.push(region.map_or(NumericContract::EXACT, |region| region.numeric));
        node_reduction_terms.push(region.map_or(1, |region| {
            u32::try_from(region.max_points).unwrap_or(u32::MAX)
        }));
        node_instructions.push(stats.instruction_count);
        node_barriers.push(stats.barrier_count);
        node_grid_syncs.push(stats.grid_sync_count);
        node_tensor_ops.push(stats.tensor_op_count);
        node_divergent_regions.push(CostCertificate::for_program(program).divergence_score);
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
