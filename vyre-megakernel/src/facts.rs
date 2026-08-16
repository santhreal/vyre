use std::collections::BTreeMap;

use vyre_foundation::ir::ProgramGraph;

use crate::{
    value_byte_count, workgroup_scratch_bytes, ArtifactNodeId, ArtifactValueId, CompileError,
    DependencyEdge, DependencyEndpoint, DependencyKind,
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
    /// Workgroup-scoped scratch bytes each node declares.
    pub(crate) node_shared_scratch_bytes: Vec<u64>,
    /// Invocations per workgroup each node's program declares.
    pub(crate) node_declared_invocations: Vec<u64>,
    /// Workgroup dimensions each node's program declares.
    pub(crate) node_declared_workgroup: Vec<[u32; 3]>,
    /// Whether a node's program stays correct under a different launch width.
    pub(crate) node_accepts_width: Vec<bool>,
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
}

pub(crate) fn derive(
    graph: &ProgramGraph,
    dependencies: &[DependencyEdge],
    bindings: &BTreeMap<String, u64>,
) -> Result<PlanningFacts, CompileError> {
    let node_count = graph.nodes().len();
    let mut node_work = Vec::with_capacity(node_count);
    let mut node_live_values = Vec::with_capacity(node_count);
    let mut node_shared_scratch_bytes = Vec::with_capacity(node_count);
    let mut node_declared_invocations = Vec::with_capacity(node_count);
    let mut node_declared_workgroup = Vec::with_capacity(node_count);
    let mut node_accepts_width = Vec::with_capacity(node_count);
    for node in graph.nodes() {
        let program = &node.program;
        let stats = program.stats();
        node_work.push(u64::try_from(stats.node_count).unwrap_or(u64::MAX));
        node_live_values.push(u64::from(stats.register_pressure_estimate));
        let scratch = workgroup_scratch_bytes(program);
        let declared = program.workgroup_size;
        let invocations = u64::from(declared[0])
            .saturating_mul(u64::from(declared[1]))
            .saturating_mul(u64::from(declared[2]));
        // A launch width is safe to replace only when nothing in the program
        // observes it. A workgroup-scoped buffer is sized for the declared
        // width, a barrier orders invocations inside the declared group, and a
        // subgroup operation reads the group's lane layout, so each one pins the
        // declared shape. A 2D or 3D declaration pins it too: the search only
        // proposes 1D widths.
        let accepts_width = declared[1] == 1
            && declared[2] == 1
            && scratch == 0
            && !stats.has_node_barrier()
            && !stats.subgroup_ops()
            && !program.non_composable_with_self;
        node_shared_scratch_bytes.push(scratch);
        node_declared_invocations.push(invocations.max(1));
        node_declared_workgroup.push(declared);
        node_accepts_width.push(accepts_width);
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
    let mut node_touched_bytes = vec![0_u64; node_count];
    for value in graph.values() {
        let bytes = value_byte_count(value, bindings)?;
        value_bytes.insert(value.id.0, bytes);
        for node in value.producer.iter().chain(value.consumers.iter()) {
            if let Some(total) = node_touched_bytes.get_mut(node.0 as usize) {
                *total = total.saturating_add(bytes);
            }
        }
    }
    Ok(PlanningFacts {
        node_work,
        node_live_values,
        node_shared_scratch_bytes,
        node_declared_invocations,
        node_declared_workgroup,
        node_accepts_width,
        node_touched_bytes,
        dataflow,
        value_bytes,
    })
}
