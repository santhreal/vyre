//! Versioned logical algorithm stage between graph topology and schedule search.
//!
//! A [`ProgramGraph`] states whole-program values and dependencies. This module
//! derives the schedule-free region contracts the compiler searches over: typed
//! extents, logical axes, effects, layouts, and bounds. The source `Program`
//! remains available during migration, but its workgroup size is excluded from
//! semantic identity. A selected schedule records that choice separately.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use thiserror::Error;

use crate::ir::{
    stats::{NODE_KIND_ALL_REDUCE, NODE_KIND_REDUCE_SCATTER, NODE_KIND_TILE_REDUCE},
    BufferAccess, GraphNodeId, GraphValueId, ProgramGraph, ShapeDim, ValueLifetime,
};
use crate::operation::OperationEffects;

/// Current logical algorithm schema and identity version.
pub const LOGICAL_ALGORITHM_VERSION: u16 = 2;

/// One validated logical extent.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum LogicalExtent {
    /// Compile-time extent.
    Static(u64),
    /// Dynamic extent read from a typed graph value contract and resolved by the compile request.
    GraphValue {
        /// Graph value whose contract declares this dimension.
        value: u32,
        /// Zero-based dimension within the graph value.
        axis: u32,
        /// Symbol name as declared by the graph value contract.
        symbol: String,
        /// Value bound by the compile request.
        bound: u64,
    },
}

impl LogicalExtent {
    fn bound(&self) -> u64 {
        match self {
            Self::Static(value) | Self::GraphValue { bound: value, .. } => *value,
        }
    }
}

/// Semantic parallelism of one region before physical mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum LogicalRegionKind {
    /// Independent points in an iteration domain.
    Parallel,
    /// Ordered points with a loop-carried dependence.
    Sequential,
    /// Associative combination over one or more axes.
    Reduction,
    /// State retained from one submission to the next.
    RetainedState,
}

/// Logical index projection, independent of lanes and workgroups.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct LogicalIndexMap {
    /// Axis names in declaration order.
    pub axes: Vec<String>,
    /// Row-major strides in elements.
    pub row_major_strides: Vec<u64>,
}

/// Tensor storage layout attached to one logical domain.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct LogicalLayout {
    /// Logical axes in increasing physical storage order.
    pub storage_order: Vec<u32>,
    /// Element strides in logical-axis order.
    pub strides: Vec<u64>,
    /// Whether every axis is densely row-major.
    pub contiguous: bool,
}

/// Closed alias facts for one logical region.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub struct LogicalAliasFacts {
    /// Retained outputs paired with the prior value whose storage they replace.
    pub retained_successors: Vec<(u32, u32)>,
    /// Graph values updated in place by a read-write input binding.
    pub in_place_values: Vec<u32>,
    /// Input values are pairwise distinct.
    pub inputs_disjoint: bool,
    /// Output values are pairwise distinct.
    pub outputs_disjoint: bool,
}

/// Kind of schedule-free dependence between logical regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum LogicalDependenceKind {
    /// A producer writes values consumed by this region.
    Flow,
    /// This region advances state produced by an earlier submission.
    RetainedState,
}

/// One explicit dependence on a preceding logical region.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogicalDependence {
    /// Preceding graph node.
    pub predecessor: GraphNodeId,
    /// Graph values that induce the dependence.
    pub values: Vec<u32>,
    /// Dependence semantics.
    pub kind: LogicalDependenceKind,
}

/// Closed read/write and synchronization effects for one logical region.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub struct LogicalEffects {
    /// Graph values read by this region.
    pub reads: Vec<u32>,
    /// Graph values written by this region.
    pub writes: Vec<u32>,
    /// Whether this region updates retained state.
    pub retained_state: bool,
    /// Whether the region contains atomic memory effects.
    pub atomics: bool,
    /// Whether the region contains an ordering or collective synchronization effect.
    pub synchronizes: bool,
}

/// One validated schedule-free algorithm region.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogicalRegion {
    /// Source graph node.
    pub node: GraphNodeId,
    /// Stable semantic node name.
    pub name: String,
    /// Logical execution kind.
    pub kind: LogicalRegionKind,
    /// Versioned domain extents.
    pub extents: Vec<LogicalExtent>,
    /// Logical index projection.
    pub index_map: LogicalIndexMap,
    /// Tensor storage layout.
    pub layout: LogicalLayout,
    /// Logical axes combined by a reduction region.
    pub reduction_axes: Vec<u32>,
    /// Closed alias facts derived from graph ports.
    pub aliases: LogicalAliasFacts,
    /// Explicit dependencies on earlier regions.
    pub dependencies: Vec<LogicalDependence>,
    /// Closed effects derived from graph ports and executable semantics.
    pub effects: LogicalEffects,
    /// Exact upper bound on logical points.
    pub max_points: u64,
}

/// A graph plus validated logical regions and schedule-free canonical identity.
#[derive(Debug)]
pub struct LogicalProgramGraph<'a> {
    graph: &'a ProgramGraph,
    regions: Vec<LogicalRegion>,
    semantic_wire: Vec<u8>,
}

impl<'a> LogicalProgramGraph<'a> {
    /// Validate and derive the logical algorithm stage.
    pub fn validate(
        graph: &'a ProgramGraph,
        bindings: &BTreeMap<String, u64>,
    ) -> Result<Self, LogicalProgramError> {
        graph
            .analyze()
            .map_err(|error| LogicalProgramError::Graph(error.to_string()))?;

        let required = graph
            .values()
            .iter()
            .flat_map(|value| &value.contract.shape)
            .filter_map(|dim| match dim {
                ShapeDim::Symbol(symbol) => Some(symbol.as_str()),
                ShapeDim::Known(_) => None,
            })
            .collect::<BTreeSet<_>>();
        for symbol in &required {
            if !bindings.contains_key(*symbol) {
                return Err(LogicalProgramError::MissingSymbol((*symbol).to_owned()));
            }
        }
        for symbol in bindings.keys() {
            if !required.contains(symbol.as_str()) {
                return Err(LogicalProgramError::UnexpectedSymbol(symbol.clone()));
            }
        }

        let mut regions = Vec::with_capacity(graph.nodes().len());
        for node in graph.nodes() {
            let (shape, source_value) =
                if let Some((port, value)) = node.output_ports.first().zip(node.outputs.first()) {
                    (port.contract.shape.as_slice(), Some(*value))
                } else if let Some(port) = node.inputs.first() {
                    (port.contract.shape.as_slice(), Some(port.value))
                } else {
                    (&[][..], None)
                };
            let extents = shape
                .iter()
                .enumerate()
                .map(|(axis, dim)| {
                    let value = source_value.ok_or(LogicalProgramError::MissingDomain(node.id))?;
                    let axis = u32::try_from(axis)
                        .map_err(|_| LogicalProgramError::DomainRankOverflow(node.id))?;
                    match dim {
                        ShapeDim::Known(0) => Err(LogicalProgramError::UnresolvedExtent {
                            node: node.id,
                            value,
                            axis,
                        }),
                        ShapeDim::Known(bound) => Ok(LogicalExtent::Static(*bound)),
                        ShapeDim::Symbol(symbol) => {
                            let bound = bindings.get(symbol).copied().ok_or_else(|| {
                                LogicalProgramError::MissingSymbol(symbol.clone())
                            })?;
                            if bound == 0 {
                                return Err(LogicalProgramError::UnresolvedExtent {
                                    node: node.id,
                                    value,
                                    axis,
                                });
                            }
                            Ok(LogicalExtent::GraphValue {
                                value: value.0,
                                axis,
                                symbol: symbol.clone(),
                                bound,
                            })
                        }
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let max_points = extents.iter().try_fold(1u64, |product, extent| {
                product
                    .checked_mul(extent.bound())
                    .ok_or(LogicalProgramError::ExtentOverflow(node.id))
            })?;
            let row_major_strides = row_major_strides(&extents, node.id)?;
            let reads = node
                .inputs
                .iter()
                .filter(|input| input.contract.access != BufferAccess::WriteOnly)
                .map(|input| input.value.0)
                .collect::<Vec<_>>();
            let in_place_values = node
                .inputs
                .iter()
                .filter(|input| {
                    matches!(
                        input.contract.access,
                        BufferAccess::ReadWrite | BufferAccess::WriteOnly
                    )
                })
                .map(|input| input.value.0)
                .collect::<Vec<_>>();
            let mut writes = in_place_values.clone();
            writes.extend(
                node.output_ports
                    .iter()
                    .zip(&node.outputs)
                    .filter(|(output, _)| output.contract.access != BufferAccess::ReadOnly)
                    .map(|(_, value)| value.0),
            );
            writes.sort_unstable();
            writes.dedup();
            let retained_successors = node
                .output_ports
                .iter()
                .zip(&node.outputs)
                .filter_map(|(port, output)| {
                    port.retained_successor_of.map(|prior| (output.0, prior.0))
                })
                .collect::<Vec<_>>();
            let retained_state = !retained_successors.is_empty()
                || node.output_ports.iter().any(|port| {
                    matches!(
                        port.contract.lifetime,
                        ValueLifetime::Retained | ValueLifetime::Output
                    ) && port.retained_successor_of.is_some()
                })
                || node.inputs.iter().any(|port| {
                    port.contract.lifetime == ValueLifetime::Retained
                        && port.contract.access == BufferAccess::ReadWrite
                });
            let program_effects = OperationEffects::from_program(&node.program);
            let reduction_mask =
                NODE_KIND_ALL_REDUCE | NODE_KIND_REDUCE_SCATTER | NODE_KIND_TILE_REDUCE;
            let kind = if retained_state {
                LogicalRegionKind::RetainedState
            } else if node.program.stats().has_any_node_kind(reduction_mask) {
                LogicalRegionKind::Reduction
            } else if node.program.stats().control_flow_count > 0
                || program_effects.atomics
                || program_effects.synchronizes
            {
                LogicalRegionKind::Sequential
            } else {
                LogicalRegionKind::Parallel
            };

            let mut dependence_values = BTreeMap::<GraphNodeId, Vec<u32>>::new();
            for input in &node.inputs {
                let value = graph
                    .values()
                    .get(input.value.0 as usize)
                    .ok_or(LogicalProgramError::MissingDomainValue(input.value))?;
                if let Some(predecessor) = value.producer {
                    if predecessor >= node.id {
                        return Err(LogicalProgramError::CyclicDomain {
                            node: node.id,
                            predecessor,
                        });
                    }
                    dependence_values
                        .entry(predecessor)
                        .or_default()
                        .push(input.value.0);
                }
            }
            let dependencies = dependence_values
                .into_iter()
                .map(|(predecessor, values)| LogicalDependence {
                    predecessor,
                    kind: if retained_state {
                        LogicalDependenceKind::RetainedState
                    } else {
                        LogicalDependenceKind::Flow
                    },
                    values,
                })
                .collect::<Vec<_>>();
            let inputs_disjoint = node
                .inputs
                .iter()
                .map(|input| input.value.0)
                .collect::<BTreeSet<_>>()
                .len()
                == node.inputs.len();
            let outputs_disjoint = node
                .outputs
                .iter()
                .map(|value| value.0)
                .collect::<BTreeSet<_>>()
                .len()
                == node.outputs.len();
            if !inputs_disjoint || !outputs_disjoint {
                return Err(LogicalProgramError::IncompatibleAliases(node.id));
            }
            let storage_order = (0..shape.len())
                .map(|axis| {
                    u32::try_from(axis)
                        .map_err(|_| LogicalProgramError::DomainRankOverflow(node.id))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let reduction_axes = if kind == LogicalRegionKind::Reduction {
                storage_order.clone()
            } else {
                Vec::new()
            };
            regions.push(LogicalRegion {
                node: node.id,
                name: node.name.clone(),
                kind,
                extents,
                index_map: LogicalIndexMap {
                    axes: (0..shape.len()).map(|axis| format!("axis{axis}")).collect(),
                    row_major_strides: row_major_strides.clone(),
                },
                layout: LogicalLayout {
                    storage_order,
                    strides: row_major_strides,
                    contiguous: true,
                },
                reduction_axes,
                aliases: LogicalAliasFacts {
                    retained_successors,
                    in_place_values,
                    inputs_disjoint,
                    outputs_disjoint,
                },
                dependencies,
                effects: LogicalEffects {
                    reads,
                    writes,
                    retained_state,
                    atomics: program_effects.atomics,
                    synchronizes: program_effects.synchronizes,
                },
                max_points,
            });
        }

        #[derive(Serialize)]
        struct IdentityDependence<'b> {
            predecessor: u32,
            values: &'b [u32],
            kind: LogicalDependenceKind,
        }
        #[derive(Serialize)]
        struct IdentityRegion<'b> {
            node: u32,
            name: &'b str,
            kind: LogicalRegionKind,
            extents: &'b [LogicalExtent],
            index_map: &'b LogicalIndexMap,
            layout: &'b LogicalLayout,
            reduction_axes: &'b [u32],
            aliases: &'b LogicalAliasFacts,
            dependencies: Vec<IdentityDependence<'b>>,
            effects: &'b LogicalEffects,
            max_points: u64,
        }
        #[derive(Serialize)]
        struct Identity<'b> {
            version: u16,
            regions: Vec<IdentityRegion<'b>>,
            graph: &'b [u8],
        }
        let graph_wire = graph
            .logical_wire()
            .map_err(|error| LogicalProgramError::CanonicalGraph(error.to_string()))?;
        let identity_regions = regions
            .iter()
            .map(|region| IdentityRegion {
                node: region.node.0,
                name: &region.name,
                kind: region.kind,
                extents: &region.extents,
                index_map: &region.index_map,
                layout: &region.layout,
                reduction_axes: &region.reduction_axes,
                aliases: &region.aliases,
                dependencies: region
                    .dependencies
                    .iter()
                    .map(|dependence| IdentityDependence {
                        predecessor: dependence.predecessor.0,
                        values: &dependence.values,
                        kind: dependence.kind,
                    })
                    .collect(),
                effects: &region.effects,
                max_points: region.max_points,
            })
            .collect();
        let semantic_wire = serde_json::to_vec(&Identity {
            version: LOGICAL_ALGORITHM_VERSION,
            regions: identity_regions,
            graph: &graph_wire,
        })
        .map_err(|error| LogicalProgramError::Identity(error.to_string()))?;

        Ok(Self {
            graph,
            regions,
            semantic_wire,
        })
    }

    /// Borrow the whole-program graph this logical stage was derived from.
    #[must_use]
    pub const fn graph(&self) -> &'a ProgramGraph {
        self.graph
    }

    /// Borrow validated logical regions in graph-node order.
    #[must_use]
    pub fn regions(&self) -> &[LogicalRegion] {
        &self.regions
    }

    /// Canonical schedule-free identity bytes.
    #[must_use]
    pub fn semantic_wire(&self) -> &[u8] {
        &self.semantic_wire
    }
}

fn row_major_strides(
    extents: &[LogicalExtent],
    node: GraphNodeId,
) -> Result<Vec<u64>, LogicalProgramError> {
    let mut strides = vec![0; extents.len()];
    let mut stride = 1u64;
    for (axis, extent) in extents.iter().enumerate().rev() {
        strides[axis] = stride;
        let value = extent.bound();
        stride = stride
            .checked_mul(value)
            .ok_or(LogicalProgramError::ExtentOverflow(node))?;
    }
    Ok(strides)
}

/// Logical-stage validation failure.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum LogicalProgramError {
    /// Whole-program topology is invalid.
    #[error("logical graph rejected invalid topology: {0}")]
    Graph(String),
    /// A symbolic extent has no request binding.
    #[error("logical graph is missing symbolic extent `{0}`")]
    MissingSymbol(String),
    /// A request binding does not occur in the graph.
    #[error("logical graph has unexpected symbolic extent `{0}`")]
    UnexpectedSymbol(String),
    /// Product of logical extents cannot fit the bounded u64 domain.
    #[error("logical region for graph node {0:?} overflows its point bound")]
    ExtentOverflow(GraphNodeId),
    /// A zero extent has not been resolved to a schedulable positive bound.
    #[error(
        "logical region for graph node {node:?} has unresolved extent at graph value {value:?} axis {axis}"
    )]
    UnresolvedExtent {
        /// Region containing the unresolved extent.
        node: GraphNodeId,
        /// Graph value that declares the extent.
        value: GraphValueId,
        /// Axis within the graph value.
        axis: u32,
    },
    /// A node has a symbolic shape but no graph value that can define it.
    #[error("logical region for graph node {0:?} has no typed domain value")]
    MissingDomain(GraphNodeId),
    /// A graph value referenced by a logical domain does not exist.
    #[error("logical domain references missing graph value {0:?}")]
    MissingDomainValue(GraphValueId),
    /// A logical domain rank cannot be represented by the versioned wire contract.
    #[error("logical region for graph node {0:?} exceeds the u32 axis range")]
    DomainRankOverflow(GraphNodeId),
    /// A domain dependence points to the same or a later graph node.
    #[error("logical region for graph node {node:?} has cyclic dependence on {predecessor:?}")]
    CyclicDomain {
        /// Region containing the invalid dependence.
        node: GraphNodeId,
        /// Non-preceding dependency.
        predecessor: GraphNodeId,
    },
    /// Graph ports do not establish pairwise-disjoint logical values.
    #[error("logical region for graph node {0:?} has incompatible alias declarations")]
    IncompatibleAliases(GraphNodeId),
    /// A whole-program graph cannot produce schedule-free canonical bytes.
    #[error("logical graph is not canonical: {0}")]
    CanonicalGraph(String),
    /// Logical identity serialization failed.
    #[error("logical identity serialization failed: {0}")]
    Identity(String),
}
