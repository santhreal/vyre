//! Schedule-free facts about how a logical region may be cut and what it
//! exchanges.
//!
//! A partition axis states that points along one logical axis are independent,
//! combine associatively, or are ordered. An exchange states that a region
//! depends on values other participants hold. Neither names a device, a mesh
//! coordinate, or a transport: which devices exist and who carries the bytes is
//! a target fact, and choosing among them is schedule selection.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ir::{
    CollectiveOp, GraphNodeId, ProgramGraph, ProgramGraphNode, ShapeDim, ValueContract,
};
use crate::transform::collectives::{collective_exchanges, CollectiveExchangeKind};

/// What splitting one logical axis means for the values computed along it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalPartitionAxisKind {
    /// Points are independent, so any subset computes the same values.
    Elementwise,
    /// Points combine associatively, so a split needs a combining exchange.
    Reduction,
    /// Points are ordered, so a split needs ordered segments.
    Sequence,
    /// Points address a spatial domain, so a split needs boundary values.
    Spatial,
    /// Points are assigned by the data, so a split needs a routing exchange.
    Routed,
}

/// One logical axis a shard may split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct LogicalPartitionAxis {
    /// Axis within the region domain.
    pub axis: u32,
    /// What splitting this axis means.
    pub kind: LogicalPartitionAxisKind,
    /// Exact bound of the axis.
    pub bound: u64,
}

/// Closed statement of how one logical region may be distributed.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub struct LogicalPartitionFacts {
    /// Axes a shard may split, in axis order.
    pub axes: Vec<LogicalPartitionAxis>,
    /// Whether every participant may hold the whole region.
    pub replicable: bool,
}

/// Semantics of one exchange between participants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalExchangeKind {
    /// Every participant contributes and receives the combined value.
    AllReduce,
    /// Every participant receives every shard.
    AllGather,
    /// Every participant receives the combined value of its own shard.
    ReduceScatter,
    /// One participant sends a value to every other.
    Broadcast,
    /// One participant sends a value to one other.
    PointToPoint,
}

impl LogicalExchangeKind {
    /// Whether this exchange combines contributions rather than moving them.
    #[must_use]
    pub const fn combines(self) -> bool {
        match self {
            Self::AllReduce | Self::ReduceScatter => true,
            Self::AllGather | Self::Broadcast | Self::PointToPoint => false,
        }
    }
}

/// One semantic exchange a region takes part in.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogicalExchange {
    /// Region that exchanges values.
    pub node: GraphNodeId,
    /// Exchange semantics.
    pub kind: LogicalExchangeKind,
    /// Participant group the exchange is scoped to.
    pub group: u32,
    /// Combining operator, when the exchange combines.
    pub combine: Option<CollectiveOp>,
    /// Graph values the exchange moves, in operand order.
    pub values: Vec<u32>,
    /// Exact payload bytes of one participant's contribution.
    pub bytes: u64,
}

/// Derive the partition facts of one region.
///
/// A reduction region combines along its reduction axes, so those are the axes a
/// split must combine across. A region whose updates land at data-dependent
/// locations routes its points, and routing is order-free, so those axes are
/// routed rather than ordered. A sequential or retained-state region is ordered,
/// so its axes are sequence axes and only the retained region itself decides how
/// far a segment reaches. A region that reads a value it also writes may read a
/// point another shard holds, so its remaining axes address a spatial domain.
///
/// Replicability is separate from axis kind: a region that advances retained
/// state or updates at data-dependent locations is not replicable, because two
/// participants holding the whole region would each apply the update.
pub(crate) fn partition_facts(
    extents: &[u64],
    reduction_axes: &[u32],
    ordered: bool,
    retained: bool,
    atomics: bool,
    in_place_reads: bool,
) -> LogicalPartitionFacts {
    let axes = extents
        .iter()
        .enumerate()
        .filter_map(|(axis, bound)| {
            let axis = u32::try_from(axis).ok()?;
            let kind = if reduction_axes.contains(&axis) {
                LogicalPartitionAxisKind::Reduction
            } else if atomics && !retained {
                LogicalPartitionAxisKind::Routed
            } else if ordered {
                LogicalPartitionAxisKind::Sequence
            } else if in_place_reads {
                LogicalPartitionAxisKind::Spatial
            } else {
                LogicalPartitionAxisKind::Elementwise
            };
            Some(LogicalPartitionAxis {
                axis,
                kind,
                bound: *bound,
            })
        })
        .collect();
    LogicalPartitionFacts {
        axes,
        replicable: !retained && !atomics,
    }
}

/// Exact packed bytes of the graph values `values`.
///
/// A placement that moves a value between devices prices the value's own
/// contract, so the bytes are read from the graph rather than restated by
/// whoever moves them.
pub(crate) fn value_bytes(
    graph: &ProgramGraph,
    bindings: &BTreeMap<String, u64>,
    values: &[u32],
) -> Result<u64, String> {
    let mut bytes = 0u64;
    for value in values {
        let Some(entry) = graph.values().iter().find(|entry| entry.id.0 == *value) else {
            continue;
        };
        bytes = bytes.saturating_add(contract_bytes(&entry.contract, bindings)?);
    }
    Ok(bytes)
}

/// Derive every exchange the graph's programs state.
///
/// The buffer an exchange names is a program-local name, and the graph binds
/// that name to one connected value, so the payload bytes are the value's own
/// contract rather than a figure the exchange restates. An exchange over a
/// buffer the graph does not connect moves no graph value and is reported with
/// no values so nothing downstream places it.
pub(crate) fn exchanges(
    graph: &ProgramGraph,
    bindings: &BTreeMap<String, u64>,
) -> Result<Vec<LogicalExchange>, String> {
    let mut derived = Vec::new();
    for node in graph.nodes() {
        for exchange in collective_exchanges(&node.program) {
            let mut values = Vec::with_capacity(exchange.buffers.len());
            let mut bytes = 0u64;
            for buffer in &exchange.buffers {
                let Some(contract) = port_contract(node, buffer.as_str()) else {
                    continue;
                };
                values.push(contract.0);
                bytes = bytes.max(contract_bytes(contract.1, bindings)?);
            }
            derived.push(LogicalExchange {
                node: node.id,
                kind: match exchange.kind {
                    CollectiveExchangeKind::AllReduce => LogicalExchangeKind::AllReduce,
                    CollectiveExchangeKind::AllGather => LogicalExchangeKind::AllGather,
                    CollectiveExchangeKind::ReduceScatter => LogicalExchangeKind::ReduceScatter,
                    CollectiveExchangeKind::Broadcast => LogicalExchangeKind::Broadcast,
                },
                group: exchange.group,
                combine: exchange.combine,
                values,
                bytes,
            });
        }
    }
    Ok(derived)
}

/// The graph value one program-local buffer name is bound to.
fn port_contract<'a>(node: &'a ProgramGraphNode, buffer: &str) -> Option<(u32, &'a ValueContract)> {
    if let Some(input) = node.inputs.iter().find(|input| input.buffer == buffer) {
        return Some((input.value.0, &input.contract));
    }
    node.output_ports
        .iter()
        .zip(&node.outputs)
        .find(|(port, _)| port.buffer == buffer)
        .map(|(port, value)| (value.0, &port.contract))
}

/// Exact packed bytes of one value contract.
fn contract_bytes(
    contract: &ValueContract,
    bindings: &BTreeMap<String, u64>,
) -> Result<u64, String> {
    let mut count = 1u64;
    for dim in &contract.shape {
        let bound = match dim {
            ShapeDim::Known(bound) => *bound,
            ShapeDim::Symbol(symbol) => *bindings
                .get(symbol)
                .ok_or_else(|| format!("exchange payload needs symbolic extent `{symbol}`"))?,
        };
        count = count
            .checked_mul(bound)
            .ok_or_else(|| "exchange payload exceeds the u64 element bound".to_owned())?;
    }
    let count = usize::try_from(count)
        .map_err(|_| "exchange payload exceeds the addressable element bound".to_owned())?;
    let packed = contract
        .dtype
        .packed_size_bytes(count)?
        .ok_or_else(|| "exchange payload has no packed byte size".to_owned())?;
    u64::try_from(packed).map_err(|_| "exchange payload exceeds the u64 byte bound".to_owned())
}
