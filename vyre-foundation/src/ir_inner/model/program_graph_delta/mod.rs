//! Canonical bounded transactional graph-delta contract.
//!
//! Interactive applications mutate a small subset of retained state on each
//! event. Recompiling or resubmitting an entire graph for a localized change
//! (such as an updated glyph run, transform, or viewport) introduces severe
//! latency overhead.
//!
//! [`GraphDelta`] specifies a transactional sequence of atomic graph mutations
//! ([`GraphDeltaOp`]) over stable typed semantic identities. Applying a delta
//! computes the exact [`AffectedGraphClosure`], deriving dirty logical regions
//! while preserving unchanged nodes, values, compiled entries, and resource
//! residency allocations.

use std::collections::BTreeSet;

use thiserror::Error;

use super::program::Program;
use super::program_graph::{
    GraphInput, GraphNodeId, GraphOutput, GraphValueId, ProgramGraph, ProgramGraphError, ShapeDim,
    ValueContract, ValueLifetime,
};

mod generation;
mod wire;

pub use generation::GenerationTracker;
use wire::*;

/// Format version for persisted [`GraphDelta`] structures.
pub const GRAPH_DELTA_VERSION: u16 = 1;

const MAGIC: &[u8; 4] = b"VGD0";
const MAX_DELTA_WIRE_BYTES: usize = 256 * 1024 * 1024;
const MAX_DELTA_OPERATIONS: usize = 1_000_000;
const MAX_PORTS_PER_NODE: usize = 1_000_000;
const MAX_NAME_BYTES: usize = 4_096;
const MAX_RANK: usize = 256;
/// Atomic mutation operations in a [`GraphDelta`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphDeltaOp {
    /// Register a new external input, constant, or initial retained value.
    InsertExternalValue {
        /// Stable value name.
        name: String,
        /// Semantic value contract.
        contract: ValueContract,
    },
    /// Append a new program node.
    InsertNode {
        /// Stable node name.
        name: String,
        /// Executable program IR.
        program: Program,
        /// Connected input bindings.
        inputs: Vec<GraphInput>,
        /// Produced output ports.
        outputs: Vec<GraphOutput>,
    },
    /// Replace an existing node's program and port bindings in-place.
    ReplaceNode {
        /// Canonical node to replace.
        node_id: GraphNodeId,
        /// Replacement executable program IR.
        program: Program,
        /// Replacement input bindings.
        inputs: Vec<GraphInput>,
        /// Replacement output ports.
        outputs: Vec<GraphOutput>,
    },
    /// Delete a node and its exclusively produced values.
    DeleteNode {
        /// Canonical node to delete.
        node_id: GraphNodeId,
    },
    /// Update a symbolic dimension bound across all consuming contracts.
    UpdateShapeBound {
        /// Symbolic dimension name (e.g. "batch", "seq_len", "width").
        symbol: String,
        /// Verified previous bound.
        old_bound: u64,
        /// New concrete bound.
        new_bound: u64,
    },
    /// Advance a retained resource generation.
    UpdateResourceGeneration {
        /// Stable resource name.
        resource_name: String,
        /// Previous generation counter.
        prior_generation: u64,
        /// New generation counter.
        new_generation: u64,
    },
    /// Rebind a retained state transition.
    UpdateStateTransition {
        /// Name of the state output.
        output_name: String,
        /// Previous predecessor value ID.
        prior_value: GraphValueId,
        /// New predecessor value ID.
        new_prior_value: GraphValueId,
    },
}

/// Exact affected semantic, proof, and residency closure derived from a delta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffectedGraphClosure {
    /// Nodes whose program, bindings, or inputs changed directly or transitively.
    pub dirty_nodes: BTreeSet<GraphNodeId>,
    /// Values produced or invalidated by dirty nodes or updated bounds.
    pub dirty_values: BTreeSet<GraphValueId>,
    /// Preserved nodes whose compiled entries and schedules remain valid.
    pub unchanged_nodes: BTreeSet<GraphNodeId>,
    /// Preserved values whose allocations and resident buffers remain valid.
    pub unchanged_values: BTreeSet<GraphValueId>,
    /// Resource names whose generations or bindings were modified.
    pub affected_resource_names: BTreeSet<String>,
    /// Whether the delta was a pure shape bound update requiring no topological re-lowering.
    pub is_pure_shape_update: bool,
    /// Whether the delta was a pure resource generation advance.
    pub is_pure_generation_bump: bool,
}

impl AffectedGraphClosure {
    /// Derive exact invalidated query keys for the deterministic query engine.
    #[must_use]
    pub fn invalidated_query_keys(&self) -> Vec<crate::substrate::QueryKey> {
        let mut keys = Vec::new();
        for node_id in &self.dirty_nodes {
            keys.push(crate::substrate::QueryKey::SemanticFacts {
                node_id: node_id.0,
                program_digest: [0u8; 32],
            });
            keys.push(crate::substrate::QueryKey::Lowering {
                node_id: node_id.0,
                target_fingerprint: 0,
            });
            keys.push(crate::substrate::QueryKey::Emission {
                node_id: node_id.0,
                // The format identity comes from the target materializer, so a graph delta
                // leaves it unset like the zero digest and fingerprint beside it.
                target_format: String::new(),
                target_fingerprint: 0,
            });
        }
        keys
    }
}

/// Transactional error encountered while validating or applying a [`GraphDelta`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GraphDeltaError {
    /// Underlying structural graph error.
    #[error("graph delta validation error: {0}")]
    Graph(#[from] ProgramGraphError),
    /// Referenced node does not exist in the base graph.
    #[error("graph delta references missing node {0:?}")]
    MissingNode(GraphNodeId),
    /// Referenced value does not exist in the base graph.
    #[error("graph delta references missing value {0:?}")]
    MissingValue(GraphValueId),
    /// Deleting a node would leave active dependents.
    #[error("cannot delete node {node:?}: node {dependent:?} still consumes its output")]
    DependencyViolation {
        /// Target node to delete.
        node: GraphNodeId,
        /// Dependent consumer node.
        dependent: GraphNodeId,
    },
    /// Shape bound update named a symbol no value in the graph declares.
    ///
    /// A symbol's concrete extent is supplied by the caller's binding map at
    /// allocation time and is not stored in the graph, so a delta can be
    /// checked against the symbols the graph declares and not against their
    /// prior values.
    #[error("shape symbol `{symbol}` is not declared by any value in the graph")]
    UnknownShapeSymbol {
        /// Symbol name the delta named.
        symbol: String,
    },
    /// Shape bound update states a new bound that is not a change, or is zero.
    #[error("shape symbol `{symbol}` bound update from {old_bound} to {new_bound} is not a legal change")]
    IllegalShapeBound {
        /// Symbol name.
        symbol: String,
        /// Bound the delta states the symbol had.
        old_bound: u64,
        /// Bound the delta states the symbol takes.
        new_bound: u64,
    },
    /// Generation tracker lock was poisoned by a previous thread panic.
    #[error(
        "generation tracker lock was poisoned for `{state}`. Fix: rebuild the generation tracker"
    )]
    LockPoisoned {
        /// Guarded state name.
        state: String,
    },
    /// Resource generation specified an incorrect prior generation.
    #[error("resource `{resource_name}` expected prior generation {expected}, found {found}")]
    ResourceGenerationMismatch {
        /// Resource name.
        resource_name: String,
        /// Expected prior generation.
        expected: u64,
        /// Actual generation.
        found: u64,
    },
    /// State transition update is invalid.
    #[error("invalid state transition for output `{output_name}`: {reason}")]
    InvalidStateTransition {
        /// Output value name.
        output_name: String,
        /// Rejection reason.
        reason: String,
    },
    /// Format version mismatch during deserialization.
    #[error("graph delta format version mismatch: expected {expected}, found {found}")]
    VersionMismatch {
        /// Supported format version.
        expected: u16,
        /// Decoded format version.
        found: u16,
    },
    /// Maximum operation count exceeded in delta container.
    #[error("graph delta exceeds operation ceiling limit of {limit}")]
    OperationLimitExceeded {
        /// Configured operation ceiling limit.
        limit: usize,
    },
    /// Name string byte length exceeded.
    #[error("graph delta name `{name}` exceeds length ceiling of {limit} bytes")]
    NameLengthExceeded {
        /// Oversized name string.
        name: String,
        /// Maximum allowed bytes.
        limit: usize,
    },
    /// Tensor rank dimensionality exceeded.
    #[error("graph delta rank {rank} exceeds ceiling of {limit}")]
    RankExceeded {
        /// Declared rank.
        rank: usize,
        /// Maximum rank limit.
        limit: usize,
    },
    /// Attempted to publish an artifact from a superseded generation.
    #[error("superseded generation for resource `{resource_name}`: current is {current_generation}, attempted {attempted_generation}")]
    SupersededGeneration {
        /// Resource name.
        resource_name: String,
        /// Current active generation.
        current_generation: u64,
        /// Stale attempted generation.
        attempted_generation: u64,
    },
    /// Wire encoding or decoding failure.
    #[error("invalid graph delta wire data: {0}")]
    Wire(String),
    /// Delta is empty and specifies no mutations.
    #[error("graph delta contains no operations")]
    EmptyDelta,
}

/// Canonical bounded transactional graph-delta container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphDelta {
    version: u16,
    operations: Vec<GraphDeltaOp>,
}

impl Default for GraphDelta {
    fn default() -> Self {
        Self {
            version: GRAPH_DELTA_VERSION,
            operations: Vec::new(),
        }
    }
}

impl GraphDelta {
    /// Create an empty [`GraphDelta`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check bounds on an individual operation.
    pub fn validate_op_bounds(op: &GraphDeltaOp) -> Result<(), GraphDeltaError> {
        match op {
            GraphDeltaOp::InsertExternalValue { name, contract } => {
                if name.len() > MAX_NAME_BYTES {
                    return Err(GraphDeltaError::NameLengthExceeded {
                        name: name.clone(),
                        limit: MAX_NAME_BYTES,
                    });
                }
                if contract.shape.len() > MAX_RANK {
                    return Err(GraphDeltaError::RankExceeded {
                        rank: contract.shape.len(),
                        limit: MAX_RANK,
                    });
                }
            }
            GraphDeltaOp::InsertNode {
                name,
                inputs,
                outputs,
                ..
            } => {
                if name.len() > MAX_NAME_BYTES {
                    return Err(GraphDeltaError::NameLengthExceeded {
                        name: name.clone(),
                        limit: MAX_NAME_BYTES,
                    });
                }
                if inputs.len() > MAX_PORTS_PER_NODE || outputs.len() > MAX_PORTS_PER_NODE {
                    return Err(GraphDeltaError::OperationLimitExceeded {
                        limit: MAX_PORTS_PER_NODE,
                    });
                }
            }
            GraphDeltaOp::ReplaceNode {
                inputs, outputs, ..
            } => {
                if inputs.len() > MAX_PORTS_PER_NODE || outputs.len() > MAX_PORTS_PER_NODE {
                    return Err(GraphDeltaError::OperationLimitExceeded {
                        limit: MAX_PORTS_PER_NODE,
                    });
                }
            }
            GraphDeltaOp::UpdateShapeBound { symbol, .. } => {
                if symbol.len() > MAX_NAME_BYTES {
                    return Err(GraphDeltaError::NameLengthExceeded {
                        name: symbol.clone(),
                        limit: MAX_NAME_BYTES,
                    });
                }
            }
            GraphDeltaOp::UpdateResourceGeneration { resource_name, .. } => {
                if resource_name.len() > MAX_NAME_BYTES {
                    return Err(GraphDeltaError::NameLengthExceeded {
                        name: resource_name.clone(),
                        limit: MAX_NAME_BYTES,
                    });
                }
            }
            GraphDeltaOp::UpdateStateTransition { output_name, .. } => {
                if output_name.len() > MAX_NAME_BYTES {
                    return Err(GraphDeltaError::NameLengthExceeded {
                        name: output_name.clone(),
                        limit: MAX_NAME_BYTES,
                    });
                }
            }
            GraphDeltaOp::DeleteNode { .. } => {}
        }
        Ok(())
    }

    /// Try pushing an operation onto the delta, returning an error if bounds are exceeded.
    pub fn try_push(&mut self, op: GraphDeltaOp) -> Result<(), GraphDeltaError> {
        if self.operations.len() >= MAX_DELTA_OPERATIONS {
            return Err(GraphDeltaError::OperationLimitExceeded {
                limit: MAX_DELTA_OPERATIONS,
            });
        }
        Self::validate_op_bounds(&op)?;
        self.operations.push(op);
        Ok(())
    }
    /// Number of operations in this delta.
    #[must_use]
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Whether this delta has no operations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Format version of this delta.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Operations declared in this delta.
    #[must_use]
    pub fn operations(&self) -> &[GraphDeltaOp] {
        &self.operations
    }

    /// Validate that this delta can be legally applied to `graph` without mutating it.
    pub fn validate(&self, graph: &ProgramGraph) -> Result<(), GraphDeltaError> {
        let _ = self.apply_transactional(graph)?;
        Ok(())
    }

    /// Transactionally apply this delta to `graph`, returning the mutated graph and the derived closure.
    ///
    /// If any operation fails validation or introduces a structural defect, the transaction
    /// aborts and returns an error without mutating `graph`.
    pub fn apply_transactional(
        &self,
        graph: &ProgramGraph,
    ) -> Result<(ProgramGraph, AffectedGraphClosure), GraphDeltaError> {
        if self.operations.is_empty() {
            return Err(GraphDeltaError::EmptyDelta);
        }

        let mut mutated = graph.clone();
        let mut dirty_nodes = BTreeSet::new();
        let mut dirty_values = BTreeSet::new();
        let mut affected_resource_names = BTreeSet::new();
        let mut is_pure_shape_update = true;
        let mut is_pure_generation_bump = true;

        for op in &self.operations {
            match op {
                GraphDeltaOp::InsertExternalValue { name, contract } => {
                    is_pure_shape_update = false;
                    is_pure_generation_bump = false;
                    let val_id = mutated.add_external_value(name, contract.clone())?;
                    dirty_values.insert(val_id);
                    affected_resource_names.insert(name.clone());
                }
                GraphDeltaOp::InsertNode {
                    name,
                    program,
                    inputs,
                    outputs,
                } => {
                    is_pure_shape_update = false;
                    is_pure_generation_bump = false;
                    let (node_id, val_ids) =
                        mutated.add_node(name, program.clone(), inputs.clone(), outputs.clone())?;
                    dirty_nodes.insert(node_id);
                    for vid in val_ids {
                        dirty_values.insert(vid);
                    }
                }
                GraphDeltaOp::ReplaceNode {
                    node_id,
                    program,
                    inputs,
                    outputs,
                } => {
                    is_pure_shape_update = false;
                    is_pure_generation_bump = false;
                    dirty_nodes.insert(*node_id);
                    for output in outputs {
                        affected_resource_names.insert(output.name.clone());
                    }
                    for value in mutated
                        .nodes()
                        .get(node_id.0 as usize)
                        .ok_or(GraphDeltaError::MissingNode(*node_id))?
                        .outputs
                        .clone()
                    {
                        dirty_values.insert(value);
                    }
                    // The graph owns port validation and consumer rewiring. A
                    // bounds check followed by a dirty mark left the node's
                    // program and inputs exactly as they were, so applying a
                    // replacement produced a graph identical to the one it was
                    // applied to.
                    mutated
                        .replace_node(*node_id, program.clone(), inputs.clone(), outputs.clone())
                        .map_err(|error| GraphDeltaError::Wire(error.to_string()))?;
                }
                GraphDeltaOp::DeleteNode { node_id } => {
                    is_pure_shape_update = false;
                    is_pure_generation_bump = false;
                    if (node_id.0 as usize) >= mutated.nodes().len() {
                        return Err(GraphDeltaError::MissingNode(*node_id));
                    }
                    let node = &mutated.nodes()[node_id.0 as usize];
                    for out_val_id in &node.outputs {
                        let val = &mutated.values()[out_val_id.0 as usize];
                        for consumer in &val.consumers {
                            if consumer != node_id && !dirty_nodes.contains(consumer) {
                                return Err(GraphDeltaError::DependencyViolation {
                                    node: *node_id,
                                    dependent: *consumer,
                                });
                            }
                        }
                        dirty_values.insert(*out_val_id);
                    }
                    dirty_nodes.insert(*node_id);
                }
                GraphDeltaOp::UpdateShapeBound {
                    symbol,
                    old_bound,
                    new_bound,
                } => {
                    is_pure_generation_bump = false;
                    // A symbol's extent lives in the caller's binding map, so
                    // the prior bound cannot be read back from the graph. What
                    // the op states about itself is checkable: a zero extent is
                    // not a legal dimension, and a bound that does not move is
                    // a delta claiming a shape change it does not make.
                    if *new_bound == 0 || new_bound == old_bound {
                        return Err(GraphDeltaError::IllegalShapeBound {
                            symbol: symbol.clone(),
                            old_bound: *old_bound,
                            new_bound: *new_bound,
                        });
                    }
                    let mut found_symbol = false;
                    for value in mutated.values() {
                        for dim in &value.contract.shape {
                            if let ShapeDim::Symbol(sym) = dim {
                                if sym == symbol {
                                    found_symbol = true;
                                    dirty_values.insert(value.id);
                                    if let Some(producer) = value.producer {
                                        dirty_nodes.insert(producer);
                                    }
                                    for consumer in &value.consumers {
                                        dirty_nodes.insert(*consumer);
                                    }
                                }
                            }
                        }
                    }
                    if !found_symbol {
                        return Err(GraphDeltaError::UnknownShapeSymbol {
                            symbol: symbol.clone(),
                        });
                    }
                    affected_resource_names.insert(symbol.clone());
                }
                GraphDeltaOp::UpdateResourceGeneration {
                    resource_name,
                    prior_generation,
                    new_generation,
                } => {
                    is_pure_shape_update = false;
                    if new_generation <= prior_generation {
                        return Err(GraphDeltaError::ResourceGenerationMismatch {
                            resource_name: resource_name.clone(),
                            expected: prior_generation + 1,
                            found: *new_generation,
                        });
                    }
                    affected_resource_names.insert(resource_name.clone());
                }
                GraphDeltaOp::UpdateStateTransition {
                    output_name,
                    prior_value,
                    new_prior_value,
                } => {
                    is_pure_shape_update = false;
                    is_pure_generation_bump = false;
                    if (prior_value.0 as usize) >= mutated.values().len() {
                        return Err(GraphDeltaError::MissingValue(*prior_value));
                    }
                    if (new_prior_value.0 as usize) >= mutated.values().len() {
                        return Err(GraphDeltaError::MissingValue(*new_prior_value));
                    }
                    let new_prior = &mutated.values()[new_prior_value.0 as usize];
                    if new_prior.contract.lifetime != ValueLifetime::Retained {
                        return Err(GraphDeltaError::InvalidStateTransition {
                            output_name: output_name.clone(),
                            reason: format!(
                                "new prior value {:?} is not a Retained value",
                                new_prior_value
                            ),
                        });
                    }
                    dirty_values.insert(*new_prior_value);
                    affected_resource_names.insert(output_name.clone());
                }
            }
        }

        // Transitive dependency closure: propagate dirty status to all downstream consumers
        let mut worklist: Vec<GraphNodeId> = dirty_nodes.iter().copied().collect();
        while let Some(node_id) = worklist.pop() {
            if (node_id.0 as usize) < mutated.nodes().len() {
                let node = &mutated.nodes()[node_id.0 as usize];
                for output_val in &node.outputs {
                    dirty_values.insert(*output_val);
                    if (output_val.0 as usize) < mutated.values().len() {
                        let val = &mutated.values()[output_val.0 as usize];
                        for consumer in &val.consumers {
                            if dirty_nodes.insert(*consumer) {
                                worklist.push(*consumer);
                            }
                        }
                    }
                }
            }
        }

        let mut unchanged_nodes = BTreeSet::new();
        for node in mutated.nodes() {
            if !dirty_nodes.contains(&node.id) {
                unchanged_nodes.insert(node.id);
            }
        }

        let mut unchanged_values = BTreeSet::new();
        for val in mutated.values() {
            if !dirty_values.contains(&val.id) {
                unchanged_values.insert(val.id);
            }
        }

        Ok((
            mutated,
            AffectedGraphClosure {
                dirty_nodes,
                dirty_values,
                unchanged_nodes,
                unchanged_values,
                affected_resource_names,
                is_pure_shape_update,
                is_pure_generation_bump,
            },
        ))
    }

    /// Encode delta into canonical wire bytes.
    pub fn to_wire(&self) -> Result<Vec<u8>, GraphDeltaError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.version.to_le_bytes());
        let count = u32::try_from(self.operations.len())
            .map_err(|_| GraphDeltaError::Wire("too many operations".into()))?;
        bytes.extend_from_slice(&count.to_le_bytes());

        for op in &self.operations {
            match op {
                GraphDeltaOp::InsertExternalValue { name, contract } => {
                    bytes.push(1);
                    put_string(&mut bytes, name)?;
                    put_contract(&mut bytes, contract)?;
                }
                GraphDeltaOp::InsertNode {
                    name,
                    program,
                    inputs,
                    outputs,
                } => {
                    bytes.push(2);
                    put_string(&mut bytes, name)?;
                    let prog_bytes = program
                        .to_wire()
                        .map_err(|e| GraphDeltaError::Wire(format!("program wire: {e}")))?;
                    put_bytes(&mut bytes, &prog_bytes)?;
                    let in_count = u32::try_from(inputs.len())
                        .map_err(|_| GraphDeltaError::Wire("too many inputs".into()))?;
                    bytes.extend_from_slice(&in_count.to_le_bytes());
                    for input in inputs {
                        put_string(&mut bytes, &input.buffer)?;
                        bytes.extend_from_slice(&input.value.0.to_le_bytes());
                        put_contract(&mut bytes, &input.contract)?;
                    }
                    let out_count = u32::try_from(outputs.len())
                        .map_err(|_| GraphDeltaError::Wire("too many outputs".into()))?;
                    bytes.extend_from_slice(&out_count.to_le_bytes());
                    for output in outputs {
                        put_string(&mut bytes, &output.buffer)?;
                        put_string(&mut bytes, &output.name)?;
                        put_contract(&mut bytes, &output.contract)?;
                        match output.retained_successor_of {
                            Some(prior) => {
                                bytes.push(1);
                                bytes.extend_from_slice(&prior.0.to_le_bytes());
                            }
                            None => bytes.push(0),
                        }
                    }
                }
                GraphDeltaOp::ReplaceNode {
                    node_id,
                    program,
                    inputs,
                    outputs,
                } => {
                    bytes.push(3);
                    bytes.extend_from_slice(&node_id.0.to_le_bytes());
                    let prog_bytes = program
                        .to_wire()
                        .map_err(|e| GraphDeltaError::Wire(format!("program wire: {e}")))?;
                    put_bytes(&mut bytes, &prog_bytes)?;
                    let in_count = u32::try_from(inputs.len())
                        .map_err(|_| GraphDeltaError::Wire("too many inputs".into()))?;
                    bytes.extend_from_slice(&in_count.to_le_bytes());
                    for input in inputs {
                        put_string(&mut bytes, &input.buffer)?;
                        bytes.extend_from_slice(&input.value.0.to_le_bytes());
                        put_contract(&mut bytes, &input.contract)?;
                    }
                    let out_count = u32::try_from(outputs.len())
                        .map_err(|_| GraphDeltaError::Wire("too many outputs".into()))?;
                    bytes.extend_from_slice(&out_count.to_le_bytes());
                    for output in outputs {
                        put_string(&mut bytes, &output.buffer)?;
                        put_string(&mut bytes, &output.name)?;
                        put_contract(&mut bytes, &output.contract)?;
                        match output.retained_successor_of {
                            Some(prior) => {
                                bytes.push(1);
                                bytes.extend_from_slice(&prior.0.to_le_bytes());
                            }
                            None => bytes.push(0),
                        }
                    }
                }
                GraphDeltaOp::DeleteNode { node_id } => {
                    bytes.push(4);
                    bytes.extend_from_slice(&node_id.0.to_le_bytes());
                }
                GraphDeltaOp::UpdateShapeBound {
                    symbol,
                    old_bound,
                    new_bound,
                } => {
                    bytes.push(5);
                    put_string(&mut bytes, symbol)?;
                    bytes.extend_from_slice(&old_bound.to_le_bytes());
                    bytes.extend_from_slice(&new_bound.to_le_bytes());
                }
                GraphDeltaOp::UpdateResourceGeneration {
                    resource_name,
                    prior_generation,
                    new_generation,
                } => {
                    bytes.push(6);
                    put_string(&mut bytes, resource_name)?;
                    bytes.extend_from_slice(&prior_generation.to_le_bytes());
                    bytes.extend_from_slice(&new_generation.to_le_bytes());
                }
                GraphDeltaOp::UpdateStateTransition {
                    output_name,
                    prior_value,
                    new_prior_value,
                } => {
                    bytes.push(7);
                    put_string(&mut bytes, output_name)?;
                    bytes.extend_from_slice(&prior_value.0.to_le_bytes());
                    bytes.extend_from_slice(&new_prior_value.0.to_le_bytes());
                }
            }
        }
        Ok(bytes)
    }

    /// Decode delta from canonical wire bytes.
    pub fn from_wire(bytes: &[u8]) -> Result<Self, GraphDeltaError> {
        if bytes.len() > MAX_DELTA_WIRE_BYTES {
            return Err(GraphDeltaError::Wire(format!(
                "delta wire input is {} bytes; maximum is {MAX_DELTA_WIRE_BYTES}",
                bytes.len()
            )));
        }
        if bytes.len() < 10 {
            return Err(GraphDeltaError::Wire("wire payload too short".into()));
        }
        if &bytes[0..4] != MAGIC {
            return Err(GraphDeltaError::Wire("invalid magic header".into()));
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != GRAPH_DELTA_VERSION {
            return Err(GraphDeltaError::VersionMismatch {
                expected: GRAPH_DELTA_VERSION,
                found: version,
            });
        }
        let count = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
        if count > MAX_DELTA_OPERATIONS {
            return Err(GraphDeltaError::Wire(format!(
                "operation count is {count}; maximum is {MAX_DELTA_OPERATIONS}"
            )));
        }
        let mut cursor = 10;
        let mut operations = Vec::with_capacity(count.min(bytes.len() - cursor));
        for _ in 0..count {
            if cursor >= bytes.len() {
                return Err(GraphDeltaError::Wire("unexpected EOF in operations".into()));
            }
            let tag = bytes[cursor];
            cursor += 1;
            match tag {
                1 => {
                    let name = read_string(bytes, &mut cursor)?;
                    let contract = read_contract(bytes, &mut cursor)?;
                    operations.push(GraphDeltaOp::InsertExternalValue { name, contract });
                }
                2 => {
                    let name = read_string(bytes, &mut cursor)?;
                    let prog_bytes = read_bytes(bytes, &mut cursor)?;
                    let program = Program::from_wire(&prog_bytes)
                        .map_err(|e| GraphDeltaError::Wire(format!("program decode: {e}")))?;
                    let in_count = read_u32(bytes, &mut cursor)? as usize;
                    if in_count > MAX_PORTS_PER_NODE {
                        return Err(GraphDeltaError::Wire(format!(
                            "input port count is {in_count}; maximum is {MAX_PORTS_PER_NODE}"
                        )));
                    }
                    let mut inputs = Vec::with_capacity(in_count.min(bytes.len() - cursor));
                    for _ in 0..in_count {
                        let buffer = read_string(bytes, &mut cursor)?;
                        let value = GraphValueId(read_u32(bytes, &mut cursor)?);
                        let contract = read_contract(bytes, &mut cursor)?;
                        inputs.push(GraphInput {
                            buffer,
                            value,
                            contract,
                        });
                    }
                    let out_count = read_u32(bytes, &mut cursor)? as usize;
                    if out_count > MAX_PORTS_PER_NODE {
                        return Err(GraphDeltaError::Wire(format!(
                            "output port count is {out_count}; maximum is {MAX_PORTS_PER_NODE}"
                        )));
                    }
                    let mut outputs = Vec::with_capacity(out_count.min(bytes.len() - cursor));
                    for _ in 0..out_count {
                        let buffer = read_string(bytes, &mut cursor)?;
                        let out_name = read_string(bytes, &mut cursor)?;
                        let contract = read_contract(bytes, &mut cursor)?;
                        if cursor >= bytes.len() {
                            return Err(GraphDeltaError::Wire("EOF reading retained tag".into()));
                        }
                        let has_retained = bytes[cursor] != 0;
                        cursor += 1;
                        let retained_successor_of = if has_retained {
                            Some(GraphValueId(read_u32(bytes, &mut cursor)?))
                        } else {
                            None
                        };
                        outputs.push(GraphOutput {
                            buffer,
                            name: out_name,
                            contract,
                            retained_successor_of,
                        });
                    }
                    operations.push(GraphDeltaOp::InsertNode {
                        name,
                        program,
                        inputs,
                        outputs,
                    });
                }
                3 => {
                    let node_id = GraphNodeId(read_u32(bytes, &mut cursor)?);
                    let prog_bytes = read_bytes(bytes, &mut cursor)?;
                    let program = Program::from_wire(&prog_bytes)
                        .map_err(|e| GraphDeltaError::Wire(format!("program decode: {e}")))?;
                    let in_count = read_u32(bytes, &mut cursor)? as usize;
                    if in_count > MAX_PORTS_PER_NODE {
                        return Err(GraphDeltaError::Wire(format!(
                            "input port count is {in_count}; maximum is {MAX_PORTS_PER_NODE}"
                        )));
                    }
                    let mut inputs = Vec::with_capacity(in_count.min(bytes.len() - cursor));
                    for _ in 0..in_count {
                        let buffer = read_string(bytes, &mut cursor)?;
                        let value = GraphValueId(read_u32(bytes, &mut cursor)?);
                        let contract = read_contract(bytes, &mut cursor)?;
                        inputs.push(GraphInput {
                            buffer,
                            value,
                            contract,
                        });
                    }
                    let out_count = read_u32(bytes, &mut cursor)? as usize;
                    if out_count > MAX_PORTS_PER_NODE {
                        return Err(GraphDeltaError::Wire(format!(
                            "output port count is {out_count}; maximum is {MAX_PORTS_PER_NODE}"
                        )));
                    }
                    let mut outputs = Vec::with_capacity(out_count.min(bytes.len() - cursor));
                    for _ in 0..out_count {
                        let buffer = read_string(bytes, &mut cursor)?;
                        let out_name = read_string(bytes, &mut cursor)?;
                        let contract = read_contract(bytes, &mut cursor)?;
                        if cursor >= bytes.len() {
                            return Err(GraphDeltaError::Wire("EOF reading retained tag".into()));
                        }
                        let has_retained = bytes[cursor] != 0;
                        cursor += 1;
                        let retained_successor_of = if has_retained {
                            Some(GraphValueId(read_u32(bytes, &mut cursor)?))
                        } else {
                            None
                        };
                        outputs.push(GraphOutput {
                            buffer,
                            name: out_name,
                            contract,
                            retained_successor_of,
                        });
                    }
                    operations.push(GraphDeltaOp::ReplaceNode {
                        node_id,
                        program,
                        inputs,
                        outputs,
                    });
                }
                4 => {
                    let node_id = GraphNodeId(read_u32(bytes, &mut cursor)?);
                    operations.push(GraphDeltaOp::DeleteNode { node_id });
                }
                5 => {
                    let symbol = read_string(bytes, &mut cursor)?;
                    let old_bound = read_u64(bytes, &mut cursor)?;
                    let new_bound = read_u64(bytes, &mut cursor)?;
                    operations.push(GraphDeltaOp::UpdateShapeBound {
                        symbol,
                        old_bound,
                        new_bound,
                    });
                }
                6 => {
                    let resource_name = read_string(bytes, &mut cursor)?;
                    let prior_generation = read_u64(bytes, &mut cursor)?;
                    let new_generation = read_u64(bytes, &mut cursor)?;
                    operations.push(GraphDeltaOp::UpdateResourceGeneration {
                        resource_name,
                        prior_generation,
                        new_generation,
                    });
                }
                7 => {
                    let output_name = read_string(bytes, &mut cursor)?;
                    let prior_value = GraphValueId(read_u32(bytes, &mut cursor)?);
                    let new_prior_value = GraphValueId(read_u32(bytes, &mut cursor)?);
                    operations.push(GraphDeltaOp::UpdateStateTransition {
                        output_name,
                        prior_value,
                        new_prior_value,
                    });
                }
                other => {
                    return Err(GraphDeltaError::Wire(format!(
                        "unknown operation tag `{other}`"
                    )));
                }
            }
        }

        Ok(Self {
            version,
            operations,
        })
    }
}

