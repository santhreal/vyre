//! Typed identities, lifetime classes, value contracts, and templates for [`ProgramGraph`](super::ProgramGraph).

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::graph::ProgramGraph;
use crate::ir_inner::model::op_signature::{BufferAccess, DataType};
use crate::ir_inner::model::program::Program;

/// Canonical graph-local identity for one connected semantic value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GraphValueId(pub u32);

/// Canonical graph-local identity for one executable program node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GraphNodeId(pub u32);

/// One value dimension, either statically known or bound by graph configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ShapeDim {
    /// Exact element extent.
    Known(u64),
    /// Configuration symbol such as `batch`, `sequence`, or `hidden`.
    Symbol(String),
}

/// Semantic lifetime class used by compilation and runtime binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ValueLifetime {
    /// Immutable constant data shared by every invocation.
    Constant,
    /// Temporary data valid for one invocation.
    Invocation,
    /// Mutable data retained across submissions.
    Retained,
    /// Caller-visible graph result.
    Output,
    /// Streaming data transferred through channels or queues across stages.
    Stream,
}

/// Domain-neutral external effect class attached to graph nodes and boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ExternalEffect {
    /// Host or device I/O transfer (e.g. socket, disk, or host-device bridge).
    Io,
    /// Asynchronous device event or synchronization signal.
    DeviceEvent,
    /// Memory ordering or storage barrier across graph execution phases.
    StorageBarrier,
    /// Host trace or observability telemetry marker.
    TraceMarker,
    /// Custom domain-neutral external effect with an authenticated token tag.
    Custom(String),
}

/// Bounded iteration and control-flow constraints.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ControlBounds {
    /// Maximum iteration step bound (must be nonzero).
    pub max_steps: u64,
    /// Whether the loop is statically guaranteed to terminate within `max_steps`.
    pub guaranteed_termination: bool,
}

/// Complete semantic contract for a connected graph value.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValueContract {
    /// Element representation.
    pub dtype: DataType,
    /// Ordered value dimensions.
    pub shape: Vec<ShapeDim>,
    /// Access required from the bound Program buffer.
    pub access: BufferAccess,
    /// Semantic lifetime.
    pub lifetime: ValueLifetime,
}

impl ValueContract {
    /// A contract over one statically known dimension of `count` elements.
    ///
    /// The shape every connected value that is a flat buffer has, which is
    /// otherwise spelled as a two-line struct literal at each graph port.
    #[must_use]
    pub fn dense_1d(
        dtype: DataType,
        count: u64,
        access: BufferAccess,
        lifetime: ValueLifetime,
    ) -> Self {
        Self {
            dtype,
            shape: vec![ShapeDim::Known(count)],
            access,
            lifetime,
        }
    }
}

/// Bind one existing graph value to a named Program buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphInput {
    /// Program-local buffer name.
    pub buffer: String,
    /// Connected graph value.
    pub value: GraphValueId,
    /// Contract expected by this consumer port.
    pub contract: ValueContract,
}

/// Declare one Program output and its graph-level contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphOutput {
    /// Program-local buffer name.
    pub buffer: String,
    /// Stable graph value name.
    pub name: String,
    /// Connected value contract.
    pub contract: ValueContract,
    /// Prior retained value replaced by this output.
    pub retained_successor_of: Option<GraphValueId>,
}

/// One executable Program and its typed graph connections.
#[derive(Debug, Clone)]
pub struct ProgramGraphNode {
    /// Canonical node identity.
    pub id: GraphNodeId,
    /// Stable semantic node name used only for display and diagnostics.
    pub name: String,
    /// Existing executable Vyre IR.
    pub program: Program,
    /// Connected input ports.
    pub inputs: Vec<GraphInput>,
    /// Produced graph values, in declaration order.
    pub outputs: Vec<GraphValueId>,
    /// Program-local output bindings in declaration order.
    pub output_ports: Vec<GraphOutput>,
}

/// One connected value and its producer/consumer ownership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramGraphValue {
    /// Canonical value identity.
    pub id: GraphValueId,
    /// Stable semantic value name used only for display and diagnostics.
    pub name: String,
    /// Type, shape, access, and lifetime contract.
    pub contract: ValueContract,
    /// Producing node, or `None` for graph inputs and constants.
    pub producer: Option<GraphNodeId>,
    /// Nodes that consume this value.
    pub consumers: Vec<GraphNodeId>,
    /// Prior retained value when this value replaces retained state.
    pub retained_successor_of: Option<GraphValueId>,
}

/// Inclusive node-index interval during which one value must remain live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LivenessInterval {
    /// Connected value.
    pub value: GraphValueId,
    /// First schedule index that needs the allocation.
    pub start: usize,
    /// Last schedule index that needs the allocation.
    pub end: usize,
}

/// Structural ProgramGraph construction failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProgramGraphError {
    /// A stable node or value name was reused.
    #[error("duplicate graph name `{0}`; use one stable identity per node or value")]
    DuplicateName(String),
    /// A port references a value that does not exist.
    #[error("graph value {0:?} does not exist")]
    MissingValue(GraphValueId),
    /// A replacement or lookup references a node that does not exist.
    #[error("graph node {0:?} does not exist")]
    MissingNode(GraphNodeId),
    /// A replacement stated output ports other than the node's current ones.
    #[error(
        "graph node {node:?} cannot be replaced with different output ports; its outputs are graph values other nodes consume by id, so changing them is a delete and an insert"
    )]
    InvalidReplacementOutputs {
        /// Node whose replacement was refused.
        node: GraphNodeId,
    },
    /// A Program does not declare the named port buffer.
    #[error("program node `{node}` has no buffer `{buffer}`")]
    MissingBuffer {
        /// Stable graph node name.
        node: String,
        /// Missing Program-local buffer name.
        buffer: String,
    },
    /// Program buffer element/access metadata disagrees with the graph value.
    #[error("program node `{node}` buffer `{buffer}` disagrees with its value contract: {reason}")]
    BufferContract {
        /// Stable graph node name.
        node: String,
        /// Program-local buffer name.
        buffer: String,
        /// Exact metadata disagreement.
        reason: String,
    },
    /// Consumer-declared type or shape differs from the connected value.
    #[error(
        "program node `{node}` buffer `{buffer}` expects {expected:?}, but graph value {value:?} provides {actual:?}"
    )]
    InputContract {
        /// Stable graph node name.
        node: String,
        /// Program-local buffer name.
        buffer: String,
        /// Connected value identity.
        value: GraphValueId,
        /// Producer or external-value contract.
        actual: ValueContract,
        /// Consumer-declared contract.
        expected: ValueContract,
    },
    /// A retained-value transition is not type preserving.
    #[error("retained output `{output}` is not a type-preserving successor of {prior:?}")]
    InvalidRetainedTransition {
        /// Produced graph value name.
        output: String,
        /// Prior retained value.
        prior: GraphValueId,
    },
    /// A node binds one Program buffer more than once.
    #[error("program node `{node}` binds buffer `{buffer}` more than once")]
    DuplicatePort {
        /// Stable graph node name.
        node: String,
        /// Repeated Program-local buffer name.
        buffer: String,
    },
    /// One value is ambiguously aliased through two input buffers.
    #[error("program node `{node}` binds graph value {value:?} more than once")]
    DuplicateValueInput {
        /// Stable graph node name.
        node: String,
        /// Repeated graph value.
        value: GraphValueId,
    },
    /// A retained successor does not consume the prior value it replaces.
    #[error("retained output `{output}` names {prior:?} without consuming that prior value")]
    MissingRetainedInput {
        /// Produced graph value name.
        output: String,
        /// Unconsumed prior retained value.
        prior: GraphValueId,
    },
    /// Bounded loop bounds were invalid.
    #[error("invalid loop bounds: {0}")]
    InvalidLoopBounds(String),
    /// Subgraph port mapping was missing or incompatible.
    #[error("subgraph port mapping error for `{subgraph}`: {reason}")]
    SubgraphMapping {
        /// Subgraph prefix.
        subgraph: String,
        /// Incompatibility reason.
        reason: String,
    },
    /// External effect specification was invalid.
    #[error("invalid external effect on node `{node}`: {reason}")]
    InvalidEffect {
        /// Node name.
        node: String,
        /// Failure reason.
        reason: String,
    },
    /// Graph identity exceeded the wire-stable u32 range.
    #[error("ProgramGraph has more than {0} addressable values or nodes")]
    IdentityOverflow(u32),
    /// Canonical graph wire encoding or decoding failed.
    #[error("invalid ProgramGraph wire data: {0}")]
    Wire(String),
}

/// Structural sharing metrics across a ProgramGraph.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramGraphSharingMetrics {
    /// Total executable nodes in the graph.
    pub total_nodes: usize,
    /// Number of unique program bodies (by canonical fingerprint).
    pub unique_program_bodies: usize,
    /// Number of node instances sharing a body with another node.
    pub shared_instances: usize,
    /// Sharing ratio: total_nodes / unique_program_bodies.
    pub sharing_ratio: f64,
    /// Estimated unshared memory bytes if every node owned an independent copy.
    pub unshared_estimated_bytes: usize,
    /// Estimated memory bytes with structural sharing.
    pub shared_estimated_bytes: usize,
}

/// Versioned parameterized template for subgraphs with immutable structural sharing.
#[derive(Debug, Clone)]
pub struct ProgramGraphTemplate {
    /// Schema / template version.
    pub version: u32,
    /// Semantic template name.
    pub name: String,
    /// Underlying program body shared across all instantiations.
    pub program: Program,
    /// Expected input buffer names in declaration order.
    pub input_ports: Vec<String>,
    /// Expected output declarations in declaration order.
    pub output_ports: Vec<GraphOutput>,
}

impl ProgramGraphTemplate {
    /// Create a version-1 parameterized template from a certified Program body.
    pub fn new(
        name: impl Into<String>,
        program: Program,
        input_ports: Vec<String>,
        output_ports: Vec<GraphOutput>,
    ) -> Self {
        Self {
            version: 1,
            name: name.into(),
            program,
            input_ports,
            output_ports,
        }
    }

    /// Instantiate this template into a target graph with input bindings and output name prefixes.
    pub fn instantiate(
        &self,
        graph: &mut ProgramGraph,
        instance_name: impl Into<String>,
        input_bindings: Vec<(String, GraphValueId, ValueContract)>,
        output_name_prefix: &str,
    ) -> Result<(GraphNodeId, Vec<GraphValueId>), ProgramGraphError> {
        let instance_name = instance_name.into();
        let mut inputs = Vec::with_capacity(input_bindings.len());
        for (buffer, value, contract) in input_bindings {
            inputs.push(GraphInput {
                buffer,
                value,
                contract,
            });
        }
        let mut outputs = Vec::with_capacity(self.output_ports.len());
        for port in &self.output_ports {
            let mut out = port.clone();
            out.name = format!("{output_name_prefix}_{}", port.name);
            outputs.push(out);
        }
        // Clone of Program shares Arc<[BufferDecl]> and Arc<Vec<Node>> in O(1)
        graph.add_node(instance_name, self.program.clone(), inputs, outputs)
    }
}
