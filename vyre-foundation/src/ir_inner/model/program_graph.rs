//! Typed connections between reusable [`Program`](crate::ir::Program) values.
//!
//! `ProgramGraph` is composition metadata over existing Vyre IR. It is not a
//! second neural IR: every executable node remains an ordinary `Program`.

use std::collections::{BTreeMap, BTreeSet};

use rustc_hash::FxHashMap;
use thiserror::Error;

use super::op_signature::{BufferAccess, DataType};
use super::program::Program;

/// Canonical graph-local identity for one connected semantic value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GraphValueId(pub u32);

/// Canonical graph-local identity for one executable program node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GraphNodeId(pub u32);

/// One value dimension, either statically known or bound by graph configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ShapeDim {
    /// Exact element extent.
    Known(u64),
    /// Configuration symbol such as `batch`, `sequence`, or `hidden`.
    Symbol(String),
}

/// Semantic lifetime class used by compilation and runtime binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ValueLifetime {
    /// Immutable constant data shared by every invocation.
    Constant,
    /// Temporary data valid for one invocation.
    Invocation,
    /// Mutable data retained across submissions.
    Retained,
    /// Caller-visible graph result.
    Output,
}

/// Complete semantic contract for a connected graph value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

/// Connected executable Programs with canonical typed values.
#[derive(Debug, Default, Clone)]
pub struct ProgramGraph {
    nodes: Vec<ProgramGraphNode>,
    values: Vec<ProgramGraphValue>,
    names: FxHashMap<String, ()>,
}

impl ProgramGraph {
    /// Create an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Lift one frontend program into the canonical graph boundary.
    ///
    /// Runtime-sized buffers retain their unresolved zero extent. Call
    /// [`Self::from_program_with_runtime_counts`] when caller bytes establish
    /// exact element counts for artifact resource planning.
    pub fn from_program(
        node_name: impl Into<String>,
        program: Program,
    ) -> Result<Self, ProgramGraphError> {
        Self::from_program_with_runtime_counts(node_name, program, &BTreeMap::new())
    }

    /// Lift a frontend program while resolving runtime-sized host buffers.
    ///
    /// `runtime_counts` keys Program buffer names and supplies exact logical
    /// element counts. Only host-visible declarations with `count == 0` accept
    /// an override; stale names and static declarations fail closed.
    ///
    /// Every host-visible buffer becomes one typed external graph value.
    /// Workgroup-local scratch remains internal because callers cannot bind or
    /// retain it.
    pub fn from_program_with_runtime_counts(
        node_name: impl Into<String>,
        program: Program,
        runtime_counts: &BTreeMap<String, u64>,
    ) -> Result<Self, ProgramGraphError> {
        let node_name = node_name.into();
        for buffer_name in runtime_counts.keys() {
            let Some(buffer) = program.buffer(buffer_name) else {
                return Err(ProgramGraphError::MissingBuffer {
                    node: node_name,
                    buffer: buffer_name.clone(),
                });
            };
            if buffer.access() == BufferAccess::Workgroup || buffer.count() != 0 {
                return Err(ProgramGraphError::BufferContract {
                    node: node_name,
                    buffer: buffer_name.clone(),
                    reason: "runtime element-count override requires a host-visible declaration with count == 0".to_string(),
                });
            }
        }
        let mut graph = Self::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for buffer in program.buffers() {
            if buffer.access() == BufferAccess::Workgroup {
                continue;
            }
            let element_count = if buffer.count() == 0 {
                runtime_counts.get(buffer.name()).copied().unwrap_or(0)
            } else {
                u64::from(buffer.count())
            };
            let contract = ValueContract {
                dtype: buffer.element(),
                shape: vec![ShapeDim::Known(element_count)],
                access: buffer.access(),
                lifetime: if buffer.is_backend_allocated_output() {
                    ValueLifetime::Output
                } else if buffer.access() == BufferAccess::ReadWrite {
                    ValueLifetime::Retained
                } else {
                    ValueLifetime::Invocation
                },
            };
            if contract.lifetime == ValueLifetime::Output {
                outputs.push(GraphOutput {
                    buffer: buffer.name().to_string(),
                    name: buffer.name().to_string(),
                    contract,
                    retained_successor_of: None,
                });
            } else {
                let value = graph.add_external_value(buffer.name(), contract.clone())?;
                inputs.push(GraphInput {
                    buffer: buffer.name().to_string(),
                    value,
                    contract,
                });
            }
        }
        graph.add_node(node_name, program, inputs, outputs)?;
        Ok(graph)
    }

    /// Register a graph input, constant, or initial retained value.
    pub fn add_external_value(
        &mut self,
        name: impl Into<String>,
        contract: ValueContract,
    ) -> Result<GraphValueId, ProgramGraphError> {
        self.push_value(name.into(), contract, None, None)
    }

    /// Register external values as one transaction.
    ///
    /// Name or identity validation completes for the entire batch before any
    /// graph collection changes.
    pub fn add_external_values(
        &mut self,
        values: Vec<(String, ValueContract)>,
    ) -> Result<Vec<GraphValueId>, ProgramGraphError> {
        let mut batch_names = BTreeSet::new();
        let mut ids = Vec::with_capacity(values.len());
        for (offset, (name, _)) in values.iter().enumerate() {
            self.ensure_name_available(name)?;
            if !batch_names.insert(name.as_str()) {
                return Err(ProgramGraphError::DuplicateName(name.clone()));
            }
            let index = self
                .values
                .len()
                .checked_add(offset)
                .ok_or(ProgramGraphError::IdentityOverflow(u32::MAX))?;
            ids.push(GraphValueId(
                u32::try_from(index).map_err(|_| ProgramGraphError::IdentityOverflow(u32::MAX))?,
            ));
        }
        for ((name, contract), id) in values.into_iter().zip(ids.iter().copied()) {
            self.names.insert(name.clone(), ());
            self.values.push(ProgramGraphValue {
                id,
                name,
                contract,
                producer: None,
                consumers: Vec::new(),
                retained_successor_of: None,
            });
        }
        Ok(ids)
    }

    /// Append one Program node after all of its producers.
    ///
    /// Construction order is the topological schedule. This makes cycles
    /// unrepresentable except for explicit retained-value successions and the
    /// final retained-to-output transition of a caller-visible result buffer.
    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        program: Program,
        inputs: Vec<GraphInput>,
        outputs: Vec<GraphOutput>,
    ) -> Result<(GraphNodeId, Vec<GraphValueId>), ProgramGraphError> {
        let name = name.into();
        self.ensure_name_available(&name)?;
        let node_id = GraphNodeId(
            u32::try_from(self.nodes.len())
                .map_err(|_| ProgramGraphError::IdentityOverflow(u32::MAX))?,
        );

        let mut new_names = BTreeSet::new();
        new_names.insert(name.as_str());
        let mut output_ids = Vec::with_capacity(outputs.len());
        for (offset, output) in outputs.iter().enumerate() {
            self.ensure_name_available(&output.name)?;
            if !new_names.insert(output.name.as_str()) {
                return Err(ProgramGraphError::DuplicateName(output.name.clone()));
            }
            let index = self
                .values
                .len()
                .checked_add(offset)
                .ok_or(ProgramGraphError::IdentityOverflow(u32::MAX))?;
            output_ids.push(GraphValueId(
                u32::try_from(index).map_err(|_| ProgramGraphError::IdentityOverflow(u32::MAX))?,
            ));
        }

        let mut bound = BTreeSet::new();
        let mut bound_values = BTreeSet::new();
        for input in &inputs {
            if !bound.insert(input.buffer.as_str()) {
                return Err(ProgramGraphError::DuplicatePort {
                    node: name,
                    buffer: input.buffer.clone(),
                });
            }
            if !bound_values.insert(input.value) {
                return Err(ProgramGraphError::DuplicateValueInput {
                    node: name,
                    value: input.value,
                });
            }
            let value = self
                .values
                .get(input.value.0 as usize)
                .ok_or(ProgramGraphError::MissingValue(input.value))?;
            if value.contract.dtype != input.contract.dtype
                || value.contract.shape != input.contract.shape
                || value.contract.lifetime != input.contract.lifetime
            {
                return Err(ProgramGraphError::InputContract {
                    node: name,
                    buffer: input.buffer.clone(),
                    value: input.value,
                    actual: value.contract.clone(),
                    expected: input.contract.clone(),
                });
            }
            validate_buffer(
                &name,
                &program,
                &input.buffer,
                &input.contract,
                PortRole::Input,
            )?;
        }
        for output in &outputs {
            if let Some(prior_id) = output.retained_successor_of {
                let prior = self
                    .values
                    .get(prior_id.0 as usize)
                    .ok_or(ProgramGraphError::MissingValue(prior_id))?;
                if !inputs.iter().any(|input| input.value == prior_id) {
                    return Err(ProgramGraphError::MissingRetainedInput {
                        output: output.name.clone(),
                        prior: prior_id,
                    });
                }
                let caller_output_transition = prior.contract.lifetime == ValueLifetime::Retained
                    && output.contract.lifetime == ValueLifetime::Output
                    && prior.contract.dtype == output.contract.dtype
                    && prior.contract.shape == output.contract.shape
                    && prior.contract.access == output.contract.access
                    && program
                        .buffers()
                        .iter()
                        .find(|buffer| buffer.name() == output.buffer)
                        .is_some_and(|buffer| buffer.is_output());
                if !caller_output_transition
                    && (prior.contract.lifetime != ValueLifetime::Retained
                        || output.contract.lifetime != ValueLifetime::Retained
                        || prior.contract != output.contract)
                {
                    return Err(ProgramGraphError::InvalidRetainedTransition {
                        output: output.name.clone(),
                        prior: prior_id,
                    });
                }
            }
            let retained_rebind = output.retained_successor_of.is_some_and(|prior| {
                inputs
                    .iter()
                    .any(|input| input.buffer == output.buffer && input.value == prior)
            });
            if !bound.insert(output.buffer.as_str()) && !retained_rebind {
                return Err(ProgramGraphError::DuplicatePort {
                    node: name,
                    buffer: output.buffer.clone(),
                });
            }
            validate_buffer(
                &name,
                &program,
                &output.buffer,
                &output.contract,
                PortRole::Output,
            )?;
        }

        self.names.insert(name.clone(), ());
        for output in &outputs {
            self.names.insert(output.name.clone(), ());
        }
        let mut consumed = BTreeSet::new();
        for input in &inputs {
            if consumed.insert(input.value) {
                self.values[input.value.0 as usize].consumers.push(node_id);
            }
        }
        let output_ports = outputs.clone();
        for (output, id) in outputs.into_iter().zip(output_ids.iter().copied()) {
            self.values.push(ProgramGraphValue {
                id,
                name: output.name,
                contract: output.contract,
                producer: Some(node_id),
                consumers: Vec::new(),
                retained_successor_of: output.retained_successor_of,
            });
        }
        self.nodes.push(ProgramGraphNode {
            id: node_id,
            name,
            program,
            inputs,
            outputs: output_ids.clone(),
            output_ports,
        });
        Ok((node_id, output_ids))
    }

    /// Swap one node's executable program and input ports, keeping its output
    /// values.
    ///
    /// A node's outputs are graph values other nodes consume by id, so changing
    /// them is a delete and an insert rather than a replacement. What this
    /// changes is the body that computes them and the values it reads, which is
    /// what a recompiled node needs. `output_ports` are required to match the
    /// node's current ports exactly, so a caller that means to change the
    /// dataflow shape is refused here instead of leaving consumers pointing at
    /// values the new program never writes.
    ///
    /// Consumer lists are rewired for the inputs that changed: the node is
    /// dropped from every value it no longer reads and added to every value it
    /// now reads.
    pub fn replace_node(
        &mut self,
        node_id: GraphNodeId,
        program: Program,
        inputs: Vec<GraphInput>,
        outputs: Vec<GraphOutput>,
    ) -> Result<(), ProgramGraphError> {
        let index = node_id.0 as usize;
        let name = self
            .nodes
            .get(index)
            .ok_or(ProgramGraphError::MissingNode(node_id))?
            .name
            .clone();

        if outputs != self.nodes[index].output_ports {
            return Err(ProgramGraphError::InvalidReplacementOutputs { node: node_id });
        }

        let mut bound = BTreeSet::new();
        let mut bound_values = BTreeSet::new();
        for input in &inputs {
            if !bound.insert(input.buffer.as_str()) {
                return Err(ProgramGraphError::DuplicatePort {
                    node: name,
                    buffer: input.buffer.clone(),
                });
            }
            if !bound_values.insert(input.value) {
                return Err(ProgramGraphError::DuplicateValueInput {
                    node: name,
                    value: input.value,
                });
            }
            let value = self
                .values
                .get(input.value.0 as usize)
                .ok_or(ProgramGraphError::MissingValue(input.value))?;
            if value.contract.dtype != input.contract.dtype
                || value.contract.shape != input.contract.shape
                || value.contract.lifetime != input.contract.lifetime
            {
                return Err(ProgramGraphError::InputContract {
                    node: name,
                    buffer: input.buffer.clone(),
                    value: input.value,
                    actual: value.contract.clone(),
                    expected: input.contract.clone(),
                });
            }
            validate_buffer(
                &name,
                &program,
                &input.buffer,
                &input.contract,
                PortRole::Input,
            )?;
        }
        for output in &outputs {
            validate_buffer(
                &name,
                &program,
                &output.buffer,
                &output.contract,
                PortRole::Output,
            )?;
        }
        // A retained output reads its predecessor through this node's inputs.
        // Changing the inputs can drop that predecessor, which `add_node`
        // refuses at insertion and which must stay refused on replacement.
        for output in &outputs {
            if let Some(prior) = output.retained_successor_of {
                if !inputs.iter().any(|input| input.value == prior) {
                    return Err(ProgramGraphError::MissingRetainedInput {
                        output: output.name.clone(),
                        prior,
                    });
                }
            }
        }

        // Nothing above this line has mutated the graph, so a refusal leaves the
        // node exactly as it was.
        let previous: BTreeSet<GraphValueId> = self.nodes[index]
            .inputs
            .iter()
            .map(|input| input.value)
            .collect();
        let next: BTreeSet<GraphValueId> = inputs.iter().map(|input| input.value).collect();
        for dropped in previous.difference(&next) {
            self.values[dropped.0 as usize]
                .consumers
                .retain(|consumer| *consumer != node_id);
        }
        for added in next.difference(&previous) {
            self.values[added.0 as usize].consumers.push(node_id);
        }

        let node = &mut self.nodes[index];
        node.program = program;
        node.inputs = inputs;
        Ok(())
    }

    /// Nodes in their validated topological execution order.
    #[must_use]
    pub fn nodes(&self) -> &[ProgramGraphNode] {
        &self.nodes
    }

    /// Canonical connected values.
    #[must_use]
    pub fn values(&self) -> &[ProgramGraphValue] {
        &self.values
    }

    /// Topological node schedule.
    #[must_use]
    pub fn schedule(&self) -> Vec<GraphNodeId> {
        self.nodes.iter().map(|node| node.id).collect()
    }

    /// Compute allocation liveness from producer and consumer schedule indices.
    #[must_use]
    pub fn liveness_intervals(&self) -> Vec<LivenessInterval> {
        self.values
            .iter()
            .map(|value| {
                let start = value.producer.map_or(0, |producer| producer.0 as usize);
                let end = value
                    .consumers
                    .iter()
                    .map(|consumer| consumer.0 as usize)
                    .max()
                    .unwrap_or(start);
                LivenessInterval {
                    value: value.id,
                    start,
                    end,
                }
            })
            .collect()
    }

    fn push_value(
        &mut self,
        name: String,
        contract: ValueContract,
        producer: Option<GraphNodeId>,
        retained_successor_of: Option<GraphValueId>,
    ) -> Result<GraphValueId, ProgramGraphError> {
        self.ensure_name_available(&name)?;
        let id = GraphValueId(
            u32::try_from(self.values.len())
                .map_err(|_| ProgramGraphError::IdentityOverflow(u32::MAX))?,
        );
        self.names.insert(name.clone(), ());
        self.values.push(ProgramGraphValue {
            id,
            name,
            contract,
            producer,
            consumers: Vec::new(),
            retained_successor_of,
        });
        Ok(id)
    }

    fn ensure_name_available(&self, name: &str) -> Result<(), ProgramGraphError> {
        if self.names.contains_key(name) {
            return Err(ProgramGraphError::DuplicateName(name.to_string()));
        }
        Ok(())
    }

    /// Compute structural sharing metrics across all nodes in the graph.
    #[must_use]
    pub fn structural_sharing_metrics(&self) -> ProgramGraphSharingMetrics {
        let total_nodes = self.nodes.len();
        if total_nodes == 0 {
            return ProgramGraphSharingMetrics {
                total_nodes: 0,
                unique_program_bodies: 0,
                shared_instances: 0,
                sharing_ratio: 1.0,
                unshared_estimated_bytes: 0,
                shared_estimated_bytes: 0,
            };
        }
        let mut unique_fingerprints = rustc_hash::FxHashSet::default();
        let mut unique_body_bytes = 0usize;
        let mut total_unshared_bytes = 0usize;

        for node in &self.nodes {
            let fp = node.program.fingerprint();
            let node_bytes = estimate_program_bytes(&node.program);
            total_unshared_bytes = total_unshared_bytes.saturating_add(node_bytes);
            if unique_fingerprints.insert(fp) {
                unique_body_bytes = unique_body_bytes.saturating_add(node_bytes);
            }
        }
        let unique_count = unique_fingerprints.len();
        let shared_instances = total_nodes.saturating_sub(unique_count);
        let sharing_ratio = if unique_count > 0 {
            total_nodes as f64 / unique_count as f64
        } else {
            1.0
        };
        let shared_estimated_bytes = unique_body_bytes
            .saturating_add(total_nodes.saturating_mul(std::mem::size_of::<ProgramGraphNode>()));

        ProgramGraphSharingMetrics {
            total_nodes,
            unique_program_bodies: unique_count,
            shared_instances,
            sharing_ratio,
            unshared_estimated_bytes: total_unshared_bytes,
            shared_estimated_bytes,
        }
    }

    /// Canonicalize and intern equivalent Program bodies across nodes so they
    /// share immutable Arc references.
    pub fn canonicalize_structural_sharing(&mut self) {
        let mut interned: rustc_hash::FxHashMap<[u8; 32], Program> = rustc_hash::FxHashMap::default();
        for node in &mut self.nodes {
            let fp = node.program.fingerprint();
            if let Some(canonical) = interned.get(&fp) {
                node.program = canonical.clone();
            } else {
                interned.insert(fp, node.program.clone());
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PortRole {
    Input,
    Output,
}

fn validate_buffer(
    node: &str,
    program: &Program,
    buffer_name: &str,
    contract: &ValueContract,
    role: PortRole,
) -> Result<(), ProgramGraphError> {
    let buffer = program
        .buffers()
        .iter()
        .find(|buffer| buffer.name() == buffer_name)
        .ok_or_else(|| ProgramGraphError::MissingBuffer {
            node: node.to_string(),
            buffer: buffer_name.to_string(),
        })?;
    if buffer.element() != contract.dtype {
        return Err(ProgramGraphError::BufferContract {
            node: node.to_string(),
            buffer: buffer_name.to_string(),
            reason: format!(
                "Program uses {:?}, graph uses {:?}",
                buffer.element(),
                contract.dtype
            ),
        });
    }
    if let Some(elements) = static_element_count(&contract.shape).map_err(|reason| {
        ProgramGraphError::BufferContract {
            node: node.to_string(),
            buffer: buffer_name.to_string(),
            reason,
        }
    })? {
        if buffer.count() != 0 && elements != u64::from(buffer.count()) {
            return Err(ProgramGraphError::BufferContract {
                node: node.to_string(),
                buffer: buffer_name.to_string(),
                reason: format!(
                    "Program declares {} elements, graph shape requires {elements}",
                    buffer.count()
                ),
            });
        }
    }
    let access_satisfies_contract = match contract.access {
        BufferAccess::ReadOnly => matches!(
            buffer.access(),
            BufferAccess::ReadOnly | BufferAccess::ReadWrite | BufferAccess::Uniform
        ),
        BufferAccess::ReadWrite => buffer.access() == BufferAccess::ReadWrite,
        BufferAccess::WriteOnly => {
            matches!(
                buffer.access(),
                BufferAccess::WriteOnly | BufferAccess::ReadWrite
            )
        }
        BufferAccess::Uniform => buffer.access() == BufferAccess::Uniform,
        _ => false,
    };
    if !access_satisfies_contract {
        return Err(ProgramGraphError::BufferContract {
            node: node.to_string(),
            buffer: buffer_name.to_string(),
            reason: format!(
                "Program access {:?} does not satisfy graph access {:?}",
                buffer.access(),
                contract.access
            ),
        });
    }
    let access = buffer.access();
    let compatible = match role {
        PortRole::Input => matches!(
            access,
            BufferAccess::ReadOnly | BufferAccess::ReadWrite | BufferAccess::Uniform
        ),
        PortRole::Output => matches!(access, BufferAccess::ReadWrite | BufferAccess::WriteOnly),
    };
    if !compatible {
        return Err(ProgramGraphError::BufferContract {
            node: node.to_string(),
            buffer: buffer_name.to_string(),
            reason: format!("{role:?} port cannot use {access:?} access"),
        });
    }
    Ok(())
}

fn static_element_count(shape: &[ShapeDim]) -> Result<Option<u64>, String> {
    let mut elements = 1_u64;
    for dimension in shape {
        match dimension {
            ShapeDim::Known(extent) => {
                elements = elements.checked_mul(*extent).ok_or_else(|| {
                    "graph shape element count overflows u64; reduce or shard dimensions"
                        .to_string()
                })?;
            }
            ShapeDim::Symbol(_) => return Ok(None),
        }
    }
    Ok(Some(elements))
}
fn estimate_program_bytes(program: &Program) -> usize {
    let base = std::mem::size_of::<Program>();
    let buffer_bytes = program
        .buffers()
        .iter()
        .map(|b| std::mem::size_of::<crate::ir::BufferDecl>() + b.name().len())
        .sum::<usize>();
    let mut node_bytes = program.entry().len() * std::mem::size_of::<crate::ir::Node>();
    for node in program.entry() {
        estimate_node_bytes(node, &mut node_bytes);
    }
    base + buffer_bytes + node_bytes
}

fn estimate_node_bytes(node: &crate::ir::Node, bytes: &mut usize) {
    use crate::ir::Node;
    *bytes += std::mem::size_of::<Node>();
    match node {
        Node::Let { name, value, .. } | Node::Assign { name, value, .. } => {
            *bytes += name.as_str().len();
            estimate_expr_bytes(value, bytes);
        }
        Node::Store { buffer, index, value, .. } => {
            *bytes += buffer.as_str().len();
            estimate_expr_bytes(index, bytes);
            estimate_expr_bytes(value, bytes);
        }
        Node::If { cond, then, otherwise } => {
            estimate_expr_bytes(cond, bytes);
            for node in then.iter().chain(otherwise.iter()) {
                estimate_node_bytes(node, bytes);
            }
        }
        Node::Loop { from, to, body, .. } => {
            estimate_expr_bytes(from, bytes);
            estimate_expr_bytes(to, bytes);
            for node in body {
                estimate_node_bytes(node, bytes);
            }
        }
        Node::AsyncLoad { offset, size, .. } | Node::AsyncStore { offset, size, .. } => {
            estimate_expr_bytes(offset, bytes);
            estimate_expr_bytes(size, bytes);
        }
        Node::Trap { address, .. } => {
            estimate_expr_bytes(address, bytes);
        }
        Node::Block(body) => {
            for node in body {
                estimate_node_bytes(node, bytes);
            }
        }
        Node::Region { body, .. } => {
            for node in body.iter() {
                estimate_node_bytes(node, bytes);
            }
        }
        Node::TileElementwise { body, .. } => {
            for node in body {
                estimate_node_bytes(node, bytes);
            }
        }
        Node::Opaque(_) => {
            *bytes += 64;
        }
        _ => {}
    }
}

fn estimate_expr_bytes(expr: &crate::ir::Expr, bytes: &mut usize) {
    use crate::ir::Expr;
    *bytes += std::mem::size_of::<Expr>();
    match expr {
        Expr::Load { buffer, index, .. } => {
            *bytes += buffer.as_str().len();
            estimate_expr_bytes(index, bytes);
        }
        Expr::BinOp { left, right, .. } => {
            estimate_expr_bytes(left, bytes);
            estimate_expr_bytes(right, bytes);
        }
        Expr::UnOp { operand, .. }
        | Expr::Cast { value: operand, .. }
        | Expr::SubgroupBallot { cond: operand }
        | Expr::SubgroupReduce { value: operand, .. } => {
            estimate_expr_bytes(operand, bytes);
        }
        Expr::Call { args, .. } => {
            for arg in args {
                estimate_expr_bytes(arg, bytes);
            }
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            estimate_expr_bytes(cond, bytes);
            estimate_expr_bytes(true_val, bytes);
            estimate_expr_bytes(false_val, bytes);
        }
        Expr::Fma { a, b, c } => {
            estimate_expr_bytes(a, bytes);
            estimate_expr_bytes(b, bytes);
            estimate_expr_bytes(c, bytes);
        }
        Expr::Atomic {
            index,
            expected,
            value,
            ..
        } => {
            estimate_expr_bytes(index, bytes);
            if let Some(expected) = expected {
                estimate_expr_bytes(expected, bytes);
            }
            estimate_expr_bytes(value, bytes);
        }
        Expr::SubgroupShuffle { value, lane } => {
            estimate_expr_bytes(value, bytes);
            estimate_expr_bytes(lane, bytes);
        }
        Expr::Var(name) => {
            *bytes += name.as_str().len();
        }
        Expr::BufLen { buffer, .. } => {
            *bytes += buffer.as_str().len();
        }
        Expr::Opaque(_) => {
            *bytes += 64;
        }
        _ => {}
    }
}

