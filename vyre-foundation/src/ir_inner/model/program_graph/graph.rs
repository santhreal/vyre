//! Core [`ProgramGraph`] data structure and whole-graph operations.

use std::collections::{BTreeMap, BTreeSet};

use rustc_hash::FxHashMap;

use super::types::{
    ControlBounds, ExternalEffect, GraphInput, GraphNodeId, GraphOutput, GraphValueId,
    LivenessInterval, ProgramGraphError, ProgramGraphNode, ProgramGraphSharingMetrics,
    ProgramGraphValue, ShapeDim, ValueContract, ValueLifetime,
};
use super::validate::{estimate_program_bytes, validate_buffer, PortRole};
use crate::ir_inner::model::op_signature::{BufferAccess, DataType};
use crate::ir_inner::model::program::Program;

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
        let mut interned: rustc_hash::FxHashMap<[u8; 32], Program> =
            rustc_hash::FxHashMap::default();
        for node in &mut self.nodes {
            let fp = node.program.fingerprint();
            if let Some(canonical) = interned.get(&fp) {
                node.program = canonical.clone();
            } else {
                interned.insert(fp, node.program.clone());
            }
        }
    }

    /// Register a streaming dataflow value (FIFO channel or queue).
    pub fn add_stream(
        &mut self,
        name: impl Into<String>,
        contract: ValueContract,
    ) -> Result<GraphValueId, ProgramGraphError> {
        let mut stream_contract = contract;
        stream_contract.lifetime = ValueLifetime::Stream;
        self.push_value(name.into(), stream_contract, None, None)
    }

    /// Add an operation node with declared inputs and outputs.
    pub fn add_operation_node(
        &mut self,
        name: impl Into<String>,
        program: Program,
        inputs: Vec<GraphInput>,
        outputs: Vec<GraphOutput>,
    ) -> Result<(GraphNodeId, Vec<GraphValueId>), ProgramGraphError> {
        self.add_node(name, program, inputs, outputs)
    }

    /// Inline a nested reusable [`ProgramGraph`] into `self`.
    ///
    /// `name_prefix` namespaces every node and produced value of the subgraph.
    /// `input_mapping` maps each external input value ID of `subgraph` to an existing
    /// value ID in `self`.
    /// Returns the mapped value IDs in `self` corresponding to the subgraph's output values.
    pub fn inline_subgraph(
        &mut self,
        name_prefix: &str,
        subgraph: &ProgramGraph,
        input_mapping: &BTreeMap<GraphValueId, GraphValueId>,
    ) -> Result<Vec<GraphValueId>, ProgramGraphError> {
        let mut value_map = BTreeMap::<GraphValueId, GraphValueId>::new();

        for sub_val in subgraph.values() {
            if sub_val.producer.is_none() {
                let mapped_in_self = input_mapping.get(&sub_val.id).copied().ok_or_else(|| {
                    ProgramGraphError::SubgraphMapping {
                        subgraph: name_prefix.to_string(),
                        reason: format!(
                            "missing input mapping for subgraph external value `{}` ({:?})",
                            sub_val.name, sub_val.id
                        ),
                    }
                })?;
                let self_val = self
                    .values
                    .get(mapped_in_self.0 as usize)
                    .ok_or(ProgramGraphError::MissingValue(mapped_in_self))?;
                if self_val.contract.dtype != sub_val.contract.dtype
                    || self_val.contract.shape != sub_val.contract.shape
                {
                    return Err(ProgramGraphError::SubgraphMapping {
                        subgraph: name_prefix.to_string(),
                        reason: format!(
                            "input contract mismatch on `{}`: expected {:?}, got {:?}",
                            sub_val.name, sub_val.contract, self_val.contract
                        ),
                    });
                }
                value_map.insert(sub_val.id, mapped_in_self);
            }
        }

        for sub_node in subgraph.nodes() {
            let namespaced_node_name = format!("{name_prefix}_{}", sub_node.name);
            let mut node_inputs = Vec::with_capacity(sub_node.inputs.len());
            for in_port in &sub_node.inputs {
                let self_val_id = value_map.get(&in_port.value).copied().ok_or_else(|| {
                    ProgramGraphError::SubgraphMapping {
                        subgraph: name_prefix.to_string(),
                        reason: format!(
                            "unresolved input value {:?} for port `{}` in node `{}`",
                            in_port.value, in_port.buffer, sub_node.name
                        ),
                    }
                })?;
                node_inputs.push(GraphInput {
                    buffer: in_port.buffer.clone(),
                    value: self_val_id,
                    contract: in_port.contract.clone(),
                });
            }

            let mut node_outputs = Vec::with_capacity(sub_node.output_ports.len());
            for out_port in &sub_node.output_ports {
                let namespaced_out_name = format!("{name_prefix}_{}", out_port.name);
                let mapped_retained = match out_port.retained_successor_of {
                    Some(prior) => Some(*value_map.get(&prior).ok_or_else(|| {
                        ProgramGraphError::SubgraphMapping {
                            subgraph: name_prefix.to_string(),
                            reason: format!(
                                "unresolved retained predecessor {:?} for output `{}`",
                                prior, out_port.name
                            ),
                        }
                    })?),
                    None => None,
                };
                node_outputs.push(GraphOutput {
                    buffer: out_port.buffer.clone(),
                    name: namespaced_out_name,
                    contract: out_port.contract.clone(),
                    retained_successor_of: mapped_retained,
                });
            }

            let (_, self_out_ids) = self.add_node(
                namespaced_node_name,
                sub_node.program.clone(),
                node_inputs,
                node_outputs,
            )?;

            for (sub_out_id, self_out_id) in sub_node.outputs.iter().zip(self_out_ids.iter()) {
                value_map.insert(*sub_out_id, *self_out_id);
            }
        }

        let mut results = Vec::new();
        for sub_val in subgraph.values() {
            if sub_val.contract.lifetime == ValueLifetime::Output {
                if let Some(mapped) = value_map.get(&sub_val.id) {
                    results.push(*mapped);
                }
            }
        }
        if results.is_empty() {
            if let Some(last_node) = subgraph.nodes().last() {
                for out_id in &last_node.outputs {
                    if let Some(mapped) = value_map.get(out_id) {
                        results.push(*mapped);
                    }
                }
            }
        }
        Ok(results)
    }

    /// Add a bounded loop by unrolling the step body subgraph for `bounds.max_steps`.
    pub fn add_bounded_loop(
        &mut self,
        name_prefix: &str,
        body: &ProgramGraph,
        loop_carried_inputs: &[GraphValueId],
        step_inputs: &[GraphValueId],
        bounds: ControlBounds,
    ) -> Result<Vec<GraphValueId>, ProgramGraphError> {
        if bounds.max_steps == 0 {
            return Err(ProgramGraphError::InvalidLoopBounds(
                "bounded loop max_steps must be greater than zero".to_string(),
            ));
        }

        let mut current_state = loop_carried_inputs.to_vec();
        for step in 0..bounds.max_steps {
            let step_prefix = format!("{name_prefix}_step{step}");
            let mut input_map = BTreeMap::new();
            let ext_values: Vec<_> = body
                .values()
                .iter()
                .filter(|v| v.producer.is_none())
                .collect();

            for (idx, state_val) in current_state.iter().enumerate() {
                if let Some(ext_val) = ext_values.get(idx) {
                    input_map.insert(ext_val.id, *state_val);
                }
            }
            for (idx, step_in) in step_inputs.iter().enumerate() {
                let ext_idx = current_state.len() + idx;
                if let Some(ext_val) = ext_values.get(ext_idx) {
                    input_map.insert(ext_val.id, *step_in);
                }
            }

            let step_outs = self.inline_subgraph(&step_prefix, body, &input_map)?;
            current_state = step_outs;
        }

        Ok(current_state)
    }

    /// Add a conditional branch by inlining then/else subgraphs and joining outputs with a select node.
    pub fn add_conditional_branch(
        &mut self,
        name_prefix: &str,
        condition: GraphValueId,
        then_graph: &ProgramGraph,
        else_graph: &ProgramGraph,
        inputs: &[GraphValueId],
    ) -> Result<Vec<GraphValueId>, ProgramGraphError> {
        let mut then_in_map = BTreeMap::new();
        let mut else_in_map = BTreeMap::new();
        let then_exts: Vec<_> = then_graph
            .values()
            .iter()
            .filter(|v| v.producer.is_none())
            .collect();
        let else_exts: Vec<_> = else_graph
            .values()
            .iter()
            .filter(|v| v.producer.is_none())
            .collect();
        for (i, val) in inputs.iter().enumerate() {
            if let Some(te) = then_exts.get(i) {
                then_in_map.insert(te.id, *val);
            }
            if let Some(ee) = else_exts.get(i) {
                else_in_map.insert(ee.id, *val);
            }
        }

        let then_outs =
            self.inline_subgraph(&format!("{name_prefix}_then"), then_graph, &then_in_map)?;
        let else_outs =
            self.inline_subgraph(&format!("{name_prefix}_else"), else_graph, &else_in_map)?;

        if then_outs.len() != else_outs.len() {
            return Err(ProgramGraphError::SubgraphMapping {
                subgraph: name_prefix.to_string(),
                reason: format!(
                    "then branch produced {} outputs but else branch produced {}",
                    then_outs.len(),
                    else_outs.len()
                ),
            });
        }

        let mut joined_outs = Vec::with_capacity(then_outs.len());
        for (idx, (t_out, e_out)) in then_outs.iter().zip(else_outs.iter()).enumerate() {
            let t_val = &self.values[t_out.0 as usize];
            let e_val = &self.values[e_out.0 as usize];
            if t_val.contract.dtype != e_val.contract.dtype
                || t_val.contract.shape != e_val.contract.shape
            {
                return Err(ProgramGraphError::SubgraphMapping {
                    subgraph: name_prefix.to_string(),
                    reason: format!(
                        "branch output {idx} contract mismatch: then is {:?}, else is {:?}",
                        t_val.contract, e_val.contract
                    ),
                });
            }

            let join_name = format!("{name_prefix}_join{idx}");
            let out_name = format!("{name_prefix}_cond_out{idx}");
            let contract = t_val.contract.clone();
            let count = match contract.shape.first() {
                Some(ShapeDim::Known(k)) => *k as u32,
                _ => 1,
            };

            let join_prog = Program::wrapped(
                vec![
                    crate::ir::BufferDecl::read("cond", 0, DataType::U32).with_count(1),
                    crate::ir::BufferDecl::read("then_buf", 1, contract.dtype.clone())
                        .with_count(count),
                    crate::ir::BufferDecl::read("else_buf", 2, contract.dtype.clone())
                        .with_count(count),
                    crate::ir::BufferDecl::output("out", 3, contract.dtype.clone())
                        .with_count(count),
                ],
                [count.max(1), 1, 1],
                vec![crate::ir::Node::if_then_else(
                    crate::ir::Expr::ne(
                        crate::ir::Expr::load("cond", crate::ir::Expr::u32(0)),
                        crate::ir::Expr::u32(0),
                    ),
                    vec![crate::ir::Node::store(
                        "out",
                        crate::ir::Expr::gid_x(),
                        crate::ir::Expr::load("then_buf", crate::ir::Expr::gid_x()),
                    )],
                    vec![crate::ir::Node::store(
                        "out",
                        crate::ir::Expr::gid_x(),
                        crate::ir::Expr::load("else_buf", crate::ir::Expr::gid_x()),
                    )],
                )],
            );

            let (_, outs) = self.add_node(
                join_name,
                join_prog,
                vec![
                    GraphInput {
                        buffer: "cond".to_string(),
                        value: condition,
                        contract: ValueContract {
                            dtype: DataType::U32,
                            shape: vec![ShapeDim::Known(1)],
                            access: BufferAccess::ReadOnly,
                            lifetime: self.values[condition.0 as usize].contract.lifetime,
                        },
                    },
                    GraphInput {
                        buffer: "then_buf".to_string(),
                        value: *t_out,
                        contract: contract.clone(),
                    },
                    GraphInput {
                        buffer: "else_buf".to_string(),
                        value: *e_out,
                        contract: contract.clone(),
                    },
                ],
                vec![GraphOutput {
                    buffer: "out".to_string(),
                    name: out_name,
                    contract,
                    retained_successor_of: None,
                }],
            )?;
            joined_outs.extend(outs);
        }

        Ok(joined_outs)
    }

    /// Add an external effect barrier node enforcing synchronization and ordering.
    pub fn add_effect_barrier(
        &mut self,
        name: impl Into<String>,
        _effect: ExternalEffect,
        inputs: Vec<GraphInput>,
        outputs: Vec<GraphOutput>,
    ) -> Result<(GraphNodeId, Vec<GraphValueId>), ProgramGraphError> {
        let node_name = name.into();
        let count = inputs
            .first()
            .map(|i| match i.contract.shape.first() {
                Some(ShapeDim::Known(k)) => *k as u32,
                _ => 1,
            })
            .unwrap_or(1);

        let mut buffer_decls = Vec::new();
        let mut stmts = vec![crate::ir::Node::LogicalBarrier {
            ordering: crate::ir::MemoryOrdering::SeqCst,
        }];
        for (i, inp) in inputs.iter().enumerate() {
            buffer_decls.push(
                crate::ir::BufferDecl::read(&inp.buffer, i as u32, inp.contract.dtype.clone())
                    .with_count(count),
            );
        }
        for (j, out) in outputs.iter().enumerate() {
            buffer_decls.push(
                crate::ir::BufferDecl::output(
                    &out.buffer,
                    (inputs.len() + j) as u32,
                    out.contract.dtype.clone(),
                )
                .with_count(count),
            );
            if let Some(inp) = inputs.get(j) {
                stmts.push(crate::ir::Node::store(
                    &out.buffer,
                    crate::ir::Expr::gid_x(),
                    crate::ir::Expr::load(&inp.buffer, crate::ir::Expr::gid_x()),
                ));
            }
        }

        let prog = Program::wrapped(buffer_decls, [count.max(1), 1, 1], stmts);

        self.add_node(node_name, prog, inputs, outputs)
    }
}
