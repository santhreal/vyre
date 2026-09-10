//! Transactional builder for [`ProgramGraph`](super::ProgramGraph) compositions.

use std::collections::BTreeMap;

use super::contracts::{
    ControlBounds, ExternalEffect, GraphInput, GraphNodeId, GraphOutput, GraphValueId,
    ProgramGraphError, ShapeDim, ValueContract, ValueLifetime,
};
use super::graph::ProgramGraph;
use crate::ir_inner::model::op_signature::{BufferAccess, DataType};
use crate::ir_inner::model::program::Program;

/// Transactional, fluent domain-neutral builder for whole [`ProgramGraph`] compositions.
#[derive(Debug, Default, Clone)]
pub struct ProgramGraphBuilder {
    graph: ProgramGraph,
}

impl ProgramGraphBuilder {
    /// Create a new empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an invocation input value.
    pub fn input(
        &mut self,
        name: impl Into<String>,
        dtype: DataType,
        shape: Vec<ShapeDim>,
    ) -> Result<GraphValueId, ProgramGraphError> {
        self.graph.add_external_value(
            name,
            ValueContract {
                dtype,
                shape,
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        )
    }

    /// Register an immutable constant value.
    pub fn constant(
        &mut self,
        name: impl Into<String>,
        dtype: DataType,
        shape: Vec<ShapeDim>,
    ) -> Result<GraphValueId, ProgramGraphError> {
        self.graph.add_external_value(
            name,
            ValueContract {
                dtype,
                shape,
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Constant,
            },
        )
    }

    /// Register a mutable retained state value.
    pub fn retained_state(
        &mut self,
        name: impl Into<String>,
        dtype: DataType,
        shape: Vec<ShapeDim>,
    ) -> Result<GraphValueId, ProgramGraphError> {
        self.graph.add_external_value(
            name,
            ValueContract {
                dtype,
                shape,
                access: BufferAccess::ReadWrite,
                lifetime: ValueLifetime::Retained,
            },
        )
    }

    /// Register a streaming dataflow value (FIFO channel or queue).
    pub fn stream(
        &mut self,
        name: impl Into<String>,
        dtype: DataType,
        shape: Vec<ShapeDim>,
    ) -> Result<GraphValueId, ProgramGraphError> {
        self.graph.add_stream(
            name,
            ValueContract {
                dtype,
                shape,
                access: BufferAccess::ReadWrite,
                lifetime: ValueLifetime::Stream,
            },
        )
    }

    /// Add an executable Program node.
    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        program: Program,
        inputs: Vec<GraphInput>,
        outputs: Vec<GraphOutput>,
    ) -> Result<(GraphNodeId, Vec<GraphValueId>), ProgramGraphError> {
        self.graph.add_node(name, program, inputs, outputs)
    }

    /// Add a registered operation node.
    pub fn add_registered_operation(
        &mut self,
        name: impl Into<String>,
        program: Program,
        inputs: Vec<GraphInput>,
        outputs: Vec<GraphOutput>,
    ) -> Result<(GraphNodeId, Vec<GraphValueId>), ProgramGraphError> {
        self.graph
            .add_operation_node(name, program, inputs, outputs)
    }

    /// Inline a nested reusable subgraph.
    pub fn inline_subgraph(
        &mut self,
        name_prefix: &str,
        subgraph: &ProgramGraph,
        input_mapping: &BTreeMap<GraphValueId, GraphValueId>,
    ) -> Result<Vec<GraphValueId>, ProgramGraphError> {
        self.graph
            .inline_subgraph(name_prefix, subgraph, input_mapping)
    }

    /// Add a bounded loop.
    pub fn add_bounded_loop(
        &mut self,
        name_prefix: &str,
        body: &ProgramGraph,
        loop_carried_inputs: &[GraphValueId],
        step_inputs: &[GraphValueId],
        bounds: ControlBounds,
    ) -> Result<Vec<GraphValueId>, ProgramGraphError> {
        self.graph
            .add_bounded_loop(name_prefix, body, loop_carried_inputs, step_inputs, bounds)
    }

    /// Add a conditional branch.
    pub fn add_conditional_branch(
        &mut self,
        name_prefix: &str,
        condition: GraphValueId,
        then_graph: &ProgramGraph,
        else_graph: &ProgramGraph,
        inputs: &[GraphValueId],
    ) -> Result<Vec<GraphValueId>, ProgramGraphError> {
        self.graph
            .add_conditional_branch(name_prefix, condition, then_graph, else_graph, inputs)
    }

    /// Add an external effect barrier.
    pub fn add_effect_barrier(
        &mut self,
        name: impl Into<String>,
        effect: ExternalEffect,
        inputs: Vec<GraphInput>,
        outputs: Vec<GraphOutput>,
    ) -> Result<(GraphNodeId, Vec<GraphValueId>), ProgramGraphError> {
        self.graph.add_effect_barrier(name, effect, inputs, outputs)
    }

    /// Finish building and return the validated [`ProgramGraph`].
    pub fn finish(self) -> Result<ProgramGraph, ProgramGraphError> {
        self.graph
            .analyze()
            .map_err(|err| ProgramGraphError::Wire(err.to_string()))?;
        Ok(self.graph)
    }

    /// Build and return the [`ProgramGraph`].
    pub fn build(self) -> Result<ProgramGraph, ProgramGraphError> {
        self.finish()
    }
}
