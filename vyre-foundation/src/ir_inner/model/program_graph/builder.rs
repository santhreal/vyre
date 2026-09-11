//! Transactional builder for [`ProgramGraph`](super::ProgramGraph) compositions.

use std::ops::{Deref, DerefMut};

use super::contracts::{GraphValueId, ProgramGraphError, ShapeDim, ValueContract, ValueLifetime};
use super::graph::ProgramGraph;
use crate::ir_inner::model::op_signature::{BufferAccess, DataType};

/// Transactional, fluent domain-neutral builder for whole [`ProgramGraph`] compositions.
///
/// The builder owns the external-value contracts and the validation that
/// [`Self::build`] runs. Node and subgraph composition is [`ProgramGraph`]'s
/// and is reached through [`Deref`], so each signature is declared once.
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

    /// Validate the composition and return the [`ProgramGraph`].
    pub fn build(self) -> Result<ProgramGraph, ProgramGraphError> {
        self.graph
            .analyze()
            .map_err(|err| ProgramGraphError::Wire(err.to_string()))?;
        Ok(self.graph)
    }
}

impl Deref for ProgramGraphBuilder {
    type Target = ProgramGraph;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for ProgramGraphBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}
