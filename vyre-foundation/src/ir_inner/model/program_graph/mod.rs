//! Typed connections between reusable [`Program`](crate::ir::Program) values.
//!
//! `ProgramGraph` is composition metadata over existing Vyre IR. It is not a
//! second neural IR: every executable node remains an ordinary `Program`.

mod builder;
mod contracts;
mod graph;
mod validate;

pub use builder::ProgramGraphBuilder;
pub use contracts::{
    ControlBounds, ExternalEffect, GraphInput, GraphNodeId, GraphOutput, GraphValueId,
    LivenessInterval, ProgramGraphError, ProgramGraphNode, ProgramGraphSharingMetrics,
    ProgramGraphTemplate, ProgramGraphValue, ShapeDim, ValueContract, ValueLifetime,
};
pub use graph::ProgramGraph;
