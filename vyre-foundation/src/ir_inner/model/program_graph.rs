//! Typed connections between reusable [`Program`](crate::ir::Program) values.
//!
//! `ProgramGraph` is composition metadata over existing Vyre IR. It is not a
//! second neural IR: every executable node remains an ordinary `Program`.

mod builder;
mod graph;
mod types;
mod validate;

pub use builder::ProgramGraphBuilder;
pub use graph::ProgramGraph;
pub use types::{
    ControlBounds, ExternalEffect, GraphInput, GraphNodeId, GraphOutput, GraphValueId,
    LivenessInterval, ProgramGraphError, ProgramGraphNode, ProgramGraphSharingMetrics,
    ProgramGraphTemplate, ProgramGraphValue, ShapeDim, ValueContract, ValueLifetime,
};
