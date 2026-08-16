//! Core IR data model.
//!
//! This module defines the pure Rust data structures that make up a vyre
//! program: expressions (`Expr`), statements (`Node`), buffer declarations
//! (`BufferDecl`), and the root `Program` container. These types are
//! serializable, validateable, and backend-agnostic.

/// Expression nodes that produce values.
///
/// `Expr` covers literals, variable references, arithmetic, comparisons,
/// buffer loads, and operation calls. Every expression has a statically
/// known type.
pub(crate) mod expr;

pub(crate) mod generated;
/// Statement nodes that execute effects.
///
/// `Node` covers variable declarations, assignments, control flow
/// (if/else, loops), and buffer stores. A `Program` is essentially a
/// sequence of nodes.
pub(crate) mod node;

/// Open node kind trait and built-in node structs.
pub(crate) mod node_kind;

/// Program structure and metadata.
///
/// `Program` is the top-level IR container. It holds buffer declarations,
/// the entry node list, and optional optimization hints.
pub(crate) mod program;

/// Typed topology over connected reusable Programs.
pub(crate) mod program_graph;
/// Whole-composition validation, liveness, and allocation analysis.
pub(crate) mod program_graph_analysis;
/// Versioned content identity for connected Program compositions.
pub(crate) mod program_graph_identity;
mod program_graph_wire;

/// The frozen IR vocabulary re-exported from `vyre-spec`.
pub(crate) mod op_signature;
