//! vyre-foundation  -  substrate-neutral compiler foundation.
//!
//! Defines the vyre IR (`Expr`, `Node`, `Program`), the type system, the
//! memory model, the wire format, visitor traits, and extension resolvers.
//! Every other vyre crate depends on this one; this crate depends only on
//! `vyre-spec`, `vyre-macros`, and lightweight third-party data crates.
//! It never knows about concrete driver APIs, a dialect, or a backend.

// Foundation owns the IR arena (`ir_inner::model::arena`), which uses two
// `unsafe` blocks to extend bumpalo lifetimes inside a single arena owner.
// Every other unsafe usage is forbidden by `check_lib_rs_headers.sh`.
#![allow(unsafe_code)]
#![allow(
    clippy::duplicate_mod,
    clippy::too_many_arguments,
    clippy::double_must_use,
    clippy::module_inception,
    clippy::should_implement_trait,
    clippy::type_complexity
)]

extern crate self as vyre;

/// Shared structured diagnostic protocol.
pub mod diagnostics;
/// Shared floating-point parity policy and typed buffer comparison.
pub mod fp_parity;
/// Canonical semantic operation registration and target facet views.
pub mod operation;

pub mod ir {
    //! The vyre intermediate representation.
    /// Backend-neutral literal evaluation for optimizer passes and lowerings.
    pub mod eval {
        // Audit cleanup A16 (2026-04-30): replaced `pub use crate::ir_eval::*`
        // wildcard with explicit named re-exports per
        // organization_contracts::foundation_wildcard_pub_reexports_are_baselined.
        pub use crate::ir_eval::{
            fold_binary_literal, fold_cast_literal, fold_fma_literal, fold_literal_tree,
            fold_unary_literal,
        };
    }
    pub use crate::ir_inner::model;
    pub use crate::ir_inner::model::arena::{ArenaProgram, ExprArena, ExprRef};
    pub use crate::ir_inner::model::expr::{Expr, ExprNode, Ident};
    pub use crate::ir_inner::model::node::{Node, NodeExtension};
    pub use crate::ir_inner::model::node_kind::{
        EvalError, InterpCtx, NodeId, NodeStorage, OpId, RegionId, Value, VarId,
    };
    pub use crate::ir_inner::model::program::{
        BufferDecl, CacheLocality, LinearType, MemoryHints, MemoryKind, Program, ShapePredicate,
        NORMALIZED_PROGRAM_CACHE_DIGEST_VERSION,
    };
    pub use crate::ir_inner::model::program_graph::{
        GraphInput, GraphNodeId, GraphOutput, GraphValueId, LivenessInterval, ProgramGraph,
        ProgramGraphError, ProgramGraphNode, ProgramGraphValue, ShapeDim, ValueContract,
        ValueLifetime,
    };
    /// Per-Node-variant bit-position constants for `ProgramStats::node_kinds_present`.
    /// Compose with `ProgramStats::has_any_node_kind` for O(1) `analyze_impl` gates.
    pub mod stats {
        pub use crate::ir_inner::model::program::{
            NODE_KIND_ALL_GATHER, NODE_KIND_ALL_REDUCE, NODE_KIND_ASSIGN, NODE_KIND_ASYNC_LOAD,
            NODE_KIND_ASYNC_STORE, NODE_KIND_ASYNC_WAIT, NODE_KIND_BARRIER, NODE_KIND_BLOCK,
            NODE_KIND_BROADCAST, NODE_KIND_EXPRESSION_BEARING_MASK, NODE_KIND_IF,
            NODE_KIND_INDIRECT_DISPATCH, NODE_KIND_LET, NODE_KIND_LOOP, NODE_KIND_OPAQUE,
            NODE_KIND_REDUCE_SCATTER, NODE_KIND_REGION, NODE_KIND_RESUME, NODE_KIND_RETURN,
            NODE_KIND_STORE, NODE_KIND_TRAP,
        };
    }
    pub use crate::ir_inner::model::program::ProgramStats;
    pub use crate::ir_inner::model::types::{
        AtomicOp, BinOp, BufferAccess, CollectiveOp, CommGroup, Convention, DataType, OpSignature,
        SubgroupReduceOp, UnOp,
    };
    pub use crate::memory_model;
    pub use crate::memory_model::MemoryOrdering;
    pub use crate::optimizer::passes::fusion_cse::{cse, dce};
    pub use crate::serial::text;
    pub use crate::transform::inline::{
        inline_calls, inline_calls_with_mode, inline_calls_with_resolver, inline_composite_calls,
        OpResolver, UnresolvedCalls,
    };
    pub use crate::validate::depth::{
        LimitState, DEFAULT_MAX_CALL_DEPTH, DEFAULT_MAX_NESTING_DEPTH, DEFAULT_MAX_NODE_COUNT,
    };
    pub use crate::validate::validate::validate;
    pub use crate::validate::validation_error::ValidationError;
}

/// CPU reference registration contract.
pub mod cpu_op;
/// Backend-neutral literal evaluation used by IR optimization and lowering.
pub(crate) mod ir_eval;
/// Domain-neutral byte-range result types.
pub mod match_result;
/// Substrate-neutral memory ordering.
pub mod memory_model;
/// Optimizer performance counters.
pub mod perf;
/// Program capability analysis.
pub mod program_caps;

/// Operation schema and opaque IR extension surfaces.
pub mod dispatch;

/// Algebraic-laws surface (algebraic_law_registry, composition).
pub mod algebra;

/// Static-analysis surface (graph_view).
pub mod analysis;

/// Substrate-neutral allocation reservation arithmetic shared by hot paths.
pub mod allocation;

pub use algebra::algebraic_law_registry;
pub use algebra::algebraic_law_registry::{
    has_law, is_associative, is_commutative, laws_for_op, AlgebraicLaw, AlgebraicLawRegistration,
};
pub use analysis::graph_view;
pub use dispatch::dialect_lookup;
pub use memory_model::MemoryOrdering;

/// Endian-fixed encode/decode helpers for `Expr::Opaque` / `Node::Opaque` payloads.
pub mod opaque_payload;

/// Packed AST (VAST) wire layout + host-side tree walks (`docs/ARCHITECTURE.md`).
pub mod vast;

pub use analysis::graph_view::{
    from_graph, to_graph, DataEdge, DataflowKind, EdgeKind, GraphNode, GraphValidateError,
    NodeGraph,
};
pub use dispatch::dialect_lookup::{AttrSchema, AttrType, Signature, TypedParam};

// The generated AST is owner-local; public consumers use `pub mod ir`.
mod ir_inner {
    pub mod model;
}
pub use algebra::composition;
pub use dispatch::extension;

/// Deterministic field framing for content-addressed hashes.
pub mod hashing;
/// Legacy lower helpers (transition surface pending driver-tier extraction).
pub mod lower;
/// Pass-orchestration optimizer framework.
pub mod optimizer;
/// Binary wire format + canonical text serialization.
pub mod serial;
/// IR → IR passes: inline, cse, dce, parallelism, compiler primitives.
pub mod transform;
/// Structural + semantic validation of vyre `Program`s.
pub mod validate;
/// Visitor traits + blanket adapters routing Expr/Node variants.
pub mod visit;

/// Self-substrate primitives that the optimizer + scheduler call into.
/// Moved in-tree from vyre-libs to break a cross-workspace dep cycle.
pub mod pass_substrate;

/// Program → substrate-neutral execution planning for fusion, readback,
/// provenance, autotune, and accuracy guard decisions.
pub mod execution_plan;

/// Foundation-owned IR and Program wire failures.
pub mod error;
pub use error::{IrError, IrResult};

/// Test utilities shared across optimizer and transform test suites.
/// `pub(crate)` because they are an internal contract  -  no consumer
/// outside vyre-foundation should depend on these helpers.
#[cfg(test)]
pub(crate) mod test_util;
