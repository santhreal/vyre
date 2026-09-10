//! Workgroup simulation: the parity engine's model of invocation coordination.
//!
//! GPU backends must reproduce the exact barrier synchronization, shared-memory
//! layout, and invocation-ID arithmetic that this module defines. The conform gate
//! compares GPU dispatch output against this deterministic CPU simulation; any
//! divergence in control flow uniformity or workgroup memory semantics is a bug.
//!
//! Invocation state and its lexical scopes are owned by the canonical evaluator
//! in `execution::hashmap`. This module holds only what a lane's identity and
//! its continuation stack are, plus the shared-memory ceiling, which the
//! evaluator and its memory allocator both read.

use vyre_foundation::ir::Node;

/// Maximum per-workgroup shared memory the reference interpreter will allocate.
pub const MAX_WORKGROUP_BYTES: usize = 64 * 1024 * 1024;

/// Identity of one compute invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvocationIds {
    /// Global invocation id.
    pub global: [u32; 3],
    /// Workgroup id.
    pub workgroup: [u32; 3],
    /// Local invocation id.
    pub local: [u32; 3],
}

impl InvocationIds {
    /// Zero-valued invocation ids for examples and unit tests.
    pub const ZERO: Self = Self {
        global: [0, 0, 0],
        workgroup: [0, 0, 0],
        local: [0, 0, 0],
    };
}

/// Interpreter continuation stack.
#[non_exhaustive]
pub enum Frame<'a> {
    /// Sequence of nodes.
    Nodes {
        /// Nodes being executed.
        nodes: &'a [Node],
        /// Next node index.
        index: usize,
        /// Whether completion pops a lexical scope.
        scoped: bool,
    },
    /// Bounded `u32` loop.
    Loop {
        /// Loop variable name.
        var: &'a str,
        /// Next induction value.
        next: u32,
        /// Exclusive upper bound.
        to: u32,
        /// Loop body.
        body: &'a [Node],
    },
}
