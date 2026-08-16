//! What a topological sort refuses, over both input shapes.

/// Errors from topological sorting.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToposortError {
    /// The input graph contains a cycle  -  returned with the first
    /// node id that participates in the cycle, for diagnostic use.
    Cycle {
        /// One node id on the cycle. Callers can walk the adjacency
        /// list from here to enumerate the full cycle.
        node: u32,
    },
    /// An edge references a node id not present in `node_count`.
    UnknownNode {
        /// Offending edge index.
        edge: usize,
        /// The out-of-range node id that tripped the check.
        node: u32,
    },
    /// A node's dependency count exceeded the `u32` counter used by the
    /// compact scheduler representation.
    IndegreeOverflow {
        /// Node whose dependency count overflowed.
        node: u32,
    },
    /// Kahn's invariant was violated after input validation, indicating
    /// inconsistent derived adjacency state.
    InconsistentState {
        /// Actionable diagnostic.
        message: String,
    },
}

/// Errors from CSR topological-sort shape or order validation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToposortCsrError {
    /// CSR row pointers or targets are malformed for the declared node count.
    BadCsr {
        /// Actionable diagnostic.
        message: String,
    },
    /// The supplied topological order is not a full valid permutation.
    BadOrder {
        /// Actionable diagnostic.
        message: String,
    },
}

pub(super) fn toposort_csr_allocation(message: String) -> ToposortCsrError {
    ToposortCsrError::BadCsr { message }
}

pub(super) fn toposort_allocation(message: String) -> ToposortError {
    ToposortError::InconsistentState { message }
}
