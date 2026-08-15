//! C semantic analysis passes.
//!
//! Production semantic analysis is built as dispatchable IR programs. CPU
//! reference helpers remain available only as explicit oracle surfaces for
//! conformance and witness generation.

/// Identifier interning IR fragments.
pub mod intern;
/// Host-side lazy scope/name resolution cache.
pub(crate) mod lazy_scope;
/// Declaration lookup IR fragments.
pub mod lookup;
mod predicates;
/// Registered C semantic-analysis programs.
pub(crate) mod registry;
mod scan;
/// Scope-walk IR fragments.
pub mod walk;

pub use lazy_scope::{DeclKind, LazyScopeTable, ScopeFrameId};
#[cfg(any(test, feature = "cpu-parity"))]
pub use registry::reference_scope_tree;
pub use registry::{
    c_sema_scope, c_sema_scope_packed_haystack, c_sema_scope_symbols_packed_haystack,
};
