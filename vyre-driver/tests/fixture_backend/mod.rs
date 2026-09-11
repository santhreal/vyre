//! One owner for the backend fixture the reference-oracle gates register.
//!
//! WHY: both gates need a dispatch-capable backend that succeeds, differing
//! only in the `reference_oracle` flag and the precedence rank. Written out per
//! suite, that is three `inventory::submit!` blocks copied per backend, and the
//! copy is what rots: a new required field on `BackendRegistration` has to be
//! added once per copy, and a suite whose copy drifts stops registering the
//! thing the gate is about.
//!
//! The registration blocks are a macro rather than a function because
//! `inventory::submit!` is an item and each backend needs its own.

use std::collections::HashSet;
use std::sync::LazyLock;

use vyre_foundation::ir::OpId;

/// The dispatch-capable double both gates register, under this crate's name
/// for it. The impl has one owner in `vyre-test-support` because the
/// resident-sequence unit tests need the same one.
pub(crate) use vyre_test_support::backend_doubles::NoOutputBackend as FixtureBackend;

/// A backend that claims no operation, so selection turns on eligibility alone.
pub(crate) fn no_supported_ops() -> &'static HashSet<OpId> {
    static OPS: LazyLock<HashSet<OpId>> = LazyLock::new(HashSet::new);
    &OPS
}

/// Register one dispatch-capable backend under `$id` at precedence `$rank`.
///
/// `$oracle` is the `reference_oracle` flag, which is the whole subject of the
/// gates that use this: it is the only field that may differ between a backend
/// preferred dispatch selects and one it refuses.
///
/// Exported so a test module reaches it by path. Every test file here is a
/// module of a harness binary, and textual macro scope between siblings depends
/// on declaration order; a path does not.
#[macro_export]
macro_rules! register_dispatchable_backend {
    (id: $id:expr, oracle: $oracle:expr, rank: $rank:expr, factory: $factory:path $(,)?) => {
        inventory::submit! {
            vyre_driver::BackendRegistration {
                id: $id,
                target_id: vyre_foundation::operation::TargetId::expect_valid($id),
                payload_format: None,
                reference_oracle: $oracle,
                factory: $factory,
                supported_ops: $crate::fixture_backend::no_supported_ops,
                semantic_operations: $crate::fixture_backend::no_supported_ops,
                target_compiler: None,
                materializer: None,
            }
        }

        inventory::submit! {
            vyre_driver::BackendCapability { id: $id, dispatches: true }
        }

        inventory::submit! {
            vyre_driver::BackendPrecedence { id: $id, rank: $rank }
        }
    };
}
