//! One owner for the backend fixture the reference-oracle gates register.
//!
//! WHY: both gates need a dispatch-capable backend that succeeds, differing
//! only in the `reference_oracle` flag and the precedence rank. Written out per
//! suite, that is a `VyreBackend` impl plus three `inventory::submit!` blocks
//! copied per backend, and the copy is what rots: a new required field on
//! `BackendRegistration` has to be added once per copy, and a suite whose copy
//! drifts stops registering the thing the gate is about.
//!
//! The registration blocks are a macro rather than a function because
//! `inventory::submit!` is an item and each backend needs its own.

use std::collections::HashSet;
use std::sync::LazyLock;

use vyre_driver::{BackendError, DispatchConfig, VyreBackend};
use vyre_foundation::ir::{OpId, Program};

/// A backend that dispatches successfully and returns nothing.
///
/// Returning no outputs is deliberate: every gate that registers this asks which
/// backend was selected, never what it computed, and a fixture that produced
/// values would be host arithmetic in a driver test.
pub(crate) struct FixtureBackend(pub(crate) &'static str);

impl vyre_driver::sealed::Sealed for FixtureBackend {}

impl VyreBackend for FixtureBackend {
    fn id(&self) -> &'static str {
        self.0
    }

    fn dispatch_borrowed(
        &self,
        _program: &Program,
        _inputs: &[&[u8]],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        Ok(Vec::new())
    }
}

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
macro_rules! register_dispatchable_backend {
    (id: $id:expr, oracle: $oracle:expr, rank: $rank:expr, factory: $factory:path $(,)?) => {
        inventory::submit! {
            vyre_driver::BackendRegistration {
                id: $id,
                target_id: vyre_foundation::operation::TargetId::expect_valid($id),
                payload_format: None,
                reference_oracle: $oracle,
                factory: $factory,
                supported_ops: $crate::common::no_supported_ops,
                semantic_operations: $crate::common::no_supported_ops,
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
