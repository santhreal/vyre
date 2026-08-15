//! Precedence orders eligible backends; a reference oracle is not eligible.
//!
//! WHY: the sibling gate `reference_oracle_is_never_implicit.rs` proves the
//! refusal when an oracle is the only choice. That leaves the inverted case: an
//! oracle ranked ahead of a real device. If the oracle were merely last in the
//! precedence order rather than excluded, a rank edit or a new backend with a
//! worse rank would silently move host arithmetic to the front. This binary
//! ranks the oracle at 0, the best rank in the table, and the device at 500.
//!
//! Both factories succeed, so the only thing separating them is the
//! `reference_oracle` flag.

use std::collections::HashSet;
use std::sync::LazyLock;

use vyre_driver::backend::{
    acquire_preferred_dispatch_backend, BackendCapability, BackendPrecedence, BackendRegistration,
};
use vyre_driver::{BackendError, DispatchConfig, VyreBackend};
use vyre_foundation::ir::{OpId, Program};

const ORACLE_ID: &str = "fixture-ranked-oracle";
const DEVICE_ID: &str = "fixture-ranked-device";

struct FixtureBackend(&'static str);

impl vyre_driver::backend::private::Sealed for FixtureBackend {}

impl VyreBackend for FixtureBackend {
    fn id(&self) -> &'static str {
        self.0
    }

    fn dispatch(
        &self,
        _program: &Program,
        _inputs: &[Vec<u8>],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        Ok(Vec::new())
    }
}

fn acquire_oracle() -> Result<Box<dyn VyreBackend>, BackendError> {
    Ok(Box::new(FixtureBackend(ORACLE_ID)))
}

fn acquire_device() -> Result<Box<dyn VyreBackend>, BackendError> {
    Ok(Box::new(FixtureBackend(DEVICE_ID)))
}

fn no_supported_ops() -> &'static HashSet<OpId> {
    static OPS: LazyLock<HashSet<OpId>> = LazyLock::new(HashSet::new);
    &OPS
}

inventory::submit! {
    BackendRegistration {
        id: ORACLE_ID,
        target_id: vyre_foundation::operation::TargetId::expect_valid(ORACLE_ID),
        payload_format: None,
        reference_oracle: true,
        factory: acquire_oracle,
        supported_ops: no_supported_ops,
        semantic_operations: no_supported_ops,
        target_compiler: None,
        materializer: None,
    }
}

inventory::submit! {
    BackendRegistration {
        id: DEVICE_ID,
        target_id: vyre_foundation::operation::TargetId::expect_valid(DEVICE_ID),
        payload_format: None,
        reference_oracle: false,
        factory: acquire_device,
        supported_ops: no_supported_ops,
        semantic_operations: no_supported_ops,
        target_compiler: None,
        materializer: None,
    }
}

inventory::submit! {
    BackendCapability {
        id: ORACLE_ID,
        dispatches: true,
    }
}

inventory::submit! {
    BackendCapability {
        id: DEVICE_ID,
        dispatches: true,
    }
}

inventory::submit! {
    BackendPrecedence {
        id: ORACLE_ID,
        rank: 0,
    }
}

inventory::submit! {
    BackendPrecedence {
        id: DEVICE_ID,
        rank: 500,
    }
}

#[test]
fn a_best_ranked_reference_oracle_still_loses_to_a_worse_ranked_device() {
    let backend = acquire_preferred_dispatch_backend()
        .expect("Fix: preferred dispatch must select the linked non-oracle backend");
    assert_eq!(
        backend.id(),
        DEVICE_ID,
        "Fix: a reference oracle must never be the preferred dispatch target, whatever its \
         precedence rank. Precedence orders ELIGIBLE backends; an oracle is excluded before the \
         order is consulted."
    );
}
