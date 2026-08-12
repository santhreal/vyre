//! Duplicate backend-provider startup rejection contract.

use std::collections::HashSet;
use std::sync::LazyLock;

use vyre_driver::{BackendError, BackendRegistration, VyreBackend};
use vyre_foundation::ir::OpId;
use vyre_foundation::operation::TargetId;

const TARGET_ID: TargetId = TargetId::expect_valid("duplicate-provider-target");

fn unavailable_backend() -> Result<Box<dyn VyreBackend>, BackendError> {
    Err(BackendError::new(
        "duplicate-provider fixture cannot dispatch. Fix: use it only for registry validation.",
    ))
}

fn no_operations() -> &'static HashSet<OpId> {
    static OPERATIONS: LazyLock<HashSet<OpId>> = LazyLock::new(HashSet::new);
    &OPERATIONS
}

inventory::submit! {
    BackendRegistration {
        id: "duplicate-provider",
        target_id: TARGET_ID,
        payload_format: None,
        reference_oracle: false,
        factory: unavailable_backend,
        supported_ops: no_operations,
        semantic_operations: no_operations,
        target_compiler: None,
        materializer: None,
    }
}

inventory::submit! {
    BackendRegistration {
        id: "duplicate-provider",
        target_id: TARGET_ID,
        payload_format: None,
        reference_oracle: false,
        factory: unavailable_backend,
        supported_ops: no_operations,
        semantic_operations: no_operations,
        target_compiler: None,
        materializer: None,
    }
}

/// WHY: provider identity must never depend on link-time inventory order.
#[test]
fn duplicate_backend_provider_returns_startup_error() {
    let error = match vyre_driver::backend::registered_backends() {
        Ok(_) => panic!("duplicate backend providers must reject registry startup"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("duplicate backend registration `duplicate-provider`")
            && message.contains("keep one concrete provider"),
        "duplicate-provider error must identify the conflict and repair: {message}"
    );
}
