//! Failure-oriented tests for routing and registered backend contracts.

use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::LazyLock;

use vyre_driver::{BackendError, BackendRegistration, RoutingTable, SortBackend, VyreBackend};
use vyre_foundation::ir::OpId;

#[test]
fn routing_table_distribution_returns_none_for_unknown_callsite() {
    let table = RoutingTable::default();
    assert!(
        table.distribution("no.such.callsite").is_none(),
        "Fix: RoutingTable::distribution must return None for unseen call sites"
    );
}

#[test]
fn routing_table_empty_input_distribution() {
    let table = RoutingTable::default();
    let backend = table
        .observe_sort_u32(Cow::Borrowed("empty.sort"), &[])
        .expect("Fix: empty input must not panic");
    assert_eq!(backend, SortBackend::InsertionSort);
    assert!(
        table.distribution("empty.sort").is_some(),
        "Fix: distribution must be recorded for empty input"
    );
}

fn unavailable_backend() -> Result<Box<dyn VyreBackend>, BackendError> {
    Err(BackendError::UnsupportedFeature {
        name: "test backend".into(),
        backend: "test".into(),
    })
}

fn no_supported_ops() -> &'static HashSet<OpId> {
    static OPS: LazyLock<HashSet<OpId>> = LazyLock::new(HashSet::new);
    &OPS
}

/// WHY: an unimplemented native facet must fail closed instead of receiving a raw Program.
#[test]
fn backend_registration_without_artifact_facets_fails_explicitly() {
    let registration = BackendRegistration {
        id: "test",
        target_id: vyre_foundation::operation::TargetId::expect_valid("test"),
        payload_format: None,
        reference_oracle: false,
        factory: unavailable_backend,
        supported_ops: no_supported_ops,
        semantic_operations: no_supported_ops,
        target_compiler: None,
        materializer: None,
    };
    let compiler = registration
        .target_compiler()
        .err()
        .expect("missing target compiler must fail");
    let materializer = registration
        .materializer()
        .err()
        .expect("missing materializer must fail");
    assert!(compiler.to_string().contains("registered target compiler"));
    assert!(materializer
        .to_string()
        .contains("registered artifact materializer"));
}
