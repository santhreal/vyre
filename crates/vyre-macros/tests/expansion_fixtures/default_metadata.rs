//! The metadata `#[vyre_pass]` emits when every optional argument is omitted.
//!
//! This is one contract, not one per test target: the defaults are what a pass
//! author gets for free, so a change to any of them is a change to the macro's
//! published behaviour and has to be visible in exactly one place. Only the
//! suites that assert the defaults declare this module, so the assertion has a
//! caller in every binary that compiles it.

use crate::expansion_fixtures::optimizer;

pub(crate) fn assert_default_metadata(metadata: &optimizer::PassMetadata, name: &str) {
    assert_eq!(metadata.name, name);
    assert_eq!(metadata.requires, &[] as &[&str]);
    assert_eq!(metadata.invalidates, &[] as &[&str]);
    assert_eq!(metadata.phase, optimizer::PassPhase::Unclassified);
    assert_eq!(
        metadata.boundary_class,
        optimizer::PassBoundaryClass::Unknown
    );
    assert_eq!(metadata.requires_caps, &[] as &[&str]);
    assert!(metadata.preserves_abi);
    assert_eq!(
        metadata.cost_model_family,
        optimizer::CostModelFamily::Unknown
    );
}
