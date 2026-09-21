//! Contract tests for Tenant Security & Capability Integration.

use std::collections::BTreeSet;
use vyre_foundation::security::{CapabilityAuthenticator, Permission, SecurityError};
use vyre_runtime::tenant::TenantRegistry;

#[test]
fn tenant_handle_issues_and_validates_unforgeable_capability() {
    let registry = TenantRegistry::new();
    let handle = registry
        .register("scanner-service")
        .expect("register tenant");
    let authenticator = CapabilityAuthenticator::default_system();
    let device_id = 1;
    let resource_id = 42;

    let mut permissions = BTreeSet::new();
    permissions.insert(Permission::SubmitWork);
    permissions.insert(Permission::ReadbackMemory);

    let capability = handle.issue_capability(&authenticator, device_id, resource_id, permissions);

    // Legitimate validation passes
    assert!(handle
        .validate_capability(
            &authenticator,
            &capability,
            device_id,
            &Permission::SubmitWork
        )
        .is_ok());

    // Validation for ungranted permission fails
    let perm_err = handle
        .validate_capability(
            &authenticator,
            &capability,
            device_id,
            &Permission::ManageQuotas,
        )
        .unwrap_err();
    assert!(matches!(perm_err, SecurityError::PermissionDenied { .. }));
}

#[test]
fn tenant_isolation_rejects_cross_tenant_capability_presentation() {
    let registry = TenantRegistry::new();
    let tenant_1 = registry.register("tenant-alpha").expect("register alpha");
    let tenant_2 = registry.register("tenant-beta").expect("register beta");
    let authenticator = CapabilityAuthenticator::default_system();
    let device_id = 1;
    let resource_id = 42;

    let mut permissions = BTreeSet::new();
    permissions.insert(Permission::AllocateBuffer);

    let cap_1 = tenant_1.issue_capability(&authenticator, device_id, resource_id, permissions);

    // Tenant 2 presenting Tenant 1's capability is rejected
    let mismatch_err = tenant_2
        .validate_capability(
            &authenticator,
            &cap_1,
            device_id,
            &Permission::AllocateBuffer,
        )
        .unwrap_err();
    assert!(matches!(mismatch_err, SecurityError::TenantMismatch { .. }));
}
