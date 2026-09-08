//! Contract tests for Security, Tenant Labels & Capability Handles (Row 119).

use std::collections::BTreeSet;
use vyre_foundation::security::*;

#[test]
fn capability_authenticator_validates_legitimate_handle() {
    let authenticator = CapabilityAuthenticator::default_system();
    let tenant_id = TenantId::new(1001);
    let device_id = 42;
    let resource_id = 1;
    let generation = GenerationId::INITIAL;

    let mut permissions = BTreeSet::new();
    permissions.insert(Permission::CompileProgram);
    permissions.insert(Permission::SubmitWork);

    let handle = authenticator.issue_handle(tenant_id, device_id, resource_id, generation, permissions);

    // Legitimate validation must succeed
    assert!(authenticator
        .validate_handle(&handle, tenant_id, device_id, generation, &Permission::SubmitWork)
        .is_ok());
}

#[test]
fn handle_forgery_is_detected_and_rejected() {
    let authenticator = CapabilityAuthenticator::default_system();
    let tenant_id = TenantId::new(1001);
    let device_id = 42;
    let resource_id = 1;
    let generation = GenerationId::INITIAL;

    let mut permissions = BTreeSet::new();
    permissions.insert(Permission::CompileProgram);

    let mut handle = authenticator.issue_handle(tenant_id, device_id, resource_id, generation, permissions);

    // Adversarial modification: forge extra permission
    handle.permissions.insert(Permission::SubmitWork);

    let result = authenticator.validate_handle(
        &handle,
        tenant_id,
        device_id,
        generation,
        &Permission::SubmitWork,
    );
    assert_eq!(result, Err(SecurityError::ForgeryDetected));
}

#[test]
fn stale_generation_is_rejected_on_resource_reuse() {
    let authenticator = CapabilityAuthenticator::default_system();
    let tenant_id = TenantId::new(1001);
    let device_id = 42;
    let resource_id = 1;
    let gen1 = GenerationId::INITIAL;
    let gen2 = gen1.next();

    let mut permissions = BTreeSet::new();
    permissions.insert(Permission::ReadbackMemory);

    let handle_gen1 = authenticator.issue_handle(tenant_id, device_id, resource_id, gen1, permissions);

    // Device advanced generation to gen2, handle_gen1 presented
    let result = authenticator.validate_handle(
        &handle_gen1,
        tenant_id,
        device_id,
        gen2,
        &Permission::ReadbackMemory,
    );
    assert_eq!(
        result,
        Err(SecurityError::StaleGeneration {
            expected: 2,
            found: 1
        })
    );
}

#[test]
fn cross_tenant_presentation_is_rejected() {
    let authenticator = CapabilityAuthenticator::default_system();
    let tenant_a = TenantId::new(1001);
    let tenant_b = TenantId::new(2002);
    let device_id = 42;
    let resource_id = 1;
    let generation = GenerationId::INITIAL;

    let mut permissions = BTreeSet::new();
    permissions.insert(Permission::AllocateBuffer);

    let handle_a = authenticator.issue_handle(tenant_a, device_id, resource_id, generation, permissions);

    // Tenant B attempts to use Tenant A's handle
    let result = authenticator.validate_handle(
        &handle_a,
        tenant_b,
        device_id,
        generation,
        &Permission::AllocateBuffer,
    );
    assert_eq!(
        result,
        Err(SecurityError::TenantMismatch {
            expected: tenant_b,
            found: tenant_a
        })
    );
}

#[test]
fn confidentiality_lattice_flow_rules() {
    assert!(ConfidentialityLevel::Public.can_flow_to(ConfidentialityLevel::Public));
    assert!(ConfidentialityLevel::Public.can_flow_to(ConfidentialityLevel::Internal));
    assert!(ConfidentialityLevel::Public.can_flow_to(ConfidentialityLevel::Confidential));
    assert!(ConfidentialityLevel::Public.can_flow_to(ConfidentialityLevel::Secret));
    assert!(ConfidentialityLevel::Public.can_flow_to(ConfidentialityLevel::Isolated));

    // Secret data cannot flow down to Public or Internal
    assert!(!ConfidentialityLevel::Secret.can_flow_to(ConfidentialityLevel::Public));
    assert!(!ConfidentialityLevel::Secret.can_flow_to(ConfidentialityLevel::Internal));
    assert!(!ConfidentialityLevel::Secret.can_flow_to(ConfidentialityLevel::Confidential));
    assert!(ConfidentialityLevel::Secret.can_flow_to(ConfidentialityLevel::Secret));
    assert!(ConfidentialityLevel::Secret.can_flow_to(ConfidentialityLevel::Isolated));
}

#[test]
fn memory_sanitization_zeroes_on_drop() {
    let mut buf = SanitizedBuffer::zeroed(128);
    buf.as_mut_slice().fill(0xAA);
    assert_eq!(buf.as_slice()[0], 0xAA);

    buf.zeroize();
    assert_eq!(buf.as_slice()[0], 0x00);
    assert!(buf.as_slice().iter().all(|&b| b == 0));
}

#[test]
fn tenant_cache_namespace_partitions_keys() {
    let tenant_1 = TenantId::new(101);
    let tenant_2 = TenantId::new(102);
    let raw_key = b"kernel_sha256_hash_12345";

    let key_t1 = TenantCacheNamespace::derive_key(tenant_1, ConfidentialityLevel::Confidential, raw_key);
    let key_t2 = TenantCacheNamespace::derive_key(tenant_2, ConfidentialityLevel::Confidential, raw_key);

    assert_ne!(key_t1, key_t2);
}

#[test]
fn compilation_quota_enforcer_bounds_untrusted_workloads() {
    let quota = CompilationQuota {
        max_nodes: 10,
        max_depth: 3,
        max_compile_time_ns: 1_000_000,
        max_memory_bytes: 1024,
    };

    let mut enforcer = CompilationBudgetEnforcer::new(quota, 0);

    // Increment nodes up to limit
    for _ in 0..10 {
        assert!(enforcer.increment_node_count().is_ok());
    }
    // 11th node must exceed quota
    let err = enforcer.increment_node_count().unwrap_err();
    assert!(matches!(err, SecurityError::QuotaExceeded { ref resource, .. } if resource == "ir_nodes"));

    // Exceed depth
    assert!(enforcer.enter_scope().is_ok()); // 1
    assert!(enforcer.enter_scope().is_ok()); // 2
    assert!(enforcer.enter_scope().is_ok()); // 3
    let depth_err = enforcer.enter_scope().unwrap_err(); // 4 > 3
    assert!(matches!(depth_err, SecurityError::QuotaExceeded { ref resource, .. } if resource == "ast_depth"));
}

#[test]
fn diagnostic_redaction_sanitizes_addresses_and_secrets() {
    let raw = "Error at 0x7ffd98b04a20 in node evaluating secret:api_token_42 and normal_var";
    let sanitized = RedactedDiagnostic::sanitize_text(raw);
    assert_eq!(
        sanitized,
        "Error at [REDACTED_ADDR] in node evaluating [REDACTED_SECRET] and normal_var"
    );
}
