//! Unforgeable, generation-bound, least-privilege capability handles (Row 119).

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::error::SecurityError;
use super::label::{Permission, TenantId};

/// Monotonic generation identifier to prevent use-after-free and reuse aliasing.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize,
)]
#[repr(transparent)]
pub struct GenerationId(pub u64);

impl GenerationId {
    /// Initial generation.
    pub const INITIAL: Self = Self(1);

    /// Create new generation.
    pub const fn new(val: u64) -> Self {
        Self(val)
    }

    /// Advance generation counter.
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// Return raw u64.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Unforgeable cryptographic capability handle bound to tenant, device, resource, and generation.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct UnforgeableCapability {
    /// Scoped tenant identifier.
    pub tenant_id: TenantId,
    /// Admitted device identifier.
    pub device_id: u64,
    /// Allocated resource or buffer identifier.
    pub resource_id: u64,
    /// Generation at time of issuance.
    pub generation: GenerationId,
    /// Explicit permissions granted to this handle.
    pub permissions: BTreeSet<Permission>,
    /// Cryptographic message authentication code (BLAKE3-MAC) proving validity.
    pub auth_mac: [u8; 16],
}

impl UnforgeableCapability {
    /// Return the tenant identifier.
    pub const fn tenant_id(&self) -> TenantId {
        self.tenant_id
    }

    /// Return the device identifier.
    pub const fn device_id(&self) -> u64 {
        self.device_id
    }

    /// Return the resource identifier.
    pub const fn resource_id(&self) -> u64 {
        self.resource_id
    }

    /// Return the generation counter.
    pub const fn generation(&self) -> GenerationId {
        self.generation
    }
}

/// Authenticator responsible for minting and verifying unforgeable capability handles.
#[derive(Clone, Debug)]
pub struct CapabilityAuthenticator {
    master_key: [u8; 32],
}

impl CapabilityAuthenticator {
    /// Create an authenticator with a 32-byte master cryptographic key.
    pub fn new(master_key: [u8; 32]) -> Self {
        Self { master_key }
    }

    /// Create an authenticator with a system-derived deterministic key.
    pub fn default_system() -> Self {
        let key_bytes = blake3::hash(b"vyre_default_system_capability_key")
            .as_bytes()
            .to_owned();
        Self {
            master_key: key_bytes,
        }
    }

    fn compute_mac(
        &self,
        tenant_id: TenantId,
        device_id: u64,
        resource_id: u64,
        generation: GenerationId,
        permissions: &BTreeSet<Permission>,
    ) -> [u8; 16] {
        let mut hasher = blake3::Hasher::new_keyed(&self.master_key);
        hasher.update(&tenant_id.as_u128().to_le_bytes());
        hasher.update(&device_id.to_le_bytes());
        hasher.update(&resource_id.to_le_bytes());
        hasher.update(&generation.as_u64().to_le_bytes());

        // Deterministic serialization of permissions
        for perm in permissions {
            let perm_str = serde_json::to_string(perm).unwrap_or_default();
            hasher.update(perm_str.as_bytes());
        }

        let output = hasher.finalize();
        let mut mac = [0u8; 16];
        mac.copy_from_slice(&output.as_bytes()[0..16]);
        mac
    }

    /// Mint a new unforgeable capability handle.
    pub fn issue_handle(
        &self,
        tenant_id: TenantId,
        device_id: u64,
        resource_id: u64,
        generation: GenerationId,
        permissions: BTreeSet<Permission>,
    ) -> UnforgeableCapability {
        let auth_mac =
            self.compute_mac(tenant_id, device_id, resource_id, generation, &permissions);
        UnforgeableCapability {
            tenant_id,
            device_id,
            resource_id,
            generation,
            permissions,
            auth_mac,
        }
    }

    /// Validate a presented capability handle against expected tenant, device, generation, and required permission.
    pub fn validate_handle(
        &self,
        handle: &UnforgeableCapability,
        expected_tenant: TenantId,
        expected_device: u64,
        current_generation: GenerationId,
        required_permission: &Permission,
    ) -> Result<(), SecurityError> {
        // 1. Verify cryptographic MAC
        let expected_mac = self.compute_mac(
            handle.tenant_id,
            handle.device_id,
            handle.resource_id,
            handle.generation,
            &handle.permissions,
        );

        if handle.auth_mac != expected_mac {
            return Err(SecurityError::ForgeryDetected);
        }

        // 2. Verify tenant scope
        if handle.tenant_id != expected_tenant && !expected_tenant.is_system() {
            return Err(SecurityError::TenantMismatch {
                expected: expected_tenant,
                found: handle.tenant_id,
            });
        }

        // 3. Verify device scope
        if handle.device_id != expected_device {
            return Err(SecurityError::DeviceMismatch {
                expected: expected_device,
                found: handle.device_id,
            });
        }

        // 4. Verify generation is not stale
        if handle.generation != current_generation {
            return Err(SecurityError::StaleGeneration {
                expected: current_generation.as_u64(),
                found: handle.generation.as_u64(),
            });
        }

        // 5. Verify required permission
        if !handle.permissions.contains(required_permission) {
            return Err(SecurityError::PermissionDenied {
                required: format!("{required_permission:?}"),
            });
        }

        Ok(())
    }
}
