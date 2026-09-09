//! Explicit tenant, authority, confidentiality, and retention labels (Row 119).

use core::fmt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Strongly-typed 128-bit tenant identifier.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize,
)]
pub struct TenantId(pub u128);

impl TenantId {
    /// System root tenant.
    pub const SYSTEM: Self = Self(0);
    /// Anonymous/unauthenticated guest tenant.
    pub const ANONYMOUS: Self = Self(u128::MAX);

    /// Create tenant identifier from raw u128.
    pub const fn new(id: u128) -> Self {
        Self(id)
    }

    /// Return raw u128.
    pub const fn as_u128(self) -> u128 {
        self.0
    }

    /// Return true if this is the privileged system tenant.
    pub const fn is_system(self) -> bool {
        self.0 == Self::SYSTEM.0
    }
}

impl fmt::Display for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_system() {
            write!(f, "tenant:system")
        } else if *self == Self::ANONYMOUS {
            write!(f, "tenant:anonymous")
        } else {
            write!(f, "tenant:{:032x}", self.0)
        }
    }
}

/// Security roles assigned to an authority principal.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SecurityRole {
    /// Root compiler administrator.
    Administrator,
    /// Automated compiler and lowering service.
    CompilerService,
    /// Standard application tenant operator.
    TenantOperator,
    /// Read-only observer for telemetry.
    ReadonlyObserver,
    /// Unauthenticated guest with minimal capabilities.
    Guest,
}

/// Granular capability permissions.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    /// Permission to submit IR programs for compilation.
    CompileProgram,
    /// Permission to allocate device and host memory.
    AllocateBuffer,
    /// Permission to submit execution commands to hardware queues.
    SubmitWork,
    /// Permission to read back device buffer contents to host memory.
    ReadbackMemory,
    /// Permission to invoke external or dialect extensions.
    AccessExtension(String),
    /// Permission to query performance counters and telemetry.
    QueryTelemetry,
    /// Permission to modify tenant quota allocations.
    ManageQuotas,
}

/// Explicit authority label identifying a tenant, principal, roles, and permissions.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct AuthorityLabel {
    /// Owning tenant identifier.
    pub tenant_id: TenantId,
    /// Named principal or service identity.
    pub principal: String,
    /// Assigned security roles.
    pub roles: BTreeSet<SecurityRole>,
    /// Explicit permissions granted.
    pub permissions: BTreeSet<Permission>,
}

impl AuthorityLabel {
    /// Construct a system root authority label.
    pub fn system() -> Self {
        let mut permissions = BTreeSet::new();
        permissions.insert(Permission::CompileProgram);
        permissions.insert(Permission::AllocateBuffer);
        permissions.insert(Permission::SubmitWork);
        permissions.insert(Permission::ReadbackMemory);
        permissions.insert(Permission::QueryTelemetry);
        permissions.insert(Permission::ManageQuotas);

        let mut roles = BTreeSet::new();
        roles.insert(SecurityRole::Administrator);

        Self {
            tenant_id: TenantId::SYSTEM,
            principal: "system::root".to_string(),
            roles,
            permissions,
        }
    }

    /// Construct a standard tenant operator authority label.
    pub fn for_tenant(tenant_id: TenantId, principal: impl Into<String>) -> Self {
        let mut permissions = BTreeSet::new();
        permissions.insert(Permission::CompileProgram);
        permissions.insert(Permission::AllocateBuffer);
        permissions.insert(Permission::SubmitWork);
        permissions.insert(Permission::ReadbackMemory);
        permissions.insert(Permission::QueryTelemetry);

        let mut roles = BTreeSet::new();
        roles.insert(SecurityRole::TenantOperator);

        Self {
            tenant_id,
            principal: principal.into(),
            roles,
            permissions,
        }
    }

    /// Check whether a given permission is granted.
    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.roles.contains(&SecurityRole::Administrator) || self.permissions.contains(permission)
    }
}

/// Multi-level security confidentiality lattice.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ConfidentialityLevel {
    /// Unclassified public data.
    #[default]
    Public = 0,
    /// Internal application data.
    Internal = 1,
    /// Confidential user data.
    Confidential = 2,
    /// Secret / cryptographic key data.
    Secret = 3,
    /// Cryptographically isolated hardware enclave data.
    Isolated = 4,
}

impl ConfidentialityLevel {
    /// Information flow security check: data at `self` can only flow to `target` if `self <= target`.
    #[inline(always)]
    pub fn can_flow_to(self, target: Self) -> bool {
        (self as u8) <= (target as u8)
    }
}

/// Retention policy for compiled artifacts, cached kernels, and retained resources.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize,
)]
#[serde(rename_all = "snake_case")]
pub enum RetentionPolicy {
    /// Ephemeral: zeroized and discarded immediately after kernel execution finishes.
    #[default]
    Ephemeral,
    /// Session-bound: retained across executions within the same client session.
    Session,
    /// Persistent: persisted in encrypted disk or cache storage.
    Persistent,
    /// Pinned: explicitly pinned in memory until explicit administrative eviction.
    Pinned,
}
