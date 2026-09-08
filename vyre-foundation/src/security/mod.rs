//! Security, Tenant Isolation, Capability Handles & Quota Authority (Row 119).
//!
//! Provides explicit tenant and confidentiality labels, unforgeable generation-bound
//! capability tokens, automatic zeroization on drop, and bounded hostile compilation quotas.

pub mod capability;
pub mod error;
pub mod label;
pub mod quota;
pub mod sanitize;

pub use capability::{CapabilityAuthenticator, GenerationId, UnforgeableCapability};
pub use error::SecurityError;
pub use label::{AuthorityLabel, ConfidentialityLevel, Permission, RetentionPolicy, SecurityRole, TenantId};
pub use quota::{CompilationBudgetEnforcer, CompilationQuota, RedactedDiagnostic};
pub use sanitize::{SanitizedBuffer, TenantCacheNamespace};
