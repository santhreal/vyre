//! Explicit failure domains and recovery classes for mutable state owners.
//!
//! WHY: closes the class "uncoordinated failure policies where poisoned locks panic,
//! recover partially mutated state, or repeat non-idempotent side effects upon retry".
//! Defines explicit failure domains, recovery classes, and typed state transitions.

use core::fmt;
use std::string::String;

/// Explicit failure domain identifying which subsystem boundary failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FailureDomain {
    /// In-memory state (caches, registries, heaps, transient buffers).
    MemoryState,
    /// Hardware device context (GPU queue, device loss, VRAM allocator).
    DeviceContext,
    /// Persistent disk journal or cache storage (disk full, I/O failure).
    DiskJournal,
    /// Supervised worker process (crash, OOM, timeout).
    WorkerProcess,
    /// Session lifecycle (connection reset, client disconnect).
    SessionLifecycle,
    /// Network and transport boundary (socket reset, framing error).
    NetworkTransport,
}

impl FailureDomain {
    /// All failure domains.
    pub const ALL: &'static [Self] = {
        const ALL: &[FailureDomain] = &[
            FailureDomain::MemoryState,
            FailureDomain::DeviceContext,
            FailureDomain::DiskJournal,
            FailureDomain::WorkerProcess,
            FailureDomain::SessionLifecycle,
            FailureDomain::NetworkTransport,
        ];
        let mut i = 0;
        while i < ALL.len() {
            match ALL[i] {
                FailureDomain::MemoryState
                | FailureDomain::DeviceContext
                | FailureDomain::DiskJournal
                | FailureDomain::WorkerProcess
                | FailureDomain::SessionLifecycle
                | FailureDomain::NetworkTransport => {}
            }
            i += 1;
        }
        ALL
    };

    /// Stable string identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MemoryState => "memory_state",
            Self::DeviceContext => "device_context",
            Self::DiskJournal => "disk_journal",
            Self::WorkerProcess => "worker_process",
            Self::SessionLifecycle => "session_lifecycle",
            Self::NetworkTransport => "network_transport",
        }
    }
}

impl fmt::Display for FailureDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Recovery class defining how a failure can be safely remediated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RecoveryClass {
    /// Side effect is rolled back via prepare/commit journal and idempotency key.
    TransactionallyRecoverable,
    /// State can be rebuilt completely from immutable canonical persisted input.
    RestartableFromCanonicalInput,
    /// Hardware device context poisoned; requires releasing handles and acquiring fresh device.
    DeviceContextFatal,
    /// Worker process died or leaked; requires terminating and respawning under supervised budget.
    ProcessFatal,
    /// Compiler or engine internal invariant violated; unrecoverable defect.
    InvariantViolation,
}

impl RecoveryClass {
    /// All recovery classes.
    pub const ALL: &'static [Self] = {
        const ALL: &[RecoveryClass] = &[
            RecoveryClass::TransactionallyRecoverable,
            RecoveryClass::RestartableFromCanonicalInput,
            RecoveryClass::DeviceContextFatal,
            RecoveryClass::ProcessFatal,
            RecoveryClass::InvariantViolation,
        ];
        let mut i = 0;
        while i < ALL.len() {
            match ALL[i] {
                RecoveryClass::TransactionallyRecoverable
                | RecoveryClass::RestartableFromCanonicalInput
                | RecoveryClass::DeviceContextFatal
                | RecoveryClass::ProcessFatal
                | RecoveryClass::InvariantViolation => {}
            }
            i += 1;
        }
        ALL
    };

    /// Stable string identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TransactionallyRecoverable => "transactionally_recoverable",
            Self::RestartableFromCanonicalInput => "restartable_from_canonical_input",
            Self::DeviceContextFatal => "device_context_fatal",
            Self::ProcessFatal => "process_fatal",
            Self::InvariantViolation => "invariant_violation",
        }
    }

    /// Whether this recovery class allows retry without process/device restart.
    #[must_use]
    pub const fn allows_inline_recovery(self) -> bool {
        match self {
            Self::TransactionallyRecoverable | Self::RestartableFromCanonicalInput => true,
            Self::DeviceContextFatal | Self::ProcessFatal | Self::InvariantViolation => false,
        }
    }
}

impl fmt::Display for RecoveryClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Actionable remediation disposition for a failure event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryDisposition {
    /// Safe to retry immediately using the same idempotency key.
    CanRetryImmediate,
    /// Rebuilding state from immutable input is required before subsequent operations.
    RequiresRebuild,
    /// Must discard existing device context and acquire a new device.
    RequiresDeviceReacquisition,
    /// Worker process must be restarted under supervised restart budget.
    RequiresProcessRestart,
    /// Fatal error; cannot be recovered.
    Fatal,
}

/// Structured error carrying failure domain, recovery class, and documented corrective action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedRecoveryError {
    /// Owning failure domain.
    pub domain: FailureDomain,
    /// Recovery class.
    pub recovery_class: RecoveryClass,
    /// Actionable disposition.
    pub disposition: RecoveryDisposition,
    /// Detailed diagnostic reason.
    pub reason: String,
    /// Documented corrective action for caller.
    pub fix: String,
}

impl fmt::Display for TypedRecoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] ({}) {}. {}",
            self.domain, self.recovery_class, self.reason, self.fix
        )
    }
}

impl core::error::Error for TypedRecoveryError {}

impl TypedRecoveryError {
    /// Create a new typed recovery error with explicit failure domain and recovery class.
    #[must_use]
    pub fn new(
        domain: FailureDomain,
        recovery_class: RecoveryClass,
        disposition: RecoveryDisposition,
        reason: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self {
            domain,
            recovery_class,
            disposition,
            reason: reason.into(),
            fix: fix.into(),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_domain_all_contains_every_variant() {
        for domain in FailureDomain::ALL {
            match domain {
                FailureDomain::MemoryState
                | FailureDomain::DeviceContext
                | FailureDomain::DiskJournal
                | FailureDomain::WorkerProcess
                | FailureDomain::SessionLifecycle
                | FailureDomain::NetworkTransport => {}
            }
        }
        assert_eq!(FailureDomain::ALL.len(), 6);
    }

    #[test]
    fn recovery_class_all_contains_every_variant() {
        for class in RecoveryClass::ALL {
            match class {
                RecoveryClass::TransactionallyRecoverable
                | RecoveryClass::RestartableFromCanonicalInput
                | RecoveryClass::DeviceContextFatal
                | RecoveryClass::ProcessFatal
                | RecoveryClass::InvariantViolation => {}
            }
        }
        assert_eq!(RecoveryClass::ALL.len(), 5);
    }
}
