//! Backend-neutral resource handles.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::accounting::checked_atomic_next_u64_with_order;
use crate::backend::error::BackendError;

/// Process-wide source of backend instance identities.
///
/// Starts at 1 so zero is never a valid owner, and only ever counts up, so a
/// backend that is dropped and recreated in the same process receives a
/// distinct identity rather than inheriting the dead one's namespace.
static NEXT_RESIDENT_OWNER: AtomicU64 = AtomicU64::new(1);

/// Identity of one backend instance's resident buffer namespace.
///
/// Resident buffer ids come from a counter private to a backend instance, so
/// two live instances hand out the same ids for unrelated device memory. An
/// owner makes those namespaces distinguishable, which is what lets a handle
/// be checked instead of merely trusted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct ResidentOwner(u64);

impl ResidentOwner {
    /// Mint an identity for a new backend instance.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidProgram`] if the process has exhausted
    /// the owner id space, which is refused rather than wrapped: a wrapped id
    /// would silently authorize a foreign handle.
    pub fn new() -> Result<Self, BackendError> {
        let id = checked_atomic_next_u64_with_order(
            &NEXT_RESIDENT_OWNER,
            Ordering::Acquire,
            Ordering::AcqRel,
            Ordering::Acquire,
            |_| {
                BackendError::InvalidProgram {
                fix: "Fix: the process exhausted backend instance identities for resident buffers. Restart the process instead of reusing an identity, which would let a stale resident handle resolve against a live buffer."
                    .to_string(),
            }
            },
        )?;
        Ok(Self(id))
    }

    /// Raw identity value, for diagnostics and stable ordering.
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }

    /// Mint a handle in this owner's namespace.
    #[must_use]
    pub fn handle(self, id: u64) -> ResidentHandle {
        ResidentHandle { owner: self, id }
    }

    /// Unwrap a handle minted by this owner, refusing a foreign one.
    ///
    /// `context` names the operation for the error message, for example
    /// `"resident upload"`.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidProgram`] if `handle` was minted by a
    /// different backend instance. Resolving it by bare id instead would find
    /// this instance's unrelated buffer of the same id and read or write the
    /// wrong device memory without any diagnostic.
    pub fn resolve(self, handle: ResidentHandle, context: &str) -> Result<u64, BackendError> {
        if handle.owner != self {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: {context} received resident handle {} owned by backend instance {}, but this instance is {}. A resident handle is only valid on the backend instance that allocated it; reallocate and re-upload the buffer on this instance, or keep the original instance alive for as long as the handle is held.",
                    handle.id,
                    handle.owner.0,
                    self.0
                ),
            });
        }
        Ok(handle.id)
    }
}

/// A resident buffer handle that names both the buffer and the backend
/// instance that owns it.
///
/// Carrying the owner is what makes presenting a foreign handle a refusal at
/// the API boundary rather than a silent resolve against unrelated device
/// memory, so a caller can hold a handle across instances without having to
/// remember a check.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ResidentHandle {
    owner: ResidentOwner,
    id: u64,
}

impl ResidentHandle {
    /// Backend instance that allocated this buffer.
    #[must_use]
    pub fn owner(self) -> ResidentOwner {
        self.owner
    }

    /// Buffer id within its owner's namespace.
    ///
    /// Only meaningful together with [`ResidentHandle::owner`]; use
    /// [`ResidentOwner::resolve`] to obtain it for a lookup.
    #[must_use]
    pub fn id(self) -> u64 {
        self.id
    }
}

impl std::fmt::Display for ResidentHandle {
    /// Prints the owning instance alongside the id, never the id alone.
    ///
    /// A bare id is precisely the identifier that was unsafe to trust: two
    /// live instances each have a buffer 3, so a log line or error naming
    /// only `3` cannot be acted on.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} on backend instance {}", self.id, self.owner.0)
    }
}

/// A GPU-resident or host-side resource used as an input to a Program.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Resource {
    /// Host-side byte slice. Replicated to the GPU on each dispatch.
    Borrowed(Vec<u8>),
    /// GPU-resident buffer handle. Zero-copy; no host transfer occurs.
    Resident(ResidentHandle),
}

impl Default for Resource {
    fn default() -> Self {
        Resource::Borrowed(Vec::new())
    }
}

impl From<Vec<u8>> for Resource {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Borrowed(bytes)
    }
}

impl From<ResidentHandle> for Resource {
    fn from(handle: ResidentHandle) -> Self {
        Self::Resident(handle)
    }
}

// Inline: `vyre_driver::backend` is `pub(crate)`, so no integration test can reach what this suite
// exercises.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_owners_refuse_each_others_handles() {
        let first = ResidentOwner::new().expect("owner ids are available");
        let second = ResidentOwner::new().expect("owner ids are available");
        assert_ne!(first, second);

        let handle = first.handle(7);
        assert_eq!(
            first.resolve(handle, "test resolve").expect("own handle"),
            7
        );

        let error = second.resolve(handle, "test resolve").expect_err(
            "Fix: a foreign resident handle must be refused, never resolved by bare id",
        );
        let BackendError::InvalidProgram { fix } = error else {
            panic!("Fix: foreign resident handle refusal must be BackendError::InvalidProgram");
        };
        assert!(
            fix.contains("owned by backend instance") && fix.contains("Fix: "),
            "Fix: foreign-handle refusal must name the owning instance and carry actionable text, got {fix}"
        );
    }

    #[test]
    fn same_id_in_two_namespaces_stays_distinct() {
        let first = ResidentOwner::new().expect("owner ids are available");
        let second = ResidentOwner::new().expect("owner ids are available");
        assert_ne!(first.handle(1), second.handle(1));
        assert_eq!(first.handle(1), first.handle(1));
    }
}
