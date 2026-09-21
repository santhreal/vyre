//! Opaque target selection for canonical compiler envelopes.

pub use vyre_foundation::operation::TargetId;

/// Resolve a linked target registration by its validated opaque identity.
///
/// # Errors
///
/// Returns the driver registry error when no linked concrete target owns `target`.
pub fn registration(
    target: &TargetId,
) -> Result<&'static vyre_driver::BackendRegistration, vyre_driver::BackendError> {
    vyre_driver::registered_backends()?
        .iter()
        .find(|registration| registration.target_id.as_str() == target.as_str())
        .ok_or_else(|| {
            vyre_driver::BackendError::new(format!(
                "target `{}` is not linked into this binary. Fix: link the concrete driver crate that owns this target or choose one of the registered target ids.",
                target.as_str()
            ))
        })
}
