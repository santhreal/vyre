//! Shared preferred-dispatch backend registry contracts for driver crates.

use vyre_driver::{acquire_preferred_dispatch_backend, backend_dispatches, backend_precedence};

/// Assert the registry reports `backend_id` as dispatching at
/// `expected_precedence`.
///
/// `LinkRegistration` is the backend's registration type. Naming it keeps the
/// backend's object file, and therefore its `inventory` submission, in the
/// linked test binary; nothing is constructed from it.
///
/// # Panics
/// Panics with `dispatch_message` when the registry does not report the backend
/// as dispatching, and with `precedence_message` when its precedence differs.
pub fn assert_backend_registry_metadata<LinkRegistration: 'static>(
    backend_id: &str,
    expected_precedence: u32,
    dispatch_message: &str,
    precedence_message: &str,
) {
    let _link_inventory_registration = std::any::TypeId::of::<LinkRegistration>();

    assert!(
        backend_dispatches(backend_id).expect(
            "valid backend registry. Fix: link the backend's registration crate so `backend_id` is \
             in the registry this assertion reads",
        ),
        "{dispatch_message}"
    );
    assert_eq!(
        backend_precedence(backend_id).expect(
            "valid backend registry. Fix: link the backend's registration crate so `backend_id` is \
             in the registry this assertion reads",
        ),
        expected_precedence,
        "{precedence_message}"
    );
}

/// Assert preferred dispatch acquires `expected_backend_id` and that the
/// acquired backend dispatches to a device.
///
/// `LinkRegistration` is the backend's registration type, named for the same
/// linking reason as in [`assert_backend_registry_metadata`].
///
/// The domain check reads
/// [`crate::backend_execution_domain`] rather than rejecting the two ids the
/// interpreter used to register under. Those two `assert_ne!` calls sat after
/// the equality above, so they could only fire for an `expected_backend_id`
/// the caller itself named, and they said nothing about a host path registered
/// under a third name.
///
/// # Panics
/// Panics with `acquisition_message` when no backend is acquired, with
/// `selection_message` when another backend wins, and when the selection
/// carries no device execution-domain decision.
pub fn assert_preferred_dispatch_selects<LinkRegistration: 'static>(
    expected_backend_id: &str,
    acquisition_message: &str,
    selection_message: &str,
) {
    let _link_inventory_registration = std::any::TypeId::of::<LinkRegistration>();
    let backend = acquire_preferred_dispatch_backend().expect(acquisition_message);
    let actual_id = backend.id();

    assert_eq!(
        actual_id, expected_backend_id,
        "{selection_message}; preferred dispatch got `{actual_id}`"
    );
    crate::backend_execution_domain::assert_dispatch_leaves_the_host(actual_id);
}
