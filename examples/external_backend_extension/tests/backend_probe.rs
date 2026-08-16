//! WHY: the backend registry is the only way an out-of-tree crate reaches
//! dispatch, and the seal on [`vyre_driver::VyreBackend`] plus the inventory
//! section it is submitted through are both invisible until something outside
//! the workspace uses them. Every assertion below fails if the published
//! surface stops admitting an external backend: the trait becomes
//! unimplementable, the registration stops reaching the registry, or `acquire`
//! stops serving it. It does not cover device dispatch, which no host has.

use external_backend_extension::{dispatch_probe, BACKEND_ID};

#[test]
fn an_out_of_tree_registration_reaches_the_registry() {
    let registrations =
        vyre_driver::registered_backends().expect("the backend registry must be valid");
    let registration = registrations
        .iter()
        .find(|registration| registration.id == BACKEND_ID)
        .expect("this crate submits a BackendRegistration from outside the workspace");

    assert_eq!(registration.target_id.as_str(), BACKEND_ID);
    assert!(registration.reference_oracle);
    assert!(
        vyre_driver::backend_dispatches(BACKEND_ID)
            .expect("a registered backend declares its dispatch capability")
    );
}

#[test]
fn acquire_serves_the_external_backend() {
    let backend = vyre_driver::acquire(BACKEND_ID).expect("the registered factory must construct");

    assert_eq!(backend.id(), BACKEND_ID);
    assert_eq!(backend.version(), env!("CARGO_PKG_VERSION"));
    assert!(!backend.supported_ops().is_empty());
}

#[test]
fn acquire_rejects_an_unregistered_id() {
    let error = match vyre_driver::acquire("example-external-backend-that-is-not-linked") {
        Ok(_) => panic!("an unregistered id must not resolve to this backend"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("Fix:"),
        "a registry error states the corrective action: {error}"
    );
}

#[test]
fn the_external_backend_dispatches_through_the_public_api() {
    let words = dispatch_probe(&[1, 2, 3, 4]).expect("the probe program must dispatch");

    assert_eq!(words, vec![2, 3, 4, 5]);
}

#[test]
fn a_row_count_that_disagrees_with_the_program_is_rejected() {
    let backend = vyre_driver::acquire(BACKEND_ID).expect("the registered factory must construct");
    let program = external_backend_extension::build_probe_program();
    let single = vec![0u8; 16];

    let error = backend
        .dispatch_borrowed(
            &program,
            &[single.as_slice()],
            &vyre_driver::DispatchConfig::default(),
        )
        .expect_err("one row for two bound buffers must be rejected");

    assert!(
        error.to_string().contains("Fix:"),
        "the row-count error states the corrective action: {error}"
    );
}
