//! What the backend registry reports when nothing is registered.
//!
//! `vyre-driver` depends on no concrete driver, so its own test binary sees an
//! empty registry. That is the interesting case: every query has to answer
//! conservatively and acquisition has to fail loudly instead of substituting a
//! host execution path.

use vyre_driver::backend::{
    acquire_preferred_dispatch_backend, backend_dispatches, backend_precedence,
    registered_backends, registered_backends_by_precedence, VyreBackend,
};

#[test]
fn an_unregistered_backend_reports_no_dispatch_capability() {
    assert!(
        !backend_dispatches("nonexistent-backend").expect("the empty registry must be readable"),
        "Fix: a backend nobody registered must report dispatches = false."
    );
}

#[test]
fn an_unregistered_backend_gets_the_lowest_precedence() {
    assert_eq!(
        backend_precedence("nonexistent-backend").expect("the empty registry must be readable"),
        u32::MAX,
        "Fix: a backend nobody registered must sort last, never ahead of a real one."
    );
}

#[test]
fn an_empty_registry_reports_no_backends_through_either_view() {
    assert!(
        registered_backends()
            .expect("the empty registry must be readable")
            .is_empty(),
        "Fix: vyre-driver alone links no concrete driver and must see zero backends."
    );
    assert!(
        registered_backends_by_precedence()
            .expect("the empty registry must be readable")
            .is_empty(),
        "Fix: the precedence-sorted view must agree with the unsorted one on an empty registry."
    );
}

/// Discovery freezes once. Rebuilding per query would leak the owned
/// registration storage, so repeated reads must hand back the same allocation.
/// This compares the slice address, not allocator internals.
#[test]
fn repeated_registry_queries_return_one_immutable_allocation() {
    let first = registered_backends().expect("the empty registry must be readable");
    for _ in 0..1_024 {
        let current = registered_backends().expect("the empty registry must be readable");
        assert_eq!(current.as_ptr(), first.as_ptr());
        assert_eq!(current.len(), first.len());
    }
}

#[test]
fn preferred_dispatch_acquisition_fails_loudly_without_a_host_fallback() {
    let error = match acquire_preferred_dispatch_backend() {
        Ok(backend) => panic!(
            "Fix: vyre-driver alone must acquire no backend, and it acquired `{}`.",
            backend.id()
        ),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("no usable GPU dispatch backend is available"),
        "Fix: acquisition failure must name the missing GPU dispatch backend: {message}"
    );
    assert!(
        message.contains("repair the GPU driver probe"),
        "Fix: acquisition failure must tell an operator to repair the GPU probe: {message}"
    );
    let lowered = message.to_lowercase();
    assert!(
        !lowered.contains("fallback") && !lowered.contains("falling back"),
        "Fix: acquisition must never advertise a fallback; a target that cannot execute fails \
         closed: {message}"
    );
}
