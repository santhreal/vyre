//! Production conformance step bounds.
//!
//! The routes that compile and submit an artifact on a device live in
//! `production_route_device.rs`. This target holds the contract that needs no
//! hardware, so it runs on every leg.

use std::time::{Duration, Instant};

use vyre_conform::production::{run_bounded_step, ProductionError};
use vyre_registry_link::backend::live_backend_registry;

/// Operation identity the bounded step carries, so a bound that expires names
/// something a reader can find.
const LIFECYCLE_OP_ID: &str = "vyre-conform::production_route::session_lifecycle";

/// A step that never returns is reported, not awaited.
///
/// The bound is the whole mechanism the lifecycle test rests on, so it is
/// exercised against work that is guaranteed never to finish: without the
/// deadline this test would hang, which is the failure it exists to convert into
/// a named error.
#[test]
fn a_bounded_step_that_never_returns_is_reported_with_its_operation_and_backend() {
    let backend = live_backend_registry()
        .expect("valid backend registry")
        .iter()
        .map(|registration| registration.id)
        .next()
        .expect("Fix: at least one backend must be linked to name in a bounded step.");
    let deadline = Duration::from_millis(50);
    let started = Instant::now();
    let error =
        run_bounded_step::<()>("wedged step", LIFECYCLE_OP_ID, backend, deadline, || loop {
            std::thread::park();
        })
        .expect_err(
            "Fix: a bounded step whose work never returns must fail, not block the caller.",
        );

    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(5),
        "Fix: a bounded step must return once its deadline elapses; this one took {elapsed:?} for a {deadline:?} ceiling."
    );
    match error {
        ProductionError::Deadline {
            step,
            op_id,
            backend: reported_backend,
            deadline: reported_deadline,
        } => {
            assert_eq!(step, "wedged step");
            assert_eq!(op_id, LIFECYCLE_OP_ID);
            assert_eq!(reported_backend, backend);
            assert_eq!(reported_deadline, deadline);
        }
        other => panic!(
            "Fix: an expired bounded step must report ProductionError::Deadline, got {other:?}."
        ),
    }
}
