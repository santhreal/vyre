//! Registration contract for the pure Rust reference backend adapter.

use vyre_driver::{acquire, backend_dispatches};
use vyre_foundation::ir::{Expr, Node, Program};

mod dispatch_fixtures;
use dispatch_fixtures::u32_out_buffer;

/// The backend id comes from [`vyre_driver_reference::registered_backend_id`]
/// and not from the `const`, because calling it is what keeps this crate's
/// object file, and its registration, in the linked test binary. Reading the
/// `const` inlines at the use site and links nothing, which left both tests
/// reporting an unregistered backend on the Mach-O leg of the matrix while the
/// ELF legs passed.
fn registered_id() -> &'static str {
    vyre_driver_reference::registered_backend_id()
        .expect("Fix: this build must compile the cpu-ref registration.")
}

#[test]
fn cpu_ref_registers_as_dispatch_backend() {
    let id = registered_id();
    assert!(
        backend_dispatches(id).expect("valid backend registry"),
        "Fix: vyre-driver-reference must register cpu-ref as a dispatch-capable backend."
    );

    let backend = acquire(id)
        .expect("Fix: cpu-ref backend registration must construct without host hardware.");
    let program = Program::wrapped(
        vec![u32_out_buffer("out", 0)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    );
    let outputs = backend
        .dispatch(&program, &[], &vyre_driver::DispatchConfig::default())
        .expect("Fix: cpu-ref backend must dispatch a minimal Program.");

    assert_eq!(
        outputs,
        vec![42u32.to_le_bytes().to_vec()],
        "Fix: cpu-ref backend output must match reference interpreter bytes."
    );
}

#[test]
fn cpu_ref_rejects_cooperative_dispatch() {
    let id = registered_id();
    let backend = acquire(id)
        .expect("Fix: cpu-ref backend registration must construct without host hardware.");
    let program = Program::wrapped(
        vec![u32_out_buffer("out", 0)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    );
    let mut config = vyre_driver::DispatchConfig::default();
    config.cooperative = true;
    let error = backend.dispatch(&program, &[], &config).expect_err(
        "Fix: cpu-ref backend must reject cooperative dispatch with UnsupportedFeature",
    );
    match error {
        vyre_driver::BackendError::UnsupportedFeature { name, backend } => {
            assert!(name.contains("cooperative"), "got {name}");
            assert_eq!(backend, id);
        }
        other => panic!("expected UnsupportedFeature, got {other:?}"),
    }
}
