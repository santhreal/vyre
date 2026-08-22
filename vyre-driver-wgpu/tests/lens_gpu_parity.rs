//! GPU-side reference parity over the canonical library operation view.

#![cfg(feature = "device-tests")]

use vyre_conform::lens::backend_parity;
use vyre_conform::lens::outcome::LensOutcome;
fn backend() -> &'static vyre_driver::BackendRegistration {
    vyre_driver::backend_registration(vyre_driver_wgpu::WGPU_BACKEND_ID)
        .expect("Fix: WGPU driver crate must register its authenticated artifact target.")
}

#[test]
fn cpu_vs_backend_lens_every_eligible_op() {
    let be = backend();
    let mut failures: Vec<String> = Vec::new();
    let mut passed = 0usize;
    for entry in vyre_libs::operation_catalog::fixture_entries() {
        match backend_parity::run(&entry, be) {
            LensOutcome::Pass { cases } => {
                passed += 1;
                println!("  pass {} ({cases} cases)", entry.id);
            }
            LensOutcome::Fail { case_index, detail } => {
                failures.push(format!("{} case {case_index}: {detail}", entry.id));
            }
        }
    }
    println!(
        "cpu_vs_backend: {passed} passed, 0 coverage gaps, {} failed",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "cpu_vs_backend lens failures:\n  - {}",
        failures.join("\n  - ")
    );
}
